
import zmq
import struct
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import numpy as np
from collections import deque
from processframe import FrameProcessor
from scipy.io import savemat
import os
# 线程A：读取图像帧
def frame_producer():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://127.0.0.1:64179")

    # Subscribe to all topics
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    print("Listening for published messages...")

    metadata = None
    current_frame_number = 0
    while True:
        if pause_event.is_set():
            time.sleep(0.05)
            continue
        message = socket.recv()
        if len(message) == 40:
            # print("Received header")
            metadata = message
            # Unpack 5 doubles: 'd' = double (8 bytes), so '5d' = 5 doubles
            pixels_per_line, lines_per_frame, num_channels, timestamp, frame_number = struct.unpack('5d', metadata)
        if len(message) > 40:
            frame = message
            # 为了匹配速度，每4帧才appedpend一次
            if current_frame_number % 4 != 0:
                current_frame_number += 1
                continue
            frame_buffer.append(np.frombuffer(frame, dtype=np.int16).reshape(int(pixels_per_line), 
                                                        int(lines_per_frame), 
                                                        int(num_channels), 
                                                        order = 'F'))
            print("Received Frame ", current_frame_number//4)
            # print("average pixel value: ", np.mean(np.frombuffer(frame, dtype=np.int16)))
            current_frame_number = current_frame_number + 1
        if current_frame_number > 10000: 
            break # or do keyboard interrupt once done
        time.sleep(0.01)

# 线程B：处理图像帧
def frame_processor(Params_post, filename_CNN, p):

    # image enhancement
    trt_path = 'Z:\LSR\DATA\checkpt\RAFTCAD_result_multiscale_scale_10_stack_28_50mW_fton10mW\DeepIE_tensorRT_windows.trt'
    cnn_path = filename_CNN

    # cuda.init()
    # 初始化
    while True:
        with buffer_lock:
            if len(frame_buffer) >= FRAME_BUFFER_SIZE:
                print("开始初始化")
                pause_event.set()  # 暂停producer
                frame_list_afteradjust = []
                trace_list = []
                # fake a frame buffer
                import tifffile as tiff
                mov = tiff.imread('mov.tiff').astype(np.float32)
                frame_buffer_fake = mov[:]
                # init FrameProcessor
                frame_processor = FrameProcessor(trt_path, cnn_path, 0, Params_post)
                # 初始化处理
                video_adjust, template, Masks, traces = frame_processor.process_frames_init(np.squeeze(np.array(frame_buffer_fake)),\
                                                                             batch_size=BATCH_SIZE,
                                                                            overlap_size=OVERLAP_SIZE)
                # 将处理后的帧存入缓冲区
                for frame in video_adjust:
                    frame_list_afteradjust.append(frame)

                # 将处理后的trace存入缓冲区
                for trace in traces:
                    trace_list.append(trace)
                    
                # 保存处理后的帧
                tiff.imwrite('Output_Frames.tiff', np.array(frame_list_afteradjust, dtype=np.float32))

                # 保存Mask
                savemat('Output_Masks.mat', {'Masks':Masks}, do_compression=True)

                # 保存trace
                savemat('Output_Trace.mat', {'traces': traces}, do_compression=True)

                # 清空frame_buffer
                frame_buffer.clear()

                break
        time.sleep(0.1)

    print("初始化完成")
    # 在线处理
    pause_event.clear()  # 恢复producer
    frame_list_afteradjust = []
    embedding_list_afteradjust = []
    history = []
    starttime = time.time()
    while True:
        with buffer_lock:
            if len(history) < OVERLAP_SIZE and len(frame_buffer) >= OVERLAP_SIZE:
                # 如果历史帧不足，先取 overlap 区域的帧
                history = [frame_buffer.popleft() for _ in range(OVERLAP_SIZE)]
            if len(frame_buffer) >= BATCH_SIZE + OVERLAP_SIZE * 2:
                # 取新 batch
                new_batch = [frame_buffer.popleft() for _ in range(BATCH_SIZE + OVERLAP_SIZE)]
                # 拼接历史 overlap 区域
                frames_to_process = history + new_batch
                # 更新 history
                history = frames_to_process[-OVERLAP_SIZE:]

                # frames_to_process = list(frame_buffer)[idx:idx + BATCH_SIZE + OVERLAP_SIZE * 2]
                # 在线处理
                frames_to_process_fake = mov[500:500 + BATCH_SIZE + OVERLAP_SIZE * 2]
                video_adjust, traces, embeddings = frame_processor.process_frames_online(np.squeeze(np.array(frames_to_process_fake)), template, 
                                                            batch_size=BATCH_SIZE,
                                                            overlap_size=OVERLAP_SIZE, Masks=Masks)
                # 将处理后的帧存入缓冲区
                for frame in video_adjust:
                    frame_list_afteradjust.append(frame)
                # 将处理后的嵌入存入缓冲区
                for embedding in embeddings:
                    embedding_list_afteradjust.append(embedding)
                # 将处理后的 trace 存入缓冲区
                for trace in traces:
                    trace_list.append(trace)
            else:
                # time.sleep(0.01)
                continue
            endtime = time.time()
            print(f"Processed {len(frame_list_afteradjust)} frames, time taken: {endtime - starttime:.2f} seconds")

if __name__ == "__main__":
    # 全局参数
    FRAME_HEIGHT = 512
    FRAME_WIDTH = 512
    FRAME_BUFFER_SIZE = 100
    BATCH_SIZE = 8
    OVERLAP_SIZE = 2

    Params_post={
                # minimum area of a neuron (unit: pixels).
                'minArea': 60, 
                # average area of a typical neuron (unit: pixels) 
                'avgArea': 100,
                # uint8 threshould of probablity map (uint8 variable, = float probablity * 256 - 1)
                'thresh_pmap': 150, 
                # values higher than "thresh_mask" times the maximum value of the mask are set to one.
                'thresh_mask': 0.4, 
                # maximum COM distance of two masks to be considered the same neuron in the initial merging (unit: pixels)
                'thresh_COM0': 2, 
                # maximum COM distance of two masks to be considered the same neuron (unit: pixels)
                'thresh_COM': 8, 
                # minimum IoU of two masks to be considered the same neuron
                'thresh_IOU': 0.6, 
                # minimum consume ratio of two masks to be considered the same neuron 
                'thresh_consume': 0.7, 
                # minimum consecutive number of frames of active neurons
                'cons': 4}
    filename_CNN = 'Z:\LSR\DATA\\2p_bench\suns\\0701\Weights\model_latest.pth'
    
    pause_event = threading.Event()

    # 使用 deque 实现共享缓冲区（线程安全操作要加锁）
    frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE + 1000)
    buffer_lock = threading.Lock()

    # 启动线程
    producer_thread = threading.Thread(target=frame_producer, daemon=True)
    processor_thread = threading.Thread(
        target=frame_processor,
        args=(Params_post, filename_CNN, None),
        daemon=True
    )

    producer_thread.start()
    processor_thread.start()

    # 保持主线程运行
    while True:
        time.sleep(1)








