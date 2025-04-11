import argparse
import os
import pickle as pk
import time
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
from torch.cuda.amp import autocast
from pysmpl.pysmpl import PySMPL
from utils.vis_tools import Visualizer
from ultralytics import YOLO

INPUT_SHAPE = (256, 256)
DETECTION_THRESH = 0.5

class TRTInference:
    def __init__(self, engine_path):
        if not cuda.Context.get_current():
            pycuda.autoinit.context.push()

        self.logger = trt.Logger(trt.Logger.INTERNAL_ERROR)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Initialize bindings and allocate memory
        self.bindings = []
        self.binding_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_bindings)]
        self.input_binding_index = None
        self.input_dtype = None
        self.input_shape = None
        self.output_binding_indices = []
        self.output_shapes = []
        self.output_dtypes = []

        for i in range(self.engine.num_bindings):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.binding_is_input(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(self.engine.get_tensor_shape(name))  # Fixed shapes now
            size = trt.volume(shape) * np.dtype(dtype).itemsize
            if size <= 0:
                raise ValueError(f"Invalid shape {shape} for binding {name}")
            device_mem = cuda.mem_alloc(size)
            self.bindings.append(device_mem)

            if is_input:
                self.input_binding_index = i
                self.input_dtype = dtype
                self.input_shape = shape
            else:
                self.output_binding_indices.append(i)
                self.output_shapes.append(shape)
                self.output_dtypes.append(dtype)

    def __call__(self, input_tensor):
        input_np = input_tensor.cpu().numpy().astype(self.input_dtype)
        cuda.memcpy_htod_async(self.bindings[self.input_binding_index], input_np, self.stream)

        self.context.execute_async_v2(bindings=[int(b) for b in self.bindings], stream_handle=self.stream.handle)

        outputs = []
        for idx, shape, dtype in zip(self.output_binding_indices, self.output_shapes, self.output_dtypes):
            out_np = np.empty(shape, dtype=dtype)
            cuda.memcpy_dtoh_async(out_np, self.bindings[idx], self.stream)
            outputs.append(out_np)

        self.stream.synchronize()
        return [torch.from_numpy(out).cuda() for out in outputs]

def preprocess_frame_pinned(frame, pinned_buffer, bbox=None):
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        frame = frame[y1:y2, x1:x2]

    frame_resized = cv2.resize(frame, INPUT_SHAPE)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_fp16 = frame_rgb.transpose(2, 0, 1).astype(np.float16) / np.float16(255.0)
    np.copyto(pinned_buffer, frame_fp16)
    return torch.from_numpy(pinned_buffer).unsqueeze(0).cuda(non_blocking=True)

def main():
    parser = argparse.ArgumentParser(description="HybrikTensorRT Optimized")
    parser.add_argument("--video-name", default="examples/dance.mp4")
    parser.add_argument("--engine-path", default="engines/hybrik.engine")
    parser.add_argument("--detector-engine", default="engines/yolo11n.engine")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--save-pk", action="store_true")
    parser.add_argument("--save-img", action="store_true")
    parser.add_argument("--show-mesh", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    img_dir = os.path.join(args.out_dir, "images") if args.save_img else None
    pkl_dir = os.path.join(args.out_dir, "pkl") if args.save_pk else None
    if img_dir:
        os.makedirs(img_dir, exist_ok=True)
    if pkl_dir:
        os.makedirs(pkl_dir, exist_ok=True)

    pose_model = TRTInference(args.engine_path)
    vis = Visualizer() if args.show_mesh else None
    smpl = PySMPL().cuda().float() if args.show_mesh else None
    output_list = []
    detection_model = YOLO(args.detector_engine, task="detect")

    cap = cv2.VideoCapture(args.video_name)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video_name}")

    pinned_buffer = cuda.pagelocked_empty((3, *INPUT_SHAPE[::-1]), dtype=np.float16)

    frame_count, start_time = 0, time.time()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            with autocast():
                results = detection_model(frame, verbose=False, half=True)

                boxes = results[0].boxes.xyxy
                confs = results[0].boxes.conf
                cls = results[0].boxes.cls
                mask = (cls == 0) & (confs > DETECTION_THRESH)
                if not mask.any():
                    continue

                best_idx = torch.argmax(confs[mask])
                best_box = boxes[mask][best_idx].int()

                pose_input = preprocess_frame_pinned(frame, pinned_buffer, best_box)

                shape_output, pose_output = pose_model(pose_input)

                if args.show_mesh:
                    pose_batch = pose_output.unsqueeze(0)
                    mesh = smpl(shape_output, pose_batch, pose2rot=False)
                    # vis.show_points([mesh])

                if args.save_img:
                    cv2.rectangle(frame, tuple(best_box[:2].tolist()), tuple(best_box[2:].tolist()), (0, 255, 0), 2)
                    cv2.imwrite(os.path.join(img_dir, f"{frame_count:06d}.jpg"), frame)

                if args.save_pk:
                    output_list.append(
                        {
                            "frame_idx": frame_count,
                            "bbox": best_box.cpu().numpy(),
                            "pose": pose_output.cpu().numpy(),
                            "shape": shape_output.cpu().numpy(),
                        }
                    )

                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Processed frame {frame_count}", end="\r")

    finally:
        end_time = time.time()
        print("\nFPS:", frame_count / (end_time - start_time))
        cap.release()
        if args.save_pk:
            with open(os.path.join(pkl_dir, "results.pkl"), "wb") as f:
                pk.dump(output_list, f)
        print(f"Results saved to {args.out_dir}")

if __name__ == "__main__":
    torch.cuda.init()
    main()