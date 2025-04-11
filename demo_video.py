"""Video demo script for HybrikTensorRT."""
import argparse
import os
import pickle as pk

import cv2
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from utils.hybrik_inference import Hybrik
from utils.pre_processings import bbox_xyxy2cs

det_transform = T.Compose([T.ToTensor()])


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def get_video_info(in_file):
    stream = cv2.VideoCapture(in_file)
    assert stream.isOpened(), 'Cannot capture source'
    # self.path = input_source
    datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
    fps = stream.get(cv2.CAP_PROP_FPS)
    frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # bitrate = int(stream.get(cv2.CAP_PROP_BITRATE))
    videoinfo = {'fourcc': fourcc, 'fps': fps, 'frameSize': frameSize}
    stream.release()

    return stream, videoinfo, datalen


def recognize_video_ext(ext=''):
    if ext == 'mp4':
        return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
    elif ext == 'avi':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    elif ext == 'mov':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    else:
        print("Unknown video format {}, will use .mp4 instead of it".format(ext))
        return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'


parser = argparse.ArgumentParser(description='HybrikTensorRT Demo')

parser.add_argument('--video-name', type=str, help='video path')
parser.add_argument('--out-dir', default='output', type=str, help='output directory')
parser.add_argument('--save-pk', action='store_true', help='save pkl')
parser.add_argument('--save-img', action='store_true', help='save imgs')
parser.add_argument('--engine-path', default='hybrik.engine', type=str, help='TensorRT engine path')
parser.add_argument('--model-input-size', default=(256, 256), type=tuple, help='model input size')
parser.add_argument('--device', default='cuda', type=str, help='device to use')

args = parser.parse_args()

det_model = fasterrcnn_resnet50_fpn(pretrained=True)
det_model.cuda().eval()

pose_model = Hybrik(
    args.engine_path, 
    model_input_size=args.model_input_size, 
    device=args.device
)

video_name = args.video_name
video_basename = os.path.basename(video_name).split('.')[0]

stream, videoinfo, datalen = get_video_info(video_name)
stream = cv2.VideoCapture(video_name)

fourcc, outext = recognize_video_ext(video_name.split('.')[-1])

out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

if args.save_img:
    img_path = os.path.join(out_dir, 'image')
    os.makedirs(img_path, exist_ok=True)

if args.save_pk:
    pkl_path = os.path.join(out_dir, 'pkl')
    os.makedirs(pkl_path, exist_ok=True)

# writer = cv2.VideoWriter(
#     os.path.join(out_dir, video_basename + outext),
#     fourcc,
#     videoinfo['fps'],
#     videoinfo['frameSize']
# )

# writer_3d = cv2.VideoWriter(
#     os.path.join(out_dir, video_basename + '_3d' + outext),
#     fourcc,
#     videoinfo['fps'],
#     videoinfo['frameSize']
# )

# Setup output video
fps_time = 0
frame_idx = 0

output_list = []

# Process Video
while True:
    ret, frame = stream.read()
    if not ret:
        break

    frame_idx += 1
    image_bgr = frame.copy()
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect humans in the image
    image_tensor = det_transform(image_rgb.copy()).cuda()
    with torch.no_grad():
        det_out = det_model([image_tensor])[0]

    # Get detections
    boxes = det_out['boxes'].cpu().numpy()
    scores = det_out['scores'].cpu().numpy()
    labels = det_out['labels'].cpu().numpy()

    # Filter for person class (label 1) with high confidence
    person_indices = np.where((labels == 1) & (scores > 0.9))[0]
    
    if len(person_indices) == 0:
        print("No person detected in frame", frame_idx)
        continue
    
    best_idx = scores[person_indices].argmax()
    bbox = boxes[person_indices[best_idx]]
    
    # Convert to numpy array for the model
    bbox_np = np.array([bbox])
    
    # Run HybrikTensorRT inference
    shape, pose = pose_model(image_rgb, bbox_np)
    
    # Create output dict to save
    output = {
        'frame_idx': frame_idx,
        'bbox': bbox,
        'shape': shape,
        'pose': pose
    }
    output_list.append(output)

    # Visualize and save results
    if args.save_img:
        # Draw bbox
        cv2.rectangle(
            image_bgr,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color=(0, 255, 0),
            thickness=2
        )
        
        # Save visualization image
        save_img_path = os.path.join(img_path, f"{frame_idx:06d}.jpg")
        cv2.imwrite(save_img_path, image_bgr)
    
    print(f"Processed frame {frame_idx}/{datalen}")

# Save all results
if args.save_pk:
    with open(os.path.join(pkl_path, f"{video_basename}.pkl"), 'wb') as fout:
        pk.dump(output_list, fout)

stream.release()
print(f"Results saved to {out_dir}") 