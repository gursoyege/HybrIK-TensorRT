import cv2 as cv
import numpy as np
from hybrik_inference import Hybrik
import time
import torch
from vis_tools import Visualizer
from pysmpl.pysmpl import PySMPL
from visdom import Visdom

pose_model = Hybrik("./hybrik.engine", model_input_size=(256,256), device="cuda")

smpl = PySMPL()
vis = Visualizer()

for i in range(1000):
    start_time = time.perf_counter()
    
    img = cv.imread("human-pose.jpg")
    dets = np.array([[  0.75959396,   1.6419703,  212.80466,    339.2592    ]])
    shape, pose = pose_model(img, dets + np.random.randint(-2,2))
    shape_tensor = torch.from_numpy(shape).to(device="cuda")
    pose_tensor = torch.from_numpy(pose).to(device="cuda")

    mesh = smpl(shape_tensor, pose_tensor, pose2rot=True)
    # vis.show_points([mesh])   
    
    end_time = time.perf_counter()
    print("Time: " + str((1/(end_time - start_time))))
