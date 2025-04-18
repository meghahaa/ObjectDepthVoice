import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from .model_loader import get_midas_transforms

def estimate_depth(frame, depth_model):
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    transforms = get_midas_transforms()
    input_tensor = transforms(frame).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        prediction = depth_model(input_tensor)

    prediction_resized = F.interpolate(
        prediction.unsqueeze(1), size=(480, 640), mode="bicubic", align_corners=False).squeeze()
    depth_map = prediction_resized.cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    return depth_map