import torch
import torchvision.transforms as T

def load_models():
    detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    detection_model = detection_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    depth_model = depth_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    depth_model.eval()

    return detection_model, depth_model

def get_midas_transforms():
    return T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])