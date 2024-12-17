import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\DaiMa_YOLO\消融实验_300\UA_Detrac_300\7YOLOv8_SGBackbone_SlimNeck_DetectSA\runs\train\exp\weights\best.pt') # select your model.pt path
    model.predict(source=r'C:\Users\段汉清\Desktop\UA_Detrac',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  save_conf = True,
                  device='0',
                  )