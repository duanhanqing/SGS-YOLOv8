import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\YOLO\4YOLOv8_SCDown_GhostModule_SlimNeck_DetectSA\YOLOv8.2\runs\train\exp\weights\best.pt')
    model.val(data=r'D:\YOLO\4YOLOv8_SCDown_GhostModule_SlimNeck_DetectSA\YOLOv8.2\ultralytics\cfg\datasets\UA_Detrac.yaml',
              split='val',
              imgsz=640,
              batch=16,
              workers=4,
              device='0',
              # rect=False,
              # save_json=True, # 这个保存coco精度指标的开关
              project='runs/val',
              name='exp',
              )