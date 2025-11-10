from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    # model = YOLO('yolov8m.pt')  # load an official model
    model = YOLO('runs/detect/train16/weights/best.pt')  # load a custom model
    results = model.predict(source="/home/xd508/DR/yolov11-Dymic/ultralytics-main/predict3", device='0', save=True, line_width=1)  # predict on an image
    print(results)
