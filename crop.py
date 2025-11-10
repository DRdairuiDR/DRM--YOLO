import cv2
import numpy as np
import os
import random
from tqdm import tqdm

def yolo_crop_augmentation(image, labels, crop_ratio_range=(0.6, 0.9), max_tries=100):
    """
    YOLO格式的裁剪数据增强（针对VisDrone小目标优化）
    Args:
        image: 输入图像 (H, W, C)
        labels: YOLO格式标签 [[class_id, x_center, y_center, width, height], ...]
        crop_ratio_range: 裁剪比例范围 (min, max) - 针对小目标设置更高最小值
        max_tries: 最大尝试次数 - 针对小目标增加尝试次数
    Returns:
        cropped_image: 裁剪后的图像
        new_labels: 裁剪后的新标签
    """
    if len(labels) == 0:
        return image, labels
        
    h, w = image.shape[:2]
    labels = np.array(labels)
    
    # 转换为绝对坐标
    abs_labels = labels.copy()
    abs_labels[:, 1] = labels[:, 1] * w  # x_center
    abs_labels[:, 2] = labels[:, 2] * h  # y_center
    abs_labels[:, 3] = labels[:, 3] * w  # width
    abs_labels[:, 4] = labels[:, 4] * h  # height
    
    # 计算所有目标的中心点
    centers = abs_labels[:, 1:3]
    
    for _ in range(max_tries):
        # 随机确定裁剪比例 - 针对小目标使用更高下限
        crop_ratio = random.uniform(*crop_ratio_range)
        crop_w = int(w * crop_ratio)
        crop_h = int(h * crop_ratio)
        
        # 随机确定裁剪起点
        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(0, h - crop_h)
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        
        # 筛选完全在裁剪区域内的目标
        inside_mask = (
            (abs_labels[:, 1] > x1) & 
            (abs_labels[:, 1] < x2) &
            (abs_labels[:, 2] > y1) & 
            (abs_labels[:, 2] < y2)
        )
        
        # 筛选部分在裁剪区域内的目标（针对小目标保留更多）
        partial_mask = (
            (abs_labels[:, 1] + abs_labels[:, 3]/2 > x1) &
            (abs_labels[:, 1] - abs_labels[:, 3]/2 < x2) &
            (abs_labels[:, 2] + abs_labels[:, 4]/2 > y1) &
            (abs_labels[:, 2] - abs_labels[:, 4]/2 < y2)
        )
        
        # 合并完全在内部和部分在内部的目标（优先保留完全在内部的）
        valid_mask = inside_mask | partial_mask
        valid_labels = abs_labels[valid_mask]
        
        # 如果找到包含目标的裁剪区域
        if len(valid_labels) > 0:
            # 裁剪图像
            cropped_image = image[y1:y2, x1:x2]
            new_h, new_w = cropped_image.shape[:2]
            
            # 调整标签坐标
            new_labels = []
            for label in valid_labels:
                class_id = int(label[0])
                
                # 计算原始边界框坐标
                x_min = max(label[1] - label[3]/2, x1)
                y_min = max(label[2] - label[4]/2, y1)
                x_max = min(label[1] + label[3]/2, x2)
                y_max = min(label[2] + label[4]/2, y2)
                
                # 转换为新坐标系下的中心点、宽高
                new_x_center = ((x_min + x_max)/2 - x1) / new_w
                new_y_center = ((y_min + y_max)/2 - y1) / new_h
                new_width = (x_max - x_min) / new_w
                new_height = (y_max - y_min) / new_h
                
                # 跳过尺寸过小的目标（小于5像素）
                if new_width * new_w < 5 or new_height * new_h < 5:
                    continue
                
                # 确保坐标在[0,1]范围内
                new_x_center = np.clip(new_x_center, 0.0, 1.0)
                new_y_center = np.clip(new_y_center, 0.0, 1.0)
                new_width = np.clip(new_width, 0.0, 1.0)
                new_height = np.clip(new_height, 0.0, 1.0)
                
                new_labels.append([class_id, new_x_center, new_y_center, new_width, new_height])
            
            # 确保裁剪后至少保留一个目标
            if len(new_labels) > 0:
                return cropped_image, new_labels
    
    # 如果未找到有效裁剪，返回原始图像和标签
    return image, labels

def process_dataset():
    """处理整个VisDrone数据集"""
    # ====== 直接在这里设置路径 ======
    input_dir = "/home/xd508/DR/VisDrone/VisDrone2019-DET-train"  # 修改为你的VisDrone数据集路径
    output_dir = "/home/xd508/DR/VisDrone/crop"  # 修改为输出路径
    
    # 针对VisDrone的优化参数
    crop_ratio_range = (0.6, 0.9)  # 更高的最小比例保留更多小目标
    max_tries = 100  # 增加尝试次数确保找到有效裁剪
    
    img_dir = os.path.join(input_dir, 'images')
    label_dir = os.path.join(input_dir, 'labels')
    
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    # 获取所有图像文件
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Processing {len(img_files)} images for VisDrone dataset...")
    
    for img_name in tqdm(img_files, desc="Cropping augmentation"):
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, base_name + '.txt')
        
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # 读取标签
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) == 5:  # YOLO格式: class_id, x_center, y_center, width, height
                        try:
                            labels.append([float(x) for x in data])
                        except ValueError:
                            continue
        
        # 应用裁剪增强
        new_image, new_labels = yolo_crop_augmentation(
            image, labels, crop_ratio_range, max_tries
        )
        
        # 保存增强后的图像和标签
        new_img_path = os.path.join(output_dir, 'images', f'crop_{img_name}')
        new_label_path = os.path.join(output_dir, 'labels', f'crop_{base_name}.txt')
        
        cv2.imwrite(new_img_path, new_image)
        
        with open(new_label_path, 'w') as f:
            for label in new_labels:
                f.write(f"{int(label[0])} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

if __name__ == "__main__":
    # 直接运行处理数据集
    process_dataset()
    print("Crop augmentation for VisDrone dataset completed!")

