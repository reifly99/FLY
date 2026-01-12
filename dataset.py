import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class TCMDataset(Dataset):
    """
    多模态中医病证分类数据集
    包含图像、文本和脉搏信号（原始数据）
    """
    def __init__(self, csv_path, image_dir, csv_dir, signal_length=1024, is_train=True):
        """
        Args:
            csv_path: label.csv路径
            image_dir: 图像文件夹路径
            csv_dir: 脉搏信号CSV文件夹路径
            signal_length: 脉搏信号长度（采样点数）
            is_train: 是否为训练集
        """
        self.image_dir = image_dir
        self.csv_dir = csv_dir
        self.signal_length = signal_length
        
        # 读取标签文件
        self.data = pd.read_csv(csv_path, header=None)
        
        # 图像预处理（基础transform，不包含CLIP特定的normalization）
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # 注意：移除了CLIP特定的normalization，将在训练时处理
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 获取图片名称
        image_name = str(row[0]).strip()
        image_path = os.path.join(self.image_dir, image_name)
        
        # 加载图像（返回原始tensor）
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.image_transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 创建黑色图像作为占位符
            image_tensor = torch.zeros((3, 224, 224))
        
        # 获取文本描述（返回原始文本）
        text_description = str(row[6]) if pd.notna(row[6]) else "normal"
        
        # 加载脉搏信号
        signal_id = image_name.replace('.png', '')
        signal_path = os.path.join(self.csv_dir, f"{signal_id}.csv")
        
        try:
            signal_data = pd.read_csv(signal_path, header=None)
            # 提取所有通道的数据
            signal_array = signal_data.values.astype(np.float32)
            
            # 如果数据行数超过signal_length，进行下采样
            if signal_array.shape[0] > self.signal_length:
                indices = np.linspace(0, signal_array.shape[0] - 1, self.signal_length, dtype=int)
                signal_array = signal_array[indices]
            # 如果数据行数少于signal_length，进行上采样（重复最后一个值）
            elif signal_array.shape[0] < self.signal_length:
                last_row = signal_array[-1:]
                padding = np.repeat(last_row, self.signal_length - signal_array.shape[0], axis=0)
                signal_array = np.vstack([signal_array, padding])
            
            # 确保是1024行
            signal_array = signal_array[:self.signal_length]
            
            # 转置，使得形状为 (channels, length)
            # 假设CSV的列是通道
            if signal_array.shape[1] > 0:
                signal_tensor = torch.from_numpy(signal_array.T).float()
            else:
                # 如果没有列，创建一个单通道信号
                signal_tensor = torch.from_numpy(signal_array).float().unsqueeze(0)
                
        except Exception as e:
            print(f"Error loading signal {signal_path}: {e}")
            # 创建零信号作为占位符 (6 channels, 1024 length)
            signal_tensor = torch.zeros((6, self.signal_length))
        
        # 处理标签：B-E列（索引1-4）对应4个病症类别
        multi_label = torch.zeros(5, dtype=torch.float32)
        
        has_disease = False
        for col_idx in range(1, 5):  # B-F列，对应类别1-4
            value = row[col_idx]
            # 检查该列是否有病症标记
            if pd.notna(value) and str(value).strip() != '' and str(value).strip().lower() != '无':
                try:
                    # 尝试解析数字（虽然数字本身不重要，重要的是该列有值）
                    label_val = int(float(str(value).strip()))
                    # 列位置对应类别：col_idx=1对应类别1，col_idx=2对应类别2，以此类推
                    class_idx = col_idx  # 类别索引（1-4）
                    if 1 <= class_idx <= 4:
                        multi_label[class_idx] = 1.0
                        has_disease = True
                except:
                    # 如果无法解析，但列有值，仍然标记为有病症
                    class_idx = col_idx
                    if 1 <= class_idx <= 4:
                        multi_label[class_idx] = 1.0
                        has_disease = True
        
        # 如果所有列都没有病症标记，则标记为正常类（类别0）
        if not has_disease:
            multi_label[0] = 1.0
        
        return {
            'image': image_tensor,        # 原始图像tensor
            'text': text_description,     # 原始文本
            'signal': signal_tensor,      # 信号数据
            'labels': multi_label,        # 标签
            'image_name': image_name
        }
