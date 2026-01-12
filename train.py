# train.py - 修正版本
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms 
import clip
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import TCMDataset
from model import TCMClassifier, create_model
from classifier_module import WeightedBCEWithLogitsLoss  # 导入新的损失函数


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification
    """
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - logits
            targets: (batch_size, num_classes) - multi-label targets
        """
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    计算多标签分类的指标
    Args:
        y_true: (n_samples, n_classes) - 真实标签
        y_pred: (n_samples, n_classes) - 预测概率
        threshold: 阈值
    Returns:
        metrics dict
    """
    # 将概率转换为二进制预测
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # 计算每个类别的指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average=None, zero_division=0
    )
    
    # 宏平均
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    # 微平均
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average='micro', zero_division=0
    )
    
    # 准确率（子集准确率）
    subset_accuracy = accuracy_score(y_true, y_pred_binary)
    
    # Hamming损失
    hamming_loss = np.mean(y_true != y_pred_binary)
    
    return {
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'accuracy': subset_accuracy,
        'hamming_loss': hamming_loss
    }


def extract_clip_features(batch, clip_model, device):
    """
    实时提取CLIP特征
    """
    # CLIP图像预处理（标准化）
    clip_preprocess = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073), 
        std=(0.26862954, 0.26130258, 0.27577711)
    )
    
    # 标准化图像
    images = batch['image'].to(device)
    images = clip_preprocess(images)
    
    # 提取图像特征
    with torch.no_grad():
        image_features = clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # 提取文本特征
    texts = batch['text']  # 使用 'text' 而不是 'text_features'
    text_tokens = clip.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features, image_features


def train_epoch(model, dataloader, criterion, optimizer, device, clip_model, reg_weight=0.01):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    total_reg_loss = 0.0
    
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        # 实时提取CLIP特征
        text_features, image_features = extract_clip_features(batch, clip_model, device)
        signals = batch['signal'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        logits = model.forward_from_features(text_features, image_features, signals)
        
        # 计算主损失（分类损失）
        main_loss = criterion(logits, labels)
        
        # 计算正则化损失（如果使用KAN分类器）
        reg_loss = model.get_regularization_loss(
            regularize_activation=1.0,
            regularize_entropy=0.1
        )
        
        # 总损失 = 主损失 + 正则化损失权重 * 正则化损失
        total_batch_loss = main_loss + reg_weight * reg_loss
        
        # 反向传播
        total_batch_loss.backward()
        optimizer.step()
        
        # 收集统计信息
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        all_preds.append(probs)
        all_labels.append(labels_np)
        
        total_loss += total_batch_loss.item()
        total_main_loss += main_loss.item()
        total_reg_loss += reg_loss.item()
        
        progress_bar.set_postfix({
            'total_loss': total_batch_loss.item(),
            'main_loss': main_loss.item(),
            'reg_loss': reg_loss.item()
        })
    
    # 计算平均损失
    avg_total_loss = total_loss / len(dataloader)
    avg_main_loss = total_main_loss / len(dataloader)
    avg_reg_loss = total_reg_loss / len(dataloader)
    
    # 计算指标
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    metrics = calculate_metrics(all_labels, all_preds)
    
    metrics.update({
        'total_loss': avg_total_loss,
        'main_loss': avg_main_loss,
        'reg_loss': avg_reg_loss
    })
    
    return metrics


def validate_epoch(model, dataloader, criterion, device, clip_model):
    """
    验证一个epoch（不计算正则化损失）
    """
    model.eval()
    total_main_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation')
        for batch in progress_bar:
            text_features, image_features = extract_clip_features(batch, clip_model, device)
            signals = batch['signal'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model.forward_from_features(text_features, image_features, signals)
            main_loss = criterion(logits, labels)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_preds.append(probs)
            all_labels.append(labels_np)
            
            total_main_loss += main_loss.item()
            progress_bar.set_postfix({'loss': main_loss.item()})
    
    avg_main_loss = total_main_loss / len(dataloader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = avg_main_loss
    
    return metrics


def test_epoch(model, dataloader, criterion, device, clip_model):
    """
    测试一个epoch
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Testing')
        for batch in progress_bar:
            # 实时提取CLIP特征（关键修改）
            text_features, image_features = extract_clip_features(batch, clip_model, device)
            signals = batch['signal'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            logits = model.forward_from_features(text_features, image_features, signals)
            
            # 计算损失
            loss = criterion(logits, labels)
            
            # 收集预测和标签
            probs = torch.sigmoid(logits).cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_preds.append(probs)
            all_labels.append(labels_np)
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # 计算指标
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = avg_loss
    
    return metrics


def print_metrics(metrics, phase='Train'):
    """
    打印指标
    """
    print(f"\n{phase} Metrics:")
    if 'total_loss' in metrics:
        print(f"  Total Loss: {metrics['total_loss']:.4f}")
        print(f"  Main Loss: {metrics['main_loss']:.4f}")
        print(f"  Reg Loss: {metrics['reg_loss']:.4f}")
    else:
        print(f"  Loss: {metrics.get('loss', 'N/A'):.4f}")
    
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Micro Precision: {metrics['micro_precision']:.4f}")
    print(f"  Micro Recall: {metrics['micro_recall']:.4f}")
    print(f"  Micro F1: {metrics['micro_f1']:.4f}")
    print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"\n  Per-class Metrics:")
    class_names = ['Normal', 'Class1', 'Class2', 'Class3', 'Class4']
    for i, name in enumerate(class_names):
        print(f"    {name}:")
        print(f"      Precision: {metrics['precision_per_class'][i]:.4f}")
        print(f"      Recall: {metrics['recall_per_class'][i]:.4f}")
        print(f"      F1: {metrics['f1_per_class'][i]:.4f}")


def plot_training_history(train_history, val_history, test_history=None, save_dir='./results'):
    """
    绘制训练历史
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(train_history['loss']) + 1)
    
    # 损失曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_history['loss'], 'b-', label='Train Loss')
    plt.plot(epochs, val_history['loss'], 'r-', label='Val Loss')
    if test_history:
        # 只绘制非None的测试点
        test_epochs = [i+1 for i, v in enumerate(test_history['loss']) if v is not None]
        test_losses = [v for v in test_history['loss'] if v is not None]
        if test_epochs:
            plt.plot(test_epochs, test_losses, 'g-o', label='Test Loss', markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training, Validation and Test Loss')
    plt.legend()
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_history['accuracy'], 'b-', label='Train Accuracy')
    plt.plot(epochs, val_history['accuracy'], 'r-', label='Val Accuracy')
    if test_history:
        test_epochs = [i+1 for i, v in enumerate(test_history['accuracy']) if v is not None]
        test_accs = [v for v in test_history['accuracy'] if v is not None]
        if test_epochs:
            plt.plot(test_epochs, test_accs, 'g-o', label='Test Accuracy', markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training, Validation and Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    # F1分数曲线
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_history['macro_f1'], 'b-', label='Train Macro F1')
    plt.plot(epochs, val_history['macro_f1'], 'r-', label='Val Macro F1')
    if test_history:
        test_epochs = [i+1 for i, v in enumerate(test_history['macro_f1']) if v is not None]
        test_f1s = [v for v in test_history['macro_f1'] if v is not None]
        if test_epochs:
            plt.plot(test_epochs, test_f1s, 'g-o', label='Test Macro F1', markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training, Validation and Test F1 Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()
# 在train.py中添加数据检查函数
def check_data_integrity(dataloader, name="训练数据"):
    """检查数据集中是否有NaN或异常值"""
    print(f"\n检查{name}完整性...")
    
    nan_count = 0
    inf_count = 0
    zero_count = 0
    extreme_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 检查图像数据
        if torch.isnan(batch['image']).any():
            nan_count += 1
            print(f"批次{batch_idx}: 图像包含NaN")
        
        if torch.isinf(batch['image']).any():
            inf_count += 1
            print(f"批次{batch_idx}: 图像包含Inf")
        
        # 检查信号数据
        if torch.isnan(batch['signal']).any():
            nan_count += 1
            print(f"批次{batch_idx}: 信号包含NaN")
        
        if torch.isinf(batch['signal']).any():
            inf_count += 1
            print(f"批次{batch_idx}: 信号包含Inf")
        
        # 检查极端值
        if (batch['signal'].abs() > 100).any():
            extreme_count += 1
            print(f"批次{batch_idx}: 信号值过大")
        
        if batch_idx >= 10:  # 只检查前10个批次
            break
    
    print(f"{name}检查完成:")
    print(f"  NaN批次: {nan_count}")
    print(f"  Inf批次: {inf_count}")
    print(f"  极端值批次: {extreme_count}")
    
    return nan_count == 0 and inf_count == 0

def check_data_leakage(train_loader, val_loader, test_loader):
    """检查数据泄露"""
    train_files = set()
    val_files = set() 
    test_files = set()
    
    for batch in train_loader:
        for name in batch['image_name']:
            train_files.add(name)
    
    for batch in val_loader:
        for name in batch['image_name']:
            val_files.add(name)
            
    for batch in test_loader:
        for name in batch['image_name']:
            test_files.add(name)
    
    train_val_overlap = train_files.intersection(val_files)
    train_test_overlap = train_files.intersection(test_files)
    val_test_overlap = val_files.intersection(test_files)
    
    print(f"训练集样本数: {len(train_files)}")
    print(f"验证集样本数: {len(val_files)}") 
    print(f"测试集样本数: {len(test_files)}")
    print(f"训练集-验证集重叠: {len(train_val_overlap)}")
    print(f"训练集-测试集重叠: {len(train_test_overlap)}")
    print(f"验证集-测试集重叠: {len(val_test_overlap)}")
    
    return len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0


# train.py - 增强的训练监控
class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, model):
        self.model = model
        self.grad_history = []
        self.weight_history = []
        self.nan_events = []
        
    def check_gradients(self, epoch, batch_idx):
        """检查梯度"""
        grad_stats = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_stats[name] = {
                    'norm': grad_norm,
                    'mean': param.grad.data.mean().item(),
                    'std': param.grad.data.std().item(),
                    'has_nan': torch.isnan(param.grad.data).any().item(),
                    'has_inf': torch.isinf(param.grad.data).any().item()
                }
                
                # 记录梯度异常
                if torch.isnan(param.grad.data).any() or torch.isinf(param.grad.data).any():
                    self.nan_events.append({
                        'epoch': epoch,
                        'batch': batch_idx,
                        'param': name,
                        'type': 'gradient'
                    })
        
        self.grad_history.append(grad_stats)
        return grad_stats
    
    def check_weights(self, epoch):
        """检查权重"""
        weight_stats = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                weight_stats[name] = {
                    'norm': param.data.norm(2).item(),
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item(),
                    'has_nan': torch.isnan(param.data).any().item(),
                    'has_inf': torch.isinf(param.data).any().item()
                }
        
        self.weight_history.append(weight_stats)
        return weight_stats
    
    def print_summary(self, epoch):
        """打印监控摘要"""
        if epoch % 10 == 0:  # 每10个epoch打印一次
            print(f"\n=== 训练监控摘要 (Epoch {epoch}) ===")
            
            # 检查最近的梯度
            if self.grad_history:
                recent_grads = self.grad_history[-1]
                print("梯度统计:")
                for name, stats in list(recent_grads.items())[:5]:  # 只显示前5个参数
                    if stats['has_nan'] or stats['has_inf']:
                        print(f"  {name}: 范数={stats['norm']:.6f} (包含NaN/Inf!)")
                    else:
                        print(f"  {name}: 范数={stats['norm']:.6f}")
            
            # 检查NaN事件
            if self.nan_events:
                print(f"\nNaN事件总数: {len(self.nan_events)}")
                for event in self.nan_events[-5:]:  # 显示最近5个事件
                    print(f"  Epoch {event['epoch']}, Batch {event['batch']}: {event['param']}")
            
            print("=" * 40)


def train_epoch(model, dataloader, criterion, optimizer, device, clip_model, 
                reg_weight=0.01, epoch_idx=0, monitor=None):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    total_reg_loss = 0.0
    
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc=f'训练 Epoch {epoch_idx+1}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # 1. 前向传播
        text_features, image_features = extract_clip_features(batch, clip_model, device)
        signals = batch['signal'].to(device)
        labels = batch['labels'].to(device)
        
        # 检查输入数据
        if torch.isnan(signals).any() or torch.isinf(signals).any():
            print(f"警告: 批次{batch_idx}的输入信号包含NaN/Inf")
            continue
        
        # 前向传播
        optimizer.zero_grad()
        logits = model.forward_from_features(text_features, image_features, signals)
        
        # 检查logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"严重警告: 批次{batch_idx}的logits包含NaN/Inf")
            print(f"  logits统计: 均值={logits.mean().item():.6f}, "
                  f"标准差={logits.std().item():.6f}")
            continue
        
        # 2. 计算损失
        main_loss = criterion(logits, labels)
        
        if torch.isnan(main_loss) or torch.isinf(main_loss):
            print(f"严重警告: 批次{batch_idx}的main_loss为NaN/Inf")
            continue
        
        # 3. 计算正则化损失（如果有）
        reg_loss = model.get_regularization_loss(
            regularize_activation=1.0,
            regularize_entropy=0.1
        )
        
        if torch.isnan(reg_loss) or torch.isinf(reg_loss):
            reg_loss = torch.tensor(0.0, device=device)
            print(f"警告: 批次{batch_idx}的reg_loss为NaN/Inf，已置零")
        
        # 4. 总损失
        total_batch_loss = main_loss + reg_weight * reg_loss
        
        # 5. 反向传播
        total_batch_loss.backward()
        
        # 6. 梯度检查和裁剪
        if monitor:
            grad_stats = monitor.check_gradients(epoch_idx, batch_idx)
            
            # 如果有梯度问题，跳过这个批次
            has_grad_problem = False
            for name, stats in grad_stats.items():
                if stats['has_nan'] or stats['has_inf']:
                    print(f"梯度问题: {name}包含NaN/Inf")
                    has_grad_problem = True
            
            if has_grad_problem:
                optimizer.zero_grad()  # 清空梯度
                continue
        
        # 梯度裁剪（使用更严格的值）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        # 7. 优化器步骤
        optimizer.step()
        
        # 8. 收集统计信息
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        all_preds.append(probs)
        all_labels.append(labels_np)
        
        total_loss += total_batch_loss.item()
        total_main_loss += main_loss.item()
        total_reg_loss += reg_loss.item()
        
        progress_bar.set_postfix({
            'loss': total_batch_loss.item(),
            'main': main_loss.item(),
            'reg': reg_loss.item()
        })
    
    # 计算指标
    if all_preds:
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        metrics = calculate_metrics(all_labels, all_preds)
    else:
        metrics = create_empty_metrics(config['num_classes'])
    
    metrics.update({
        'total_loss': total_loss / max(len(dataloader), 1),
        'main_loss': total_main_loss / max(len(dataloader), 1),
        'reg_loss': total_reg_loss / max(len(dataloader), 1)
    })
    
    return metrics


def create_empty_metrics(num_classes):
    """创建空的指标字典"""
    return {
        'precision_per_class': np.zeros(num_classes),
        'recall_per_class': np.zeros(num_classes),
        'f1_per_class': np.zeros(num_classes),
        'macro_precision': 0.0,
        'macro_recall': 0.0,
        'macro_f1': 0.0,
        'micro_precision': 0.0,
        'micro_recall': 0.0,
        'micro_f1': 0.0,
        'accuracy': 0.0,
        'hamming_loss': 1.0
    }

# 确保main函数内部的代码都正确缩进

def main():
    # 配置参数
    config = {
        'csv_path': '/data/LLN/src/data/label.csv',
        'image_dir': '/data/LLN/src/data/image',
        'csv_dir': '/data/LLN/src/data/csv',
        'batch_size': 8,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'signal_length': 1024,
        'num_classes': 5,
        'train_ratio': 0.7,
        'val_ratio': 0.2,
        'test_ratio': 0.1,
        'test_interval': 5,
        'save_dir': './results',
        'model_save_path': './results/best_model.pth',
        
        # 模型配置
        'model_config': {
            'signal_channels': 6,
            'signal_length': 1024,
            'num_classes': 5,
            'classifier_type': 'kan',
            'use_sparse_attention': True,  # 启用稀疏注意力
            'use_enhanced_extractor': False,  # 是否使用增强版提取器
        },
        
        # 训练配置
        'reg_weight': 0.01,
        'loss_type': 'weighted_bce'
    }
    
    
    print(f"Using device: {config['device']}")
    
    print(f"Using device: {config['device']}")
    print(f"Number of classes: {config['num_classes']}")
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 加载CLIP模型
    print("Loading CLIP model...")
    clip_model, _ = clip.load("ViT-B/32", device=config['device'])
    clip_model.eval()
    # 冻结CLIP模型参数
    for param in clip_model.parameters():
        param.requires_grad = False
    
    # 创建数据集
    print("Creating dataset...")
    full_dataset = TCMDataset(
        csv_path=config['csv_path'],
        image_dir=config['image_dir'],
        csv_dir=config['csv_dir'],
        signal_length=config['signal_length']
    )
    
    # 划分训练集、验证集和测试集（7:2:1）
    total_size = len(full_dataset)
    train_size = int(config['train_ratio'] * total_size)
    val_size = int(config['val_ratio'] * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(0)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    
    print("检查数据泄露...")
    is_clean = check_data_leakage(train_loader, val_loader, test_loader)
    if is_clean:
        print("✓ 数据划分正确，无泄露")
    else:
        print("⚠ 发现数据泄露问题！")
        return
    
    # 创建模型
    print("Creating model...")
    model = create_model(config['model_config'], config['device'])
    
    # 创建训练监控器
    monitor = TrainingMonitor(model)
    
    # 使用更稳定的优化器
    optimizer = optim.AdamW(
        [
            {'params': model.signal_extractor.parameters(), 'lr': config['learning_rate']},
            {'params': model.fusion_classifier.parameters(), 'lr': config['learning_rate'] * 0.5},
        ],
        lr=config['learning_rate'],
        betas=(0.9, 0.999),  # 默认值
        eps=1e-8,  # 增加epsilon提高稳定性
        weight_decay=config['weight_decay']
    )
    
    
        # 更激进的学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,  # 更短的耐心值
        threshold=0.0001,  # 更小的阈值
        threshold_mode='rel', 
        cooldown=0, 
        min_lr=1e-7,  # 更小的最小学习率
        verbose=True
    )
    
    # 添加学习率预热
    def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
        def f(x):
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha
        return optim.lr_scheduler.LambdaLR(optimizer, f)
    
    warmup_iters = min(10, len(train_loader))
    warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, 0.001)
    
    # 检查数据完整性
    print("检查数据完整性...")
    train_clean = check_data_integrity(train_loader, "训练数据")
    val_clean = check_data_integrity(val_loader, "验证数据")
    test_clean = check_data_integrity(test_loader, "测试数据")
    
    if not (train_clean and val_clean and test_clean):
        print("警告: 数据中包含NaN/Inf，可能需要数据清洗")
    
    # 损失函数
    if config['loss_type'] == 'weighted_bce':
        criterion = WeightedBCEWithLogitsLoss(alpha=1.0, gamma=2.0)
    else:
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
    
    # 分段训练策略
    training_phases = [
        {'name': '相位1-预热', 'epochs': 5, 'lr_factor': 0.1, 'reg_weight': 0.0},
        {'name': '相位2-主训练', 'epochs': 30, 'lr_factor': 1.0, 'reg_weight': 0.001},
        {'name': '相位3-微调', 'epochs': 65, 'lr_factor': 0.1, 'reg_weight': 0.01},
    ]
    
    current_epoch = 0
    best_val_f1 = 0.0
    
    for phase_idx, phase in enumerate(training_phases):
        print(f"\n{'='*60}")
        print(f"开始{phase['name']} (Epoch {current_epoch+1}-{current_epoch+phase['epochs']})")
        print(f"{'='*60}")
        
        # 调整学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = config['learning_rate'] * phase['lr_factor']
        
        # 相位特定的训练
        for epoch in range(phase['epochs']):
            actual_epoch = current_epoch + epoch
            
            print(f"\nEpoch {actual_epoch + 1}/{config['num_epochs']}")
            print(f"学习率: {optimizer.param_groups[0]['lr']:.2e}")
            
            # 训练
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, 
                config['device'], clip_model, phase['reg_weight'],
                epoch_idx=actual_epoch, monitor=monitor
            )
            
            # 验证
            val_metrics = validate_epoch(
                model, val_loader, criterion, config['device'], clip_model
            )
            
            # 打印指标
            print_metrics(train_metrics, 'Train')
            print_metrics(val_metrics, 'Validation')
            
            # 更新学习率调度器
            scheduler.step(val_metrics['loss'])
            
            # 监控器摘要
            if monitor:
                monitor.print_summary(actual_epoch)
                monitor.check_weights(actual_epoch)
            
            # 检查NaN问题
            if np.isnan(val_metrics['loss']) or np.isinf(val_metrics['loss']):
                print(f"严重错误: 验证损失为NaN/Inf，停止当前相位")
                
                # 加载最佳模型
                if os.path.exists(config['model_save_path']):
                    print("加载最佳模型...")
                    checkpoint = torch.load(config['model_save_path'])
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                break
            
            # 保存最佳模型
            if val_metrics['macro_f1'] > best_val_f1:
                best_val_f1 = val_metrics['macro_f1']
                torch.save({
                    'epoch': actual_epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_f1': best_val_f1,
                    'config': config
                }, config['model_save_path'])
                print(f"\n✓ 保存最佳模型 (Val F1: {best_val_f1:.4f})")
            
            # 每10个epoch保存一次检查点
            if (actual_epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(
                    config['save_dir'], 
                    f'checkpoint_epoch_{actual_epoch+1}.pth'
                )
                torch.save({
                    'epoch': actual_epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_metrics['macro_f1'],
                    'val_loss': val_metrics['loss']
                }, checkpoint_path)
        
        current_epoch += phase['epochs']
        
        # 如果已经达到总epoch数，提前结束
        if current_epoch >= config['num_epochs']:
            break
    


if __name__ == '__main__':
    main()
