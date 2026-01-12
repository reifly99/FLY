# model.py - 更新版本，集成稀疏注意力
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import CrossRegionAttntion, TSLANet_layer
from attention_modules import MultiDomainSparseAttention, SimplifiedSparseAttention1D
from classifier_module import ClassifierHead

class SignalFeatureExtractor(nn.Module):
    def __init__(self, input_channels=6, signal_length=1024, 
                 cnn_channels=[64, 128, 256], 
                 transformer_dim=256, num_heads=8, num_layers=4,
                 use_sparse_attention=True):
        super(SignalFeatureExtractor, self).__init__()
        
        self.use_sparse_attention = use_sparse_attention
        
        # === 添加输入归一化 ===
        self.input_norm = nn.BatchNorm1d(input_channels)
        
        # 1D CNN特征提取
        self.cnn_layers = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in cnn_channels:
            self.cnn_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            ))
            in_channels = out_channels
        
        # 计算经过CNN后的序列长度
        cnn_output_length = signal_length // (2 ** len(cnn_channels))
        
        # 多域稀疏注意力（可选）
        if use_sparse_attention:
            self.sparse_attention = MultiDomainSparseAttention(
                n_hashes=4,
                channels=cnn_channels[-1],
                reduction=4,
                chunk_size=16,
                res_scale=1
            )
        
        # 频率域处理
        self.fre = TSLANet_layer(transformer_dim)
        self.cmr = CrossRegionAttntion(transformer_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 投影层：将CNN输出投影到Transformer维度
        self.projection = nn.Linear(cnn_channels[-1], transformer_dim)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(
            torch.randn(1, cnn_output_length, transformer_dim)
        )
        
        self.cnn_output_length = cnn_output_length
        self.transformer_dim = transformer_dim
        
    def forward(self, x):
        """前向传播"""
        # === 添加输入归一化 ===
        x = self.input_norm(x)
        
        # 1D CNN特征提取
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        
        # 应用稀疏注意力（可选）
        if self.use_sparse_attention:
            x = self.sparse_attention(x)
        
        # 转置：(batch_size, length, channels)
        x = x.transpose(1, 2)
        
        # 投影到Transformer维度
        x = self.projection(x)
        
        # 添加位置编码
        x = x + self.pos_encoding
        
        # === 删除或修复错误的代码行 ===
        # B,C,L = x.shape  # 这些行代码有问题，建议删除
        # H,W = 16,16
        # x = x.view(B,C,16,16)
        # x = x.self.sparse_attention  # 这行代码是错误的
        # x = x.view(B,C,L)
        
        # 频率域处理
        x = self.fre(x)
        
        # Transformer编码
        x = self.transformer(x)
        x = self.cmr(x)
        
        # 全局平均池化
        x = x.mean(dim=1)  # (batch_size, transformer_dim)
        
        return x


class EnhancedSignalFeatureExtractor(nn.Module):
    """增强版信号特征提取器，集成多种注意力机制"""
    
    def __init__(self, input_channels=6, signal_length=1024, 
                 base_channels=64, num_blocks=3):
        super(EnhancedSignalFeatureExtractor, self).__init__()
        
        # 初始卷积
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # 多尺度稀疏注意力块
        self.attention_blocks = nn.ModuleList()
        current_channels = base_channels
        
        for i in range(num_blocks):
            # 稀疏注意力块
            attention_block = nn.Sequential(
                nn.Conv1d(current_channels, current_channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm1d(current_channels * 2),
                nn.ReLU(inplace=True),
                MultiDomainSparseAttention(
                    n_hashes=4 if i < 2 else 2,
                    channels=current_channels * 2,
                    reduction=4,
                    chunk_size=8 * (2 ** i)
                ),
                nn.Conv1d(current_channels * 2, current_channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm1d(current_channels * 2),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2 if i < num_blocks - 1 else 1)
            )
            self.attention_blocks.append(attention_block)
            current_channels = current_channels * 2
        
        # 全局特征提取
        self.global_features = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1)
        )
        
        # 特征变换
        self.final_projection = nn.Linear(current_channels, 256)
        
    def forward(self, x):
        """前向传播"""
        # 初始卷积
        x = self.initial_conv(x)
        
        # 多尺度稀疏注意力块
        for block in self.attention_blocks:
            x = block(x)
        
        # 全局特征
        x = self.global_features(x)
        
        # 特征变换
        x = self.final_projection(x)
        
        return x


class MultimodalFusion(nn.Module):
    """
    多模态特征融合模块
    现在使用ClassifierHead作为最终分类器
    """
    def __init__(self, text_dim=512, image_dim=512, signal_dim=256, 
                 fusion_dim=512, num_classes=5, classifier_type='kan'):
        super(MultimodalFusion, self).__init__()
        
        # 文本和图像特征融合
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        
        # 跨模态注意力（文本和图像）
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 脉搏信号特征投影
        self.signal_proj = nn.Linear(signal_dim, fusion_dim)
        
        # 最终融合（带稀疏注意力）
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 使用ClassifierHead代替原来的线性分类器
        self.classifier = ClassifierHead(
            input_dim=fusion_dim // 2,
            num_classes=num_classes,
            classifier_type=classifier_type,
            pool_type="none",
            kan_config={
                "grid_size": 3,
                "spline_order": 2,
                "scale_noise": 0.01,
                "scale_base": 0.1,
                "scale_spline": 0.1,
                "grid_eps": 0.1,
                "grid_range": [-2, 2],
            }
        )
        
    def forward(self, text_features, image_features, signal_features):
        """
        Args:
            text_features: (batch_size, text_dim)
            image_features: (batch_size, image_dim)
            signal_features: (batch_size, signal_dim)
        Returns:
            logits: (batch_size, num_classes)
        """
        # 投影文本和图像特征
        text_proj = self.text_proj(text_features.float())
        image_proj = self.image_proj(image_features.float())
        
        # 跨模态注意力
        cross_features, _ = self.cross_modal_attention(
            text_proj.unsqueeze(1),
            image_proj.unsqueeze(1),
            image_proj.unsqueeze(1)
        )
        cross_features = cross_features.squeeze(1)
        
        # 融合文本和图像特征
        multimodal_features = torch.cat([text_proj, cross_features], dim=1)
        multimodal_features = self.fusion_layer(multimodal_features)
        
        # 投影脉搏信号特征
        signal_proj = self.signal_proj(signal_features)
        
        # 融合多模态特征和脉搏信号特征
        combined_features = torch.cat([multimodal_features, signal_proj], dim=1)
        final_features = self.final_fusion(combined_features)
        
        # 分类
        logits = self.classifier(final_features)
        
        return logits
    
    def get_regularization_loss(self, **kwargs):
        """获取正则化损失（仅KAN分类器支持）"""
        return self.classifier.get_regularization_loss(**kwargs)


class TCMClassifier(nn.Module):
    """
    完整的多模态中医病证分类模型
    """
    def __init__(self, signal_channels=6, signal_length=1024, 
                 num_classes=5, device='cuda', classifier_type='kan',
                 use_enhanced_extractor=False):
        """
        Args:
            classifier_type: 分类器类型，可选 'linear', 'kan', 'mlp'
            use_enhanced_extractor: 是否使用增强版特征提取器
        """
        super(TCMClassifier, self).__init__()
        
        self.device = device
        self.classifier_type = classifier_type
        
        # 脉搏信号特征提取器
        if use_enhanced_extractor:
            self.signal_extractor = EnhancedSignalFeatureExtractor(
                input_channels=signal_channels,
                signal_length=signal_length
            )
            signal_dim = 256  # 增强版输出维度
        else:
            self.signal_extractor = SignalFeatureExtractor(
                input_channels=signal_channels,
                signal_length=signal_length,
                use_sparse_attention=True  # 启用稀疏注意力
            )
            signal_dim = 256  # 原版输出维度
        
        # 多模态融合和分类
        self.fusion_classifier = MultimodalFusion(
            text_dim=512,
            image_dim=512,
            signal_dim=signal_dim,
            fusion_dim=512,
            num_classes=num_classes,
            classifier_type=classifier_type
        )
    
    def forward_from_features(self, text_features, image_features, signals):
        """
        直接从特征进行前向传播
        Args:
            text_features: (batch_size, 512)
            image_features: (batch_size, 512)
            signals: (batch_size, channels, length)
        Returns:
            logits: (batch_size, num_classes)
        """
        signals = signals.to(self.device)
        signal_features = self.signal_extractor(signals)
        
        logits = self.fusion_classifier(text_features, image_features, signal_features)
        
        return logits
    
    def get_regularization_loss(self, **kwargs):
        """获取模型的正则化损失"""
        return self.fusion_classifier.get_regularization_loss(**kwargs)


# ==================== 模型工厂函数 ====================

def create_model(config, device='cuda'):
    """
    创建模型实例的工厂函数
    
    Args:
        config: 配置字典，包含模型参数
        device: 设备
        
    Returns:
        TCMClassifier实例
    """
    
    model = TCMClassifier(
        signal_channels=config.get('signal_channels', 6),
        signal_length=config.get('signal_length', 1024),
        num_classes=config.get('num_classes', 5),
        device=device,
        classifier_type=config.get('classifier_type', 'kan'),
        use_enhanced_extractor=config.get('use_enhanced_extractor', False)
    ).to(device)
    
    # 权重初始化
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    
    model.apply(init_weights)
    
    # 特殊处理KAN层的初始化
    if config.get('classifier_type', 'kan') == 'kan':
        def init_kan_weights(m):
            if hasattr(m, 'base_weight'):
                nn.init.normal_(m.base_weight, mean=0.0, std=0.01)
            if hasattr(m, 'spline_weight'):
                nn.init.normal_(m.spline_weight, mean=0.0, std=0.005)
            if hasattr(m, 'spline_scaler'):
                nn.init.constant_(m.spline_scaler, 0.1)
        
        # 只对KAN层应用特殊初始化
        for module in model.modules():
            if hasattr(module, '__class__') and 'KAN' in module.__class__.__name__:
                init_kan_weights(module)
    
    return model


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 测试模型
    config = {
        'signal_channels': 6,
        'signal_length': 1024,
        'num_classes': 5,
        'classifier_type': 'kan',
        'use_enhanced_extractor': True  # 使用增强版提取器
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(config, device)
    
    # 测试输入
    batch_size = 4
    test_text = torch.randn(batch_size, 512).to(device)
    test_image = torch.randn(batch_size, 512).to(device)
    test_signal = torch.randn(batch_size, 6, 1024).to(device)
    
    # 前向传播
    logits = model.forward_from_features(test_text, test_image, test_signal)
    print(f"模型输出形状: {logits.shape}")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 获取正则化损失
    reg_loss = model.get_regularization_loss(
        regularize_activation=1.0,
        regularize_entropy=0.1
    )
    print(f"正则化损失: {reg_loss}")
