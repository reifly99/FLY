# file: classifier_module.py
"""
可移植的分类模块
包含：全局池化、展平、多种分类器选择
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Union, Optional

class KANLinear(nn.Module):
    """
    KAN网络的线性层
    结合样条基函数和线性基函数
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: nn.Module = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: List[float] = [-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # 网格初始化
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(
            -spline_order, grid_size + spline_order + 1
        ) * h + grid_range[0]
        grid = grid.expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)
        
        # 可学习参数
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # 超参数
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
    
    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """计算B样条基函数"""
        assert x.dim() == 2 and x.size(1) == self.in_features
        
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x)
                / (grid[:, k + 1:] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        
        return bases.contiguous()
    
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """从曲线点计算样条系数"""
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        
        return result.contiguous()
    
    @property
    def scaled_spline_weight(self):
        """缩放后的样条权重"""
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        
        # 线性基函数输出
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # 样条基函数输出
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        
        # 合并输出
        output = base_output + spline_output
        output = output.view(*original_shape[:-1], self.out_features)
        
        return output
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """计算正则化损失"""
        l1_fake = self.scaled_spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / (regularization_loss_activation + 1e-8)
        regularization_loss_entropy = -torch.sum(p * p.log())
        
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(nn.Module):
    """KAN网络（多个KANLinear层堆叠）"""
    
    def __init__(
        self,
        layers_hidden: List[int],
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: nn.Module = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: List[float] = [-1, 1],
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features=in_features,
                    out_features=out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
    
    def forward(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        """前向传播"""
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """总正则化损失"""
        total_loss = 0
        for layer in self.layers:
            total_loss += layer.regularization_loss(regularize_activation, regularize_entropy)
        return total_loss


# classifier_module.py
class ClassifierHead(nn.Module):
    """
    通用的分类头部模块
    支持多种分类器类型
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        classifier_type: str = "linear",
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.5,
        kan_config: Optional[dict] = None,
        pool_type: str = "avg",  # "avg", "max", "attention", "none"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.classifier_type = classifier_type
        self.pool_type = pool_type
        
        # 池化层
        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pool_type == "attention":
            self.pool = AttentionPooling(input_dim)
        elif pool_type == "none":
            self.pool = None  # 不需要池化层
        else:
            raise ValueError(f"不支持的池化类型: {pool_type}")
        
        # 构建分类器
        if classifier_type == "linear":
            self.classifier = self._build_linear_classifier(
                input_dim, num_classes, hidden_dims, dropout_rate
            )
        elif classifier_type == "kan":
            self.classifier = self._build_kan_classifier(
                input_dim, num_classes, kan_config
            )
        elif classifier_type == "mlp":
            self.classifier = self._build_mlp_classifier(
                input_dim, num_classes, hidden_dims, dropout_rate
            )
        else:
            raise ValueError(f"不支持的分类器类型: {classifier_type}")
        
        # 展平层（仅在需要池化时才需要）
        if pool_type != "none":
            self.flatten = nn.Flatten(start_dim=1)
        else:
            self.flatten = None
        
    def _build_linear_classifier(self, input_dim, num_classes, hidden_dims, dropout_rate):
        """构建线性分类器"""
        if hidden_dims is None:
            return nn.Linear(input_dim, num_classes)
        else:
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout_rate))
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, num_classes))
            return nn.Sequential(*layers)
    
    def _build_mlp_classifier(self, input_dim, num_classes, hidden_dims, dropout_rate):
        """构建MLP分类器"""
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        return nn.Sequential(*layers)
    
    def _build_kan_classifier(self, input_dim, num_classes, kan_config):
        """构建KAN分类器"""
        if kan_config is None:
            kan_config = {}
        
        # 默认配置
        default_config = {
            "grid_size": 5,
            "spline_order": 3,
            "scale_noise": 0.1,
            "scale_base": 1.0,
            "scale_spline": 1.0,
            "base_activation": nn.SiLU,
            "grid_eps": 0.02,
            "grid_range": [-1, 1],
        }
        
        # 合并配置
        config = {**default_config, **kan_config}
        
        # 创建KAN网络
        kan_layers = [input_dim, num_classes]
        return KAN(layers_hidden=kan_layers, **config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
                - 如果pool_type != "none": 形状为 (B, C, L)
                - 如果pool_type == "none": 形状为 (B, C)
        
        Returns:
            输出张量，形状为 (B, num_classes)
        """
        # 如果需要池化
        if self.pool is not None:
            # 检查输入维度
            if x.dim() != 3:
                raise ValueError(f"当pool_type不为'none'时，输入应该是3D张量，但得到的是{x.dim()}D")
            
            # 池化
            x = self.pool(x)  # (B, C, 1) 或 (B, C, 1) 等
            
            # 展平
            if self.flatten is not None:
                x = self.flatten(x)  # (B, C)
        else:
            # 不需要池化，直接处理
            if x.dim() == 3:
                # 如果是3D输入，但不需要池化，可能是误用
                # 这里我们可以进行平均池化或直接取最后一个维度
                print("警告: 输入是3D但pool_type='none'，将进行全局平均池化")
                x = x.mean(dim=-1)  # (B, C)
            elif x.dim() != 2:
                raise ValueError(f"当pool_type='none'时，输入应该是2D张量，但得到的是{x.dim()}D")
        
        # 分类
        x = self.classifier(x)  # (B, num_classes)
        
        return x
    
    def get_regularization_loss(self, **kwargs):
        """获取正则化损失（仅KAN分类器支持）"""
        if self.classifier_type == "kan":
            return self.classifier.regularization_loss(**kwargs)
        return torch.tensor(0.0, device=next(self.classifier.parameters()).device)


class AttentionPooling(nn.Module):
    """注意力池化层"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.Tanh(),
            nn.Linear(input_dim // 4, 1),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        注意力池化
        
        Args:
            x: 输入张量，形状为 (B, C, L)
        
        Returns:
            池化后的张量，形状为 (B, C, 1)
        """
        B, C, L = x.shape
        
        # 转置以应用注意力
        x_t = x.permute(0, 2, 1)  # (B, L, C)
        
        # 计算注意力权重
        attn_weights = self.attention(x_t)  # (B, L, 1)
        
        # 加权平均
        weighted = torch.bmm(x, attn_weights.transpose(1, 2))  # (B, C, 1)
        
        return weighted


class MultiLabelLoss(nn.Module):
    """
    多标签分类损失函数集合
    """
    
    def __init__(
        self,
        loss_type: str = "bce",
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        alpha: float = 1.0,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight,
                reduction=reduction
            )
        elif loss_type == "focal":
            self.criterion = FocalLoss(
                alpha=alpha,
                gamma=gamma,
                reduction=reduction
            )
        elif loss_type == "weighted_bce":
            self.criterion = WeightedBCEWithLogitsLoss(
                alpha=alpha,
                gamma=gamma
            )
        elif loss_type == "label_smoothing":
            self.criterion = LabelSmoothingBCE(
                smoothing=label_smoothing,
                pos_weight=pos_weight,
                reduction=reduction
            )
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(preds, targets)


class WeightedBCEWithLogitsLoss(nn.Module):
    """带权重的BCE损失（带Focal Loss特性）"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C)
            targets: (B, C) 0/1
        """
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        pt = torch.where(targets == 1, p, 1 - p)
        focal_weight = (1 - pt) ** self.gamma
        
        return torch.mean(self.alpha * focal_weight * ce_loss)


class LabelSmoothingBCE(nn.Module):
    """标签平滑的BCE损失"""
    
    def __init__(
        self,
        smoothing: float = 0.1,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.pos_weight = pos_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 应用标签平滑
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )


# ==================== 使用示例 ====================

def example_usage():
    """使用示例"""
    
    # 1. 创建分类头（使用KAN分类器）
    classifier = ClassifierHead(
        input_dim=128,
        num_classes=6,
        classifier_type="kan",
        pool_type="avg",
        kan_config={
            "grid_size": 5,
            "spline_order": 3,
        }
    )
    
    # 测试前向传播
    batch_size = 4
    test_input = torch.randn(batch_size, 128, 256)  # (B, C, L)
    output = classifier(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    
    # 2. 创建多标签损失
    criterion = MultiLabelLoss(
        loss_type="weighted_bce",
        alpha=1.0,
        gamma=2.0
    )
    
    # 3. 计算损失
    targets = torch.randint(0, 2, (batch_size, 6)).float()
    loss = criterion(output, targets)
    print(f"损失值: {loss.item():.4f}")
    
    # 4. 获取正则化损失（仅KAN）
    if classifier.classifier_type == "kan":
        reg_loss = classifier.get_regularization_loss(
            regularize_activation=1.0,
            regularize_entropy=0.1
        )
        print(f"正则化损失: {reg_loss.item():.4f}")


if __name__ == "__main__":
    example_usage()
