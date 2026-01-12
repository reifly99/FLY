# attention_modules.py - 更新版本
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# ==================== 辅助函数 ====================
def batched_index_select(values, indices):
    """批量索引选择"""
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

# ==================== 多域稀疏注意力机制 ====================

class MultiDomainSparseAttention(nn.Module):
    """
    多域稀疏注意力机制
    结合局部敏感哈希(LSH)实现稀疏注意力，降低计算复杂度
    适用于多模态特征融合场景
    """
    
    def __init__(self, n_hashes=4, channels=64, reduction=4, chunk_size=8, res_scale=1):
        """
        初始化参数
        Args:
            n_hashes: 哈希轮数
            channels: 输入通道数
            reduction: 通道缩减比例
            chunk_size: 分块大小
            res_scale: 残差缩放因子
        """
        super(MultiDomainSparseAttention, self).__init__()
        
        self.chunk_size = chunk_size
        self.n_hashes = n_hashes
        self.reduction = reduction
        self.res_scale = res_scale
        
        # 匹配卷积（用于计算相似度）
        self.conv_match = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels // reduction),
            nn.ReLU(inplace=True)
        )
        
        # 组装卷积（用于特征变换）
        self.conv_assembly = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 归一化层
        self.norm = nn.BatchNorm1d(channels) if channels >= 8 else nn.Identity()
    
    def LSH(self, hash_buckets, x):
        """
        局部敏感哈希 - 将相似特征映射到相同桶中
        Args:
            hash_buckets: 哈希桶数量
            x: 输入特征 [N, L, C]
        Returns:
            hash_codes: 哈希编码 [N, n_hashes*L]
        """
        N = x.shape[0]
        device = x.device
        
        # 生成随机旋转矩阵
        rotations_shape = (1, x.shape[-1], self.n_hashes, hash_buckets // 2)
        random_rotations = torch.randn(rotations_shape, dtype=x.dtype, device=device).expand(N, -1, -1, -1)
        
        # 局部敏感哈希
        rotated_vecs = torch.einsum('btf,bfhi->bhti', x, random_rotations)
        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
        
        # 获取哈希码
        hash_codes = torch.argmax(rotated_vecs, dim=-1)
        
        # 添加偏移避免哈希码重叠
        offsets = torch.arange(self.n_hashes, device=device).reshape(1, -1, 1)
        hash_codes = torch.reshape(hash_codes + offsets, (N, -1))
        
        return hash_codes
    
    def add_adjacent_buckets(self, x):
        """添加相邻桶以增加感受野"""
        x_extra_back = torch.cat([x[:, :, -1:, ...], x[:, :, :-1, ...]], dim=2)
        x_extra_forward = torch.cat([x[:, :, 1:, ...], x[:, :, :1, ...]], dim=2)
        return torch.cat([x, x_extra_back, x_extra_forward], dim=3)
    
    def forward(self, input_tensor):
        """
        前向传播
        Args:
            input_tensor: 输入特征 [N, C, L] (1D序列)
        Returns:
            输出特征 [N, C, L]
        """
        N, C, L = input_tensor.shape
        
        # 特征变换
        x_embed = self.conv_match(input_tensor).view(N, -1, L).contiguous().permute(0, 2, 1)  # [N, L, C//reduction]
        y_embed = self.conv_assembly(input_tensor).view(N, -1, L).contiguous().permute(0, 2, 1)  # [N, L, C]
        
        # 哈希桶数量
        hash_buckets = min(L // self.chunk_size + (L // self.chunk_size) % 2, 128)
        
        # 哈希编码
        hash_codes = self.LSH(hash_buckets, x_embed).detach()
        
        # 按哈希码排序分组
        _, indices = hash_codes.sort(dim=-1)
        _, undo_sort = indices.sort(dim=-1)
        mod_indices = indices % L
        
        x_embed_sorted = batched_index_select(x_embed, mod_indices)
        y_embed_sorted = batched_index_select(y_embed, mod_indices)
        
        # 重塑为多轮哈希
        x_embed_sorted = x_embed_sorted.view(N, self.n_hashes, -1, C // self.reduction)
        y_embed_sorted = y_embed_sorted.view(N, self.n_hashes, -1, C)
        
        # 填充
        padding = self.chunk_size - L % self.chunk_size if L % self.chunk_size != 0 else 0
        if padding:
            x_pad = x_embed_sorted[:, :, -padding:, :].clone()
            y_pad = y_embed_sorted[:, :, -padding:, :].clone()
            x_embed_sorted = torch.cat([x_embed_sorted, x_pad], dim=2)
            y_embed_sorted = torch.cat([y_embed_sorted, y_pad], dim=2)
        
        # 重塑分块
        x_blocks = x_embed_sorted.view(N, self.n_hashes, -1, self.chunk_size, C // self.reduction)
        y_blocks = y_embed_sorted.view(N, self.n_hashes, -1, self.chunk_size, C)
        
        # 归一化
        x_match = F.normalize(x_blocks, p=2, dim=-1, eps=5e-5)
        
        # 扩展邻接桶
        x_match = self.add_adjacent_buckets(x_match)
        y_blocks = self.add_adjacent_buckets(y_blocks)
        
        # 注意力计算
        raw_score = torch.einsum('bhkie,bhkje->bhkij', x_blocks, x_match)
        bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
        score = torch.exp(raw_score - bucket_score)
        
        # 注意力聚合
        attention_output = torch.einsum('bukij,bukje->bukie', score, y_blocks)
        attention_output = attention_output.view(N, self.n_hashes, -1, C)
        
        # 恢复原始顺序
        attention_output = attention_output.view(N, -1, C)
        attention_output = batched_index_select(attention_output, undo_sort)
        
        # 重塑并加权融合
        attention_output = attention_output.view(N, self.n_hashes, L, C)
        bucket_score = bucket_score.view(N, self.n_hashes, L, 1)
        probs = F.softmax(bucket_score, dim=1)
        final_output = torch.sum(attention_output * probs, dim=1)
        
        # 重塑并添加残差
        final_output = final_output.permute(0, 2, 1).contiguous()
        final_output = self.norm(final_output) * self.res_scale + input_tensor
        
        return final_output


class CrossRegionAttention(nn.Module):
    """
    跨区域注意力机制（用于舌脉数据）
    模拟中医诊断中舌诊和脉诊信息的交互
    """
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(CrossRegionAttention, self).__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 跨区域注意力
        self.cross_q_proj = nn.Linear(dim, dim)
        self.cross_k_proj = nn.Linear(dim, dim)
        self.cross_v_proj = nn.Linear(dim, dim)
        
        # 输出投影
        self.out_proj = nn.Linear(dim, dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, tongue_features, pulse_features):
        """
        舌诊和脉诊特征的跨区域注意力
        Args:
            tongue_features: 舌诊特征 (B, L_t, D)
            pulse_features: 脉诊特征 (B, L_p, D)
        """
        batch_size = tongue_features.size(0)
        
        # 投影
        q = self.cross_q_proj(tongue_features).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)  # (B, H, L_t, D)
        
        k = self.cross_k_proj(pulse_features).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)  # (B, H, L_p, D)
        
        v = self.cross_v_proj(pulse_features).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)  # (B, H, L_p, D)
        
        # 计算跨区域注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, -1, self.dim
        )
        
        # 输出投影
        output = self.out_proj(attended)
        
        return output, attn_weights


# ==================== 简化版本（更易使用） ====================

class SimplifiedSparseAttention1D(nn.Module):
    """
    简化版稀疏注意力（1D版本）
    更适合时间序列数据
    """
    
    def __init__(self, channels=64, n_hashes=4, reduction=4):
        super(SimplifiedSparseAttention1D, self).__init__()
        
        self.n_hashes = n_hashes
        self.reduction = reduction
        
        # 轻量级特征变换
        self.query_conv = nn.Conv1d(channels, channels // reduction, 1)
        self.key_conv = nn.Conv1d(channels, channels // reduction, 1)
        self.value_conv = nn.Conv1d(channels, channels, 1)
        
        self.out_conv = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # 归一化
        self.norm = nn.BatchNorm1d(channels)
    
    def hash_attention(self, q, k, v):
        """哈希注意力计算"""
        B, C, L = q.shape
        
        # 展平序列维度
        q = q.permute(0, 2, 1)  # [B, L, C]
        k = k.permute(0, 2, 1)  # [B, L, C]
        v = v.permute(0, 2, 1)  # [B, L, C]
        
        # 简单哈希函数
        hash_buckets = min(L // 4, 32)
        hash_indices = torch.randint(0, hash_buckets, (B, self.n_hashes, L), device=q.device)
        
        outputs = []
        scores = []
        
        for h in range(self.n_hashes):
            # 按哈希索引分组
            unique_indices = torch.unique(hash_indices[:, h, :])
            group_output = torch.zeros_like(v)
            
            for idx in unique_indices:
                mask = (hash_indices[:, h, :] == idx).unsqueeze(-1)
                
                # 组内注意力
                group_q = torch.masked_select(q, mask).view(B, -1, C // self.reduction)
                group_k = torch.masked_select(k, mask).view(B, -1, C // self.reduction)
                group_v = torch.masked_select(v, mask).view(B, -1, C)
                
                if group_q.size(1) > 0:
                    attn = torch.bmm(group_q, group_k.transpose(1, 2))
                    attn = F.softmax(attn, dim=-1)
                    group_attn = torch.bmm(attn, group_v)
                    
                    # 放回原位置
                    group_output = group_output.scatter_add(1, 
                        mask.expand_as(group_output).nonzero(as_tuple=True)[0:2] + 
                        (group_attn.view(-1, C),))
            
            outputs.append(group_output)
            scores.append(torch.ones(B, 1, device=q.device) / self.n_hashes)
        
        # 加权融合
        outputs = torch.stack(outputs, dim=1)  # [B, n_hashes, L, C]
        scores = torch.stack(scores, dim=1).unsqueeze(-1)  # [B, n_hashes, 1, 1]
        
        final_output = (outputs * scores).sum(dim=1)
        final_output = final_output.permute(0, 2, 1)  # [B, C, L]
        
        return final_output
    
    def forward(self, x):
        """前向传播"""
        identity = x
        
        # 生成Q, K, V
        q = self.query_conv(x)
        k = self.key_conv(x)
        v = self.value_conv(x)
        
        # 哈希注意力
        attn_output = self.hash_attention(q, k, v)
        
        # 输出变换
        out = self.out_conv(attn_output)
        out = self.gamma * out + identity
        out = self.norm(out)
        
        return out


# ==================== 集成示例 ====================

class SignalWithSparseAttention(nn.Module):
    """集成稀疏注意力的信号处理模块"""
    
    def __init__(self, input_channels=6, hidden_channels=64, output_channels=128):
        super(SignalWithSparseAttention, self).__init__()
        
        # 基础特征提取
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 稀疏注意力模块
        self.sparse_attention = MultiDomainSparseAttention(
            n_hashes=4,
            channels=hidden_channels,
            reduction=4,
            chunk_size=16
        )
        
        # 后续处理
        self.conv2 = nn.Conv1d(hidden_channels, output_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(output_channels)
        
        # 全局池化
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        """前向传播"""
        # 基础特征提取
        x = self.relu(self.bn1(self.conv1(x)))
        
        # 稀疏注意力
        x = self.sparse_attention(x)
        
        # 后续处理
        x = self.relu(self.bn2(self.conv2(x)))
        
        # 全局池化
        x = self.pool(x).squeeze(-1)
        
        return x


def test_attention_modules():
    """测试注意力模块"""
    print("测试注意力模块...")
    
    # 测试MultiDomainSparseAttention
    batch_size = 4
    channels = 64
    seq_len = 256
    
    input_tensor = torch.randn(batch_size, channels, seq_len)
    
    # 测试完整版
    attn = MultiDomainSparseAttention(n_hashes=4, channels=channels)
    output = attn(input_tensor)
    print(f"完整版输入形状: {input_tensor.shape}")
    print(f"完整版输出形状: {output.shape}")
    print(f"完整版参数数量: {sum(p.numel() for p in attn.parameters())}")
    
    # 测试简化版
    simple_attn = SimplifiedSparseAttention1D(channels=channels)
    output_simple = simple_attn(input_tensor)
    print(f"\n简化版输入形状: {input_tensor.shape}")
    print(f"简化版输出形状: {output_simple.shape}")
    print(f"简化版参数数量: {sum(p.numel() for p in simple_attn.parameters())}")
    
    # 测试集成模块
    integrated = SignalWithSparseAttention(
        input_channels=6,
        hidden_channels=64,
        output_channels=128
    )
    test_signal = torch.randn(batch_size, 6, 1024)
    output_integrated = integrated(test_signal)
    print(f"\n集成模块输入形状: {test_signal.shape}")
    print(f"集成模块输出形状: {output_integrated.shape}")
    print(f"集成模块参数数量: {sum(p.numel() for p in integrated.parameters())}")
    
    print("\n所有测试通过！")


if __name__ == "__main__":
    test_attention_modules()
