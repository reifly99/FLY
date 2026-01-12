# config.py - 更新版本
class ModelConfig:
    """模型配置类"""
    
    # KAN分类器配置
    KAN_CONFIG = {
        'classifier_type': 'kan',
        'kan_config': {
            'grid_size': 5,
            'spline_order': 3,
            'scale_noise': 0.1,
            'scale_base': 1.0,
            'scale_spline': 1.0,
            'grid_eps': 0.02,
            'grid_range': [-1, 1],
        },
        'reg_weight': 0.01  # 正则化权重
    }
    
    # 稀疏注意力配置
    SPARSE_ATTENTION_CONFIG = {
        'use_sparse_attention': True,
        'sparse_attention_params': {
            'n_hashes': 4,
            'reduction': 4,
            'chunk_size': 16,
            'res_scale': 1
        },
        'use_enhanced_extractor': False  # 是否使用增强版特征提取器
    }
    
    # 线性分类器配置
    LINEAR_CONFIG = {
        'classifier_type': 'linear',
        'reg_weight': 0.0  # 无正则化
    }
    
    # MLP分类器配置
    MLP_CONFIG = {
        'classifier_type': 'mlp',
        'hidden_dims': [128, 64],  # MLP隐藏层维度
        'reg_weight': 0.0
    }
    
    @classmethod
    def get_config(cls, classifier_type='kan', use_sparse_attention=True):
        """获取指定类型的配置"""
        base_config = {}
        
        if classifier_type == 'kan':
            base_config = cls.KAN_CONFIG
        elif classifier_type == 'linear':
            base_config = cls.LINEAR_CONFIG
        elif classifier_type == 'mlp':
            base_config = cls.MLP_CONFIG
        else:
            raise ValueError(f"未知的分类器类型: {classifier_type}")
        
        # 添加稀疏注意力配置
        if use_sparse_attention:
            base_config.update(cls.SPARSE_ATTENTION_CONFIG)
        
        return base_config
