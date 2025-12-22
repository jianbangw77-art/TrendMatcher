# 核心算法：加权相似度计算
import numpy as np
from utils.config import AppConfig

class WeightedSimilarity:
    def __init__(self, config: AppConfig):
        self.cfg = config

    def compute(self, target: np.ndarray, candidate: np.ndarray):
        """
        计算加权余弦相似度。
        """
        # 确保长度对齐
        min_len = min(len(target), len(candidate))
        t = target[:min_len]
        c = candidate[:min_len]
        
        # 生成权重：从 0.5 线性增长到 1.5
        # 也可以改为指数增长 np.exp(...) 以更强调近期数据
        weights = np.linspace(0.5, 1.5, min_len)
        
        # 加权点积
        weighted_dot = np.sum(weights * t * c)
        
        # 加权模长 (L2 Norm)
        norm_t = np.sqrt(np.sum(weights * t**2))
        norm_c = np.sqrt(np.sum(weights * c**2))
        
        if norm_t == 0 or norm_c == 0:
            return 0.0
            
        similarity = weighted_dot / (norm_t * norm_c)
        return similarity