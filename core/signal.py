# 信号处理：归一化与重采样
import numpy as np
from scipy.ndimage import gaussian_filter1d
from utils.config import AppConfig

class SignalProcessor:
    def __init__(self, config: AppConfig):
        self.cfg = config

    def preprocess(self, series: np.ndarray) -> np.ndarray:
        # 1. 平滑去噪
        smoothed = gaussian_filter1d(series, sigma=self.cfg.SIGMA_SMOOTH)
        
        # 2. Z-Score 归一化 (Standardization)
        # 公式: (x - mean) / std
        # 这样处理后，所有序列均值为0，方差为1，只比较“形状”而不比较“绝对高度”
        mean_val = np.mean(smoothed)
        std_val = np.std(smoothed)
        
        if std_val < 1e-6: # 防止除零
            return np.zeros_like(smoothed)
            
        normalized = (smoothed - mean_val) / std_val
        return normalized