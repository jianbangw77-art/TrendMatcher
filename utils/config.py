# 参数配置
# utils/config.py
from dataclasses import dataclass

@dataclass
class AppConfig:
    # 图像处理参数
    RESIZE_SIZE: tuple = (800, 600)
    SAMPLING_POINTS: int = 100
    
    # =========================================================
    # 核心修改：针对你的图片颜色 (绿涨/橙跌) 进行调整
    # =========================================================
    
    # 【正向信号】：绿色 (Green)
    # 绿色在 HSV (OpenCV) 中通常在 H=35 到 H=90 之间
    POS_LOWER: tuple = (35, 43, 46)
    POS_UPPER: tuple = (90, 255, 255)

    # 【负向信号】：橙色 (Orange) 
    # 橙色在 HSV 中通常在 H=10 到 H=25 之间
    # 为了保险，我们设置 0-25，这样也能覆盖一点红色
    NEG_LOWER1: tuple = (0, 43, 46)
    NEG_UPPER1: tuple = (25, 255, 255)
    
    # 备用区间：防止它偏红 (H=160-180)，留着也没坏处
    NEG_LOWER2: tuple = (160, 43, 46)
    NEG_UPPER2: tuple = (180, 255, 255)

    # 算法权重
    WEIGHT_EXPONENT: float = 1.0
    SIGMA_SMOOTH: float = 2.0