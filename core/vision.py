# 视觉提取：HSV颜色分割
# core/vision.py
import cv2
import numpy as np
from utils.config import AppConfig
import os

class ColorTrendExtractor:
    def __init__(self, config: AppConfig):
        self.cfg = config

    def extract(self, img_path: str) -> np.ndarray:
        img = None # 1. 先初始化，防止报错 'name not defined'
        
        # 2. 尝试读取图片 (兼容中文路径)
        try:
            # 使用 numpy 读取二进制流，再解码，完美绕过路径编码问题
            img_stream = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(img_stream, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[Warning] 读取出错: {e}")
            img = None

        # 3. 检查是否读取成功
        if img is None:
            # 打印绝对路径，方便排查
            abs_path = os.path.abspath(img_path)
            raise ValueError(f"无法读取或解码图片: {abs_path}")
        
        # 4. 图像处理流程
        img = cv2.resize(img, self.cfg.RESIZE_SIZE)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # --- 提取正向信号 (绿色) ---
        mask_pos = cv2.inRange(hsv, self.cfg.POS_LOWER, self.cfg.POS_UPPER)
        
        # --- 提取负向信号 (橙色/红色) ---
        # 橙色区间
        mask_neg1 = cv2.inRange(hsv, self.cfg.NEG_LOWER1, self.cfg.NEG_UPPER1)
        # 深红区间 (备用)
        mask_neg2 = cv2.inRange(hsv, self.cfg.NEG_LOWER2, self.cfg.NEG_UPPER2)
        # 合并
        mask_neg = cv2.bitwise_or(mask_neg1, mask_neg2)
        
        # 5. 积分计算
        return self._integrate_columns(mask_pos, mask_neg)

    def _integrate_columns(self, mask_pos, mask_neg):
        h, w = mask_pos.shape
        series = []
        bin_width = w / self.cfg.SAMPLING_POINTS
        
        for i in range(self.cfg.SAMPLING_POINTS):
            start_x = int(i * bin_width)
            end_x = int((i + 1) * bin_width)
            
            # 计算这一列有多少“绿色像素”和“橙色像素”
            pos_score = cv2.countNonZero(mask_pos[:, start_x:end_x])
            neg_score = cv2.countNonZero(mask_neg[:, start_x:end_x])
            
            # 净值 = 绿色 - 橙色
            # 如果全是绿色，就是正数；全是橙色，就是负数
            net_value = pos_score - neg_score
            series.append(net_value)
            
        return np.array(series, dtype=np.float32)