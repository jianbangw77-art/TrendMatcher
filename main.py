# main.py
# main.py
import os
import datetime
import matplotlib.pyplot as plt
import sys

# === 【修复乱码关键点】 ===
# 设置字体为 SimHei (黑体) 或 Microsoft YaHei (微软雅黑)
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题
# ========================

from core.loader import DataLoader
from core.vision import ColorTrendExtractor
from core.signal import SignalProcessor
from core.metrics import WeightedSimilarity
from utils.config import AppConfig

def setup_output_dir():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("outputs", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def main():
    cfg = AppConfig()
    
    # 这里的路径不需要改，DataLoader会自动扫描
    loader = DataLoader(root_dir="./dataset") 
    
    extractor = ColorTrendExtractor(cfg)
    processor = SignalProcessor(cfg)
    matcher = WeightedSimilarity(cfg)
    
    output_dir = setup_output_dir()
    print(f"🚀 任务开始，结果将保存至: {output_dir}")

    dataset = loader.scan()
    print(f"📂 发现 {len(dataset)} 组案例...\n")

    for case in dataset:
        print(f"正在分析案例: [{case.case_name}]")
        try:
            # 1. 处理主图
            raw_main = extractor.extract(case.main_path)
            clean_main = processor.preprocess(raw_main)
            
            # 2. 处理幅图
            for sub_path in case.sub_paths:
                sub_filename = os.path.basename(sub_path)
                
                raw_sub = extractor.extract(sub_path)
                clean_sub = processor.preprocess(raw_sub)
                
                score = matcher.compute(clean_main, clean_sub)
                
                # 判定
                result_text = "合格" if score > 0.8 else "不合格"
                print(f"   --> 对比 {sub_filename}: 得分 {score:.4f} ({result_text})")
                
                visualize_and_save(
                    clean_main, clean_sub, score, result_text,
                    output_dir, case.case_name, sub_filename
                )
                
        except Exception as e:
            # 打印详细错误，方便调试
            import traceback
            print(f"   [Error] {case.case_name} 崩溃了:")
            print(traceback.format_exc())

    print("\n✅ 处理完毕。")

def visualize_and_save(seq1, seq2, score, result, out_dir, case_name, sub_name):
    plt.figure(figsize=(10, 4))
    plt.plot(seq1, label='Main (Ref)', color='black', alpha=0.7)
    # 合格画绿线，不合格画红线
    line_color = 'green' if result == "合格" else 'red'
    plt.plot(seq2, label=f'Sub: {sub_name}', color=line_color, linestyle='--')
    
    plt.title(f"Case: {case_name} | Score: {score:.3f} [{result}]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    safe_sub_name = os.path.splitext(sub_name)[0]
    save_path = os.path.join(out_dir, f"{case_name}_VS_{safe_sub_name}.png")
    
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    main()