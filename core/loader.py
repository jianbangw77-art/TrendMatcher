# core/loader.py
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class CaseData:
    case_name: str
    main_path: str
    sub_paths: List[str]

class DataLoader:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.root}")

    def scan(self) -> List[CaseData]:
        """扫描 dataset 下的所有子文件夹"""
        cases = []
        
        # 遍历 dataset 下的每一个子目录 (例如 case_01, case_02)
        for case_dir in sorted(self.root.iterdir()):
            if case_dir.is_dir():
                main_img = self._find_main_image(case_dir)
                sub_imgs = self._find_sub_images(case_dir)
                
                if main_img and sub_imgs:
                    cases.append(CaseData(
                        case_name=case_dir.name,
                        main_path=str(main_img),
                        sub_paths=[str(p) for p in sub_imgs]
                    ))
                else:
                    print(f"[Warning] 跳过 {case_dir.name}: 缺少主图或幅图")
        
        return cases

    def _find_main_image(self, folder: Path) -> Optional[Path]:
        """寻找文件名包含 'main' 或 'zhu' 的图片"""
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            # 优先找名字里带 'main' 的
            matches = list(folder.glob(f"*main*{ext[1:]}"))
            if not matches:
                matches = list(folder.glob(f"*主*{ext[1:]}"))
            
            if matches:
                return matches[0] # 返回找到的第一张
        return None

    def _find_sub_images(self, folder: Path) -> List[Path]:
        """寻找除主图以外的所有图片"""
        subs = []
        main_img = self._find_main_image(folder)
        main_name = main_img.name if main_img else "INVALID_NAME"
        
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img_path in folder.glob(ext):
                # 排除掉主图
                if img_path.name != main_name:
                    subs.append(img_path)
        return sorted(subs)