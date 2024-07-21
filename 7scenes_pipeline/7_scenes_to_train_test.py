from pathlib import Path
import numpy as np
import os
import shutil
#from scantools.capture.session import Session
import torch
#from hloc.pipelines.Cambridge.utils import scale_sfm_images
#from hloc.extract_features import ImageDataset

if __name__ == "__main__":
    scenes = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
    for scene in scenes:
        base_path = Path(f'datasets/7scenes/{scene}')
        split_path = Path(f"datasets/7scenes/{scene}/splits")
        for subset in ["Train", "Test"]:
            with open(base_path / f"{subset}Split.txt", "r") as f:
                split_seq_nums = [seqname.split("sequence")[1] for seqname in f.read().split("\n")[:-1]]
                for seq_num in split_seq_nums:
                    shutil.copytree(
                        base_path / f"seq-{int(seq_num):02d}", 
                        split_path / subset / "images/", 
                        dirs_exist_ok=True)