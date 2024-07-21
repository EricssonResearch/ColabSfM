from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval, triangulation
import numpy as np
import pycolmap
import pickle as pkl
import os
#from scantools.capture.session import Session
import torch
from hloc.pipelines.Cambridge.utils import scale_sfm_images
from hloc.extract_features import ImageDataset

if __name__ == "__main__":
    scenes = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
    for scene in scenes:
        base_path = Path(f'datasets/7scenes/{scene}')
        sattler_path = Path(f"datasets/7scenes/7scenes_sfm_triangulated/{scene}/triangulated")
        ref_model = pycolmap.Reconstruction(sattler_path) 
        for subset in ["Train", "Test"]:
            reference_model_split_path = Path(f"datasets/7scenes/7scenes_sfm_triangulated/{scene}/{subset}")
            if os.path.exists(reference_model_split_path):
                print("reference_model_split_path exists")
                #continue
            try:
                os.makedirs(reference_model_split_path, exist_ok=True)
                with open(base_path / f"{subset}Split.txt", "r") as f:
                    split_seq_nums = [seqname.split("sequence")[1] for seqname in f.read().split("\n")[:-1]]
                    reference_model_split = pycolmap.Reconstruction()
                    for seq_num in split_seq_nums:
                        seq_num_formatted = f"seq-{int(seq_num):02d}"
                        image_dir =  Path(base_path, seq_num_formatted)
                        images = ImageDataset(image_dir, {}).names
                        for imname in images:
                            im = ref_model.find_image_with_name(Path(seq_num_formatted) / imname)
                            if im is None:
                                continue
                            reference_model_split.add_image(im)
                            cam = ref_model.cameras[im.camera_id]
                            try:
                                reference_model_split.add_camera(cam)
                            except:
                                pass # pycolmap is a bit stupid and throws exceptions if you try to add already added cameras.
                    reference_model_split.write_text(str(reference_model_split_path))
            except Exception as e:
                print(e)
                print("Continuing...")