from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval, triangulation
import numpy as np
import pycolmap
import pickle as pkl
import os
#from scantools.capture.session import Session
import torch
#from hloc.pipelines.7scenes.utils import scale_sfm_images

if __name__ == "__main__":
    scenes = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
    for scene in scenes:
        base_path = f'data/7scenes_sfm_triangulated/{scene}/triangulated'
        for subset in ["train", "test"]:
            keys = []
            scene_path = Path(base_path, subset)
            image_dir =  Path(scene_path, "images")
            outputs = Path(f'datasets/7scenes_sfm_triangulated/{scene}', subset)
            sfm_dir = outputs / 'sift-sattler'
            if os.path.exists(sfm_dir):
                print("sfm_dir exists")
                continue
            print(f"Creating a triangulation for session {subset}")
            try:
                os.makedirs(outputs, exist_ok=True)
                sfm_pairs = outputs / 'pairs-netvlad.txt'
                retrieval_conf = extract_features.confs['netvlad']
                feature_conf = extract_features.confs['sift-1024']
                matcher_conf = match_features.confs['NN-ratio']
                
                reference_model_A_path = Path(f"datasets/7scenes/7scenes-retriangulated/{scene}/{subset}")

                added_camera_ids = set()
                retrieval_path = extract_features.main(retrieval_conf, image_dir, outputs)
                torch.cuda.empty_cache()
                pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched = 40)
                feature_path = extract_features.main(feature_conf, image_dir, outputs)
                match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)
                model = triangulation.main(sfm_dir, reference_model_A_path, image_dir, sfm_pairs, feature_path, match_path)
                #visualization.visualize_sfm_2d(model, image_dir)
                model.write_text(str(sfm_dir))
            except Exception as e:
                print(e)
                print("Continuing...")