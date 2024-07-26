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
    base_path = Path('datasets/7scenes/')
    triangulated_path = base_path / "7scenes_sfm_triangulated"
    for scene in scenes:
        scene_path = triangulated_path / scene
        for subset in ["Train", "Test"]:
            subset_path = scene_path / subset
            image_dir =  base_path / scene
            outputs = subset_path
            sfm_dir = subset_path / "sfm"
            print(f"Creating a triangulation for {scene=} {subset=}")
            ref_reconstr = pycolmap.Reconstruction(subset_path)
            im_names = [im.name for idx, im in ref_reconstr.images.items()]
            os.makedirs(outputs, exist_ok=True)
            sfm_pairs = outputs / 'pairs-netvlad.txt'
            retrieval_conf = extract_features.confs['netvlad']
            feature_conf = extract_features.confs['sosnet-1024']
            matcher_conf = match_features.confs['NN-ratio']
            reference_model_path = subset_path

            retrieval_path = extract_features.main(retrieval_conf, image_dir, outputs, image_list=im_names)
            pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched = 10)
            feature_path = extract_features.main(feature_conf, image_dir, outputs, image_list=im_names)
            match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)
            model = triangulation.main(sfm_dir, reference_model_path, image_dir, sfm_pairs, feature_path, match_path)
            model.write_text(str(sfm_dir))
