from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval, triangulation
import numpy as np
import pycolmap
import pickle as pkl
import os
from scantools.capture.session import Session


# TODO: the lamar->colmap extraction contains cameras that seem to be iphone, check that lamar_to_colmap.py is correct.
if __name__ == "__main__":
    base_path = 'datasets/tum/tum_testing'
    for subset in os.listdir(base_path):#['hl_2022-01-24-14-27-51-161.000', 'hl_2022-01-24-14-09-03-892.057']:
        keys = []
        scene_path = Path(base_path, subset)
        image_dir =  Path(scene_path, 'images')
        outputs = Path('tum_triangulated',subset)
        if os.path.exists(outputs):
            print("outputs exists")
            pass
        print(f"Creating a reconstruction for session {subset}")

        try:
            os.makedirs(outputs, exist_ok=True)
            sfm_pairs = outputs / 'pairs-netvlad.txt'
            sfm_dir = outputs / 'sfm_disk+lightglue'
            retrieval_conf = extract_features.confs['netvlad']
            feature_conf = extract_features.confs['disk-2000']
            matcher_conf = match_features.confs['disk+lightglue']
            image_ids = set()
            
            retrieval_path = extract_features.main(retrieval_conf, image_dir, outputs)
            pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched= 20)
            feature_path = extract_features.main(feature_conf, image_dir, outputs)
            match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

            reference_model_A_path = Path(scene_path.joinpath("input_model"))
            model = triangulation.main(sfm_dir, reference_model_A_path, image_dir, sfm_pairs, feature_path, match_path)
            visualization.visualize_sfm_2d(model, image_dir)
            model.write(sfm_dir)
        except Exception as e:
            print(e)
            print("Continuing...")