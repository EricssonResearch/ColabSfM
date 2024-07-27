from pathlib import Path
from hloc import extract_features, match_features, pairs_from_retrieval, triangulation
import os
import torch
import pycolmap

if __name__ == "__main__":
    scenes = ["GreatCourt", "KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch"]
    for scene in scenes:
        for subset in ["train", "test"]:
            keys = []
            base_path = Path('datasets/cambridge')
            scene_path = Path(base_path, scene)
            image_dir = scene_path
            reference_model_path = base_path / f"cambridge-retriangulated/{scene}/{subset}"
            ref_reconstr = pycolmap.Reconstruction(reference_model_path)
            outputs = reference_model_path / 'sift'
            if os.path.exists(outputs):
                if any(outputs.iterdir()):
                    print("outputs exists")
                    continue
            print(f"Creating a triangulation for session {subset}")
            os.makedirs(outputs, exist_ok=True)
            sfm_pairs = outputs / 'pairs-netvlad.txt'
            retrieval_conf = extract_features.confs['netvlad']
            feature_conf = extract_features.confs['sift-1024']
            matcher_conf = match_features.confs['NN-ratio']
            im_names = [im.name for idx, im in ref_reconstr.images.items()]
            added_camera_ids = set()
            retrieval_path = extract_features.main(retrieval_conf, image_dir, outputs, image_list=im_names)
            torch.cuda.empty_cache()
            pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched = 40)
            feature_path = extract_features.main(feature_conf, image_dir, outputs, image_list=im_names)
            match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)
            model = triangulation.main(outputs, reference_model_path, image_dir, sfm_pairs, feature_path, match_path)
            model.write_text(str(outputs))
