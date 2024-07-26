from pathlib import Path
from hloc import extract_features, match_features, pairs_from_retrieval, triangulation
import os
import torch

if __name__ == "__main__":
    scene = "GreatCourt"
    scenes = ["GreatCourt", "KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch"]
    for scene in scenes:
        base_path = f'datasets/cambridge/{scene}/colmap'
        for subset in ["train", "test"]:
            keys = []
            scene_path = Path(base_path, subset)
            image_dir =  Path(scene_path, "images")
            outputs = Path(f'datasets/cambridge/{scene}_triangulated', subset)
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
                
                reference_model_A_path = Path(f"datasets/cambridge/cambridge-retriangulated/{scene}/{subset}")

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