from pathlib import Path
from hloc import extract_features, match_features, reconstruction, triangulation, visualization, pairs_from_retrieval, triangulation
import numpy as np
import pycolmap
import pickle as pkl
import os
import torch

TRIANGULATE = True
# TODO: the lamar->colmap extraction contains cameras that seem to be iphone, check that lamar_to_colmap.py is correct.
if __name__ == "__main__":
    base_path = Path("/home/ejaealb/work/hloc/datasets/sfmreg/Quad/ArtsQuad_dataset/")
    keys = []
    image_dir =  Path(base_path, "images")
    image_list = []
    if TRIANGULATE:
        print("Creating a triangulation")
        outputs = Path(base_path, "sfm_tr")
        with open(base_path / "input_model" / "images.txt") as f:
            for l in f.readlines():
                if len(l) < 4: continue
                image_list.append(l.strip().split()[-1])
        print("Detected %d images" % len(image_list))
    else:
        print("Creating a reconsturction")
        outputs = Path(base_path, "sfm")
        

    os.makedirs(outputs, exist_ok=True)
    sfm_pairs = outputs / 'pairs-netvlad.txt'
    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['sift-1024']
    matcher_conf = match_features.confs['NN-ratio']
    sfm_dir = outputs / 'sift'   
    
    retrieval_path = extract_features.main(retrieval_conf, image_dir, outputs, image_list=image_list)
    torch.cuda.empty_cache()
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched = 20)
    feature_path = extract_features.main(feature_conf, image_dir, outputs, image_list=image_list)
    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

    if TRIANGULATE:
        ref_model_path = base_path / "input_model"
        model = triangulation.main(sfm_dir, ref_model_path, image_dir, sfm_pairs, feature_path, match_path)
    else:
        model = reconstruction.main(sfm_dir, image_dir, sfm_pairs, feature_path, match_path)
    #visualization.visualize_sfm_2d(model, image_dir)
    model.write(sfm_dir)