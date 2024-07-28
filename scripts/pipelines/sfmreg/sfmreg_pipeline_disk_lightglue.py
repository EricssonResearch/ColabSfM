from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval, triangulation
import numpy as np
import pycolmap
import pickle as pkl
import os

if __name__ == "__main__":
    num_subsets = 10
    mega_scenes = ['0015']#os.listdir('datasets/megadepth/MegaDepth_v1_SfM')
    for mega_scene in mega_scenes:
        print(f"Creating partial reconstructions for scene {mega_scene}")
        reference_model_path = Path(f'datasets/megadepth/MegaDepth_v1_SfM/{mega_scene}/sparse/manhattan/0')
        ref_reconstruction = pycolmap.Reconstruction(reference_model_path)
        images = Path(f'datasets/megadepth/MegaDepth_v1_SfM/{mega_scene}/images/')
        images_ids = list(ref_reconstruction.images.keys())
        min_num_obs = 100
        max_num_obs = 120
        import shutil
        for subset in range(num_subsets):
            outputs = Path(f'outputs/sfm/{mega_scene}/{subset}')
            if os.path.exists(outputs):
                print(f"Path {outputs} already exists, skipping")
                pass
            try:
                os.makedirs(outputs, exist_ok=True)
                sfm_pairs = outputs / 'pairs-netvlad.txt'
                sfm_dir = outputs / 'sfm_disk+lightglue'
                retrieval_conf = extract_features.confs['netvlad']
                feature_conf = extract_features.confs['disk-2000']
                matcher_conf = match_features.confs['disk+lightglue']
                image_ids = set()
                if os.path.exists(outputs / "image_ids"):
                    image_ids = pkl.load(outputs / "image_ids")
                else:
                    random_points = np.random.choice(ref_reconstruction.points3D, 50, replace = False)
                    for random_point in random_points:
                        new_image_ids = image_ids.union(set([el.image_id for el in ref_reconstruction.points3D[random_point].track.elements]))
                        if len(new_image_ids) >= min_num_obs:
                            if len(new_image_ids) <= max_num_obs:
                                image_ids = new_image_ids
                                break
                            else:
                                continue
                        image_ids = new_image_ids
                    pkl.dump(image_ids, open(outputs / "image_ids", "wb"))
                image_list_A = [ref_reconstruction.images[item].name for item in image_ids]
                
                retrieval_path = extract_features.main(retrieval_conf, images, outputs, image_list = image_list_A)
                pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=10)
                feature_path = extract_features.main(feature_conf, images, outputs, image_list = image_list_A)
                match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

                reference_model_A = pycolmap.Reconstruction()
                reference_model_A_path = outputs / 'reference_model'
                os.makedirs(reference_model_A_path, exist_ok=True)
                for im_id in image_ids:
                    im = ref_reconstruction.images[im_id]
                    cam = ref_reconstruction.cameras[im.camera_id]
                    reference_model_A.add_image(im)
                    reference_model_A.add_camera(cam)

                reference_model_A.write(reference_model_A_path)

                model = triangulation.main(sfm_dir, reference_model_A_path, images, sfm_pairs, feature_path, match_path)
                model.write(sfm_dir)
            except Exception as e:
                print(e)
                print("Continuing...")