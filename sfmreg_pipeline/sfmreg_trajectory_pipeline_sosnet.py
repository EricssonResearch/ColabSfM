from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval, triangulation
import numpy as np
import pycolmap
import pickle as pkl
import os
import torch
import argparse

def generate_trajectories(positions, position_distances, rot_distances, max_num_trajectories = 20, min_num_images = 75, max_num_images = 300):
    remaining_inds = set(range(len(positions)))
    trajectories = []
    trajectory_lengths = np.arange(min_num_images, min(max_num_images, len(remaining_inds)//2))
    for t in range(max_num_trajectories):
        if len(remaining_inds) < max(trajectory_lengths):
            break
        trajectory = []
        current_ind = remaining_inds.pop()
        trajectory.append(current_ind)
        distances = position_distances + 1*np.random.rand()*rot_distances
        trajectory_length = np.random.choice(trajectory_lengths)
        for t in range(trajectory_length):
            closest_ind = list(remaining_inds)[distances[current_ind, list(remaining_inds)].argmin().item()]
            remaining_inds.remove(closest_ind)
            current_ind = closest_ind
            trajectory.append(current_ind)
        trajectories.append(trajectory)
    return trajectories

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type = str)
    args = parser.parse_args()
    
    mega_scenes = os.listdir('datasets/megadepth/MegaDepth_v1_SfM')
    output_path = args.output_path
    #mega_scenes = ['0008', "0015", '0019', '0021', "0022", '0024', '0025', '0032', '0063', '1589']
    for mega_scene in np.random.permutation(mega_scenes):
        scene_path = Path(output_path).joinpath(Path(f'reconstructions/sfm_trajectory/{mega_scene}'))
        if os.path.exists(scene_path):
            print(f"{scene_path} exists, skipping.")
            continue
        print(f"Creating partial reconstructions for scene {mega_scene}")
        reference_model_path = Path(f'datasets/megadepth/MegaDepth_v1_SfM/{mega_scene}/sparse/manhattan/0')
        try:
            ref_reconstruction = pycolmap.Reconstruction(reference_model_path)
        except Exception as e:
            print(e)
            continue
        images = Path(f'datasets/megadepth/MegaDepth_v1_SfM/{mega_scene}/images/')
        # TODO: these positions are technically up to scale, which means that the distances
        # may vary between reconstruction, I'm viewing it as some sort of augmentation
        # but future work might be to use e.g. number of overlapping 3D points to generate trajectories
        # or to normalize the positions (but normalizing might have other issues)
        positions = [im.projection_center() for im in ref_reconstruction.images.values()]
        positions = torch.from_numpy(np.array(positions))
        qvecs = np.array([im.qvec for im in ref_reconstruction.images.values()])
        qvecs = torch.from_numpy(qvecs)
        rot_distances = torch.acos(2 * (qvecs@qvecs.T)**2 - 1)
        position_distances = torch.cdist(positions, positions)
        import shutil
        sorted_to_name = {idx:ref_reconstruction.images[item].name for idx, item in enumerate(ref_reconstruction.images.keys())}
        sorted_to_id = {idx:item for idx, item in enumerate(ref_reconstruction.images.keys())}
    
        remaining_inds = set(range(len(positions)))
        os.makedirs(scene_path, exist_ok=True)
        try:
            if os.path.exists(scene_path / 'trajectories.pkl'):
                trajectories = pkl.load(open(scene_path / 'trajectories.pkl', 'rb'))
                print(f"Loaded trajectories for {outputs}")
            else:
                trajectories = generate_trajectories(positions, position_distances, rot_distances, max_num_trajectories = 20)
                pkl.dump(trajectories, open(scene_path / 'trajectories.pkl','wb'))
        except Exception as e:
            print("Failed to create trajectory with error")
            print(e)
            continue
        for idx, trajectory in enumerate(trajectories):
            outputs = Path(scene_path, Path(f"trajectory_{idx}"))
            try:
                traj_ids = [sorted_to_id[i] for i in trajectory]
                traj_names = [sorted_to_name[i] for i in trajectory]
                os.makedirs(outputs, exist_ok=True)
                sfm_pairs = outputs / 'pairs-netvlad.txt'
                sfm_dir = outputs / 'sosnet'
                if os.path.exists(sfm_dir):
                    print(f"Reconstruction {sfm_dir} already exists, continue")
                    continue
                np.save(outputs / 'trajectory_ids', traj_ids)
                retrieval_conf = extract_features.confs['netvlad']
                feature_conf = extract_features.confs['sosnet-1024']
                matcher_conf = match_features.confs['NN-ratio']
                retrieval_path = extract_features.main(retrieval_conf, images, outputs, image_list = traj_names)
                torch.cuda.empty_cache()
                pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched= min(40, len(traj_names)))
                feature_path = extract_features.main(feature_conf, images, outputs, image_list = traj_names)
                match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

                reference_model_A = pycolmap.Reconstruction()
                reference_model_A_path = outputs / 'reference_model'
                added_camera_ids = set()
                os.makedirs(reference_model_A_path, exist_ok=True)
                for im_id in traj_ids:
                    im = ref_reconstruction.images[im_id]
                    reference_model_A.add_image(im)
                    if im.camera_id not in added_camera_ids:
                        cam = ref_reconstruction.cameras[im.camera_id]
                        reference_model_A.add_camera(cam)
                    added_camera_ids.add(im.camera_id)
                reference_model_A.write(reference_model_A_path)
                model = triangulation.main(sfm_dir, reference_model_A_path, images, sfm_pairs, feature_path, match_path)
                model.write(sfm_dir)
            except Exception as e:
                print(e)
                print("Continuing...")