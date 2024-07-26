from pathlib import Path
import pycolmap
import os
from hloc.pipelines.Cambridge.utils import scale_sfm_images
from hloc.extract_features import ImageDataset
# TODO: the lamar->colmap extraction contains cameras that seem to be iphone, check that lamar_to_colmap.py is correct.
if __name__ == "__main__":
    scenes = ["KingsCollege", "GreatCourt", "OldHospital", "ShopFacade", "StMarysChurch"]
    base_path = Path('datasets/cambridge')
    for scene in scenes:
        scene_path = base_path / scene
        sattler_path = Path(f"datasets/cambridge/CambridgeLandmarks_Colmap_Retriangulated_1024px/{scene}/empty_all")
        ref_model = pycolmap.Reconstruction(sattler_path) 
        for subset in ["train", "test"]:
            with open(scene_path / f"dataset_{subset}.txt", "r") as f:
                reference_model_split = pycolmap.Reconstruction()
                split_seq_nums = set([seqname[3:].split("/")[0] for seqname in f.read().split("\n")[:-1] if seqname[:3] == "seq"])
                outputs = Path(f'{scene}_triangulated', subset)
                reference_model_split_path = Path(f"datasets/cambridge/cambridge-retriangulated/{scene}/{subset}")
                if os.path.exists(reference_model_split_path):
                    if any(Path(reference_model_split_path).iterdir()):
                        print("reference_model_split_path exists")
                        continue
                else:
                    os.makedirs(reference_model_split_path)
                for seq_num in split_seq_nums:
                    seq_name_formatted = f"seq{seq_num}"
                    image_dir =  scene_path / seq_name_formatted
                    images = ImageDataset(image_dir, {}).names
                    for imname in images:
                        im = ref_model.find_image_with_name(Path(seq_name_formatted) / imname)
                        if im is None:
                            continue
                        reference_model_split.add_image(im)
                        cam = ref_model.cameras[im.camera_id]
                        reference_model_split.add_camera(cam)
                reference_model_split.write_text(str(reference_model_split_path))
                scale_sfm_images(reference_model_split_path, reference_model_split_path, scene_path, ext = ".txt")
