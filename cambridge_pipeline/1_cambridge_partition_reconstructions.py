from pathlib import Path
import pycolmap
import os
from hloc.pipelines.Cambridge.utils import scale_sfm_images
from hloc.extract_features import ImageDataset
# TODO: the lamar->colmap extraction contains cameras that seem to be iphone, check that lamar_to_colmap.py is correct.
if __name__ == "__main__":
    scenes = ["GreatCourt", "KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch"]
    for scene in scenes:
        base_path = f'datasets/cambridge/{scene}/colmap'
        sattler_path = Path(f"datasets/cambridge/CambridgeLandmarks_Colmap_Retriangulated_1024px/{scene}/empty_all")
        ref_model = pycolmap.Reconstruction(sattler_path) 
        for subset in ["train", "test"]:
            keys = []
            scene_path = Path(base_path, subset)
            image_dir =  Path(scene_path, "images")
            outputs = Path(f'{scene}_triangulated', subset)
            reference_model_split_path = Path(f"datasets/cambridge/cambridge-retriangulated/{scene}/{subset}")
            if os.path.exists(reference_model_split_path):
                print("reference_model_split_path exists")
                continue
            try:
                os.makedirs(reference_model_split_path)
                reference_model_split = pycolmap.Reconstruction()
                images = ImageDataset(image_dir, {}).names
                for imname in images:
                    im = ref_model.find_image_with_name(imname)
                    reference_model_split.add_image(im)
                    cam = ref_model.cameras[im.camera_id]
                    reference_model_split.add_camera(cam)
                reference_model_split.write_text(str(reference_model_split_path))
                scale_sfm_images(reference_model_split_path, reference_model_split_path, image_dir, ext = ".txt")
            except Exception as e:
                print(e)
                print("Continuing...")