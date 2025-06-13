import pycolmap
from colabsfm.api import RefineRoITr

def test_se3():
    device = "cuda"
    reconstruction_path = "assets/5014"
    sfm_model_A = pycolmap.Reconstruction(reconstruction_path)
    sfm_model_B = pycolmap.Reconstruction(reconstruction_path)
    weights_path = "data/colabsfm/models/sfmreg_finetuned.pth"
    sfm_registrator = RefineRoITr(weights_path=weights_path, mode = "se3")
    registration_results = sfm_registrator.register_reconstructions(sfm_model_A, sfm_model_B)
    print(f"{registration_results['num_matches']=}")
    print(f"{registration_results['transformation']=}")

def test_sim3():
    device = "cuda"
    reconstruction_path = "assets/5014"
    sfm_model_A = pycolmap.Reconstruction(reconstruction_path)
    sfm_model_B = pycolmap.Reconstruction(reconstruction_path)
    weights_path = "data/colabsfm/models/sfmreg_only_sim3.pth"
    sfm_registrator = RefineRoITr(weights_path=weights_path, mode = "sim3")
    registration_results = sfm_registrator.register_reconstructions(sfm_model_A, sfm_model_B)
    print(f"{registration_results['num_matches']=}")
    print(f"{registration_results['transformation']=}")



if __name__ == "__main__":
    test_se3()
    test_sim3()