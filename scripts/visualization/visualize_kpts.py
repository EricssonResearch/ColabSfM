import h5py

if __name__ == "__main__":
    with h5py.File("datasets/7scenes/7scenes_sfm_triangulated/chess/Test/superglue/feats-superpoint-n4096-r1600.h5", "r") as fd:
        print(fd['seq-03']['frame-000988.color.png']['keypoints'].__array__())