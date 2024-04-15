import pycolmap
import numpy as np
import torch
import os
import json
import pickle as pkl


sfm_dir = "datasets/sfmreg/megadepth"
methods = ["sift", "sosnet"]
metas = []
split = "test"
test_scenes = ['0008', "0015", '0019', '0021', "0022", '0024', '0025', '0032', '0063', '1589']
versions = ["random","trajectories1", "trajectories2", "trajectories3", ]
metas = {k:[] for k in pkl.load(open(f"{sfm_dir}/pointclouds_megadepth/random/0000/sift_infos","rb")).keys()}
for version in versions:
    pointcloud_dir = f"{sfm_dir}/pointclouds_megadepth/{version}"
    scenes = os.listdir(pointcloud_dir)
    print(f"{len(scenes)=} for {version=}")
    for scene in scenes:
        if scene in test_scenes and split == "train":
            continue
        elif not scene in test_scenes and split == "test":
            continue
        for method in methods:
            try:
                meta = pkl.load(open(f"{pointcloud_dir}/{scene}/{method}_infos","rb"))
                metas = {k:metas[k]+meta[k] for k in metas.keys()}
            except:
                continue

all_overlaps = metas["overlaps"]
print(len(all_overlaps))
counts, bins = torch.histogram(torch.tensor(all_overlaps))
import matplotlib.pyplot as plt
plt.plot(bins[:-1], counts)
plt.xlim(0,1)
plt.savefig("overlap_histogram")
pkl.dump(metas, open(f"{sfm_dir}/sfmreg_combined_{split}_infos.pkl","wb"))
