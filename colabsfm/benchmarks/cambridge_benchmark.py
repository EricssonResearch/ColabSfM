import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from colabsfm.metrics import compute_feature_matching_recall, compute_inlier_ratio
from colabsfm.models.model import OmniGlue
from colabsfm.utils import get_best_device, collate_ragged
from .benchmark_utils import ransac_pose_estimation_correspondences
import colabsfm


class TDMatchBenchmark:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, shuffle = False, batch_size = 1, num_workers = 0, collate_fn = collate_ragged) # NOTE: Important to not shuffle!
    def benchmark(self, model: OmniGlue, device = get_best_device(), refiner = True):
        IRs = []
        print("Start running TDMatch benchmark.")
        overlaps = []
        nmatches = []
        RRs = []
        for idx, batch in enumerate(tqdm(self.dataloader)):
            data = model.match(batch['pc_A'].to(device), batch['pc_B'].to(device), batch['normals_A'].to(device), batch['normals_B'].to(device))
            for stage in ["refiner"]:
                if stage == "refiner":
                    p = data["refiner_points_A"]
                    q = data["refiner_points_B"]
                    matches = data["refiner_matches"]
                else:
                    p = data["keypoints_A"]
                    q = data["keypoints_B"]
                    matches = data["matches"]
                nmatches.append(len(matches[0]))
                threshold = batch.get("threshold",None)
                IR = compute_inlier_ratio(p[matches[0],matches[1]], q[matches[0],matches[2]], batch["tsfm"].to(device), threshold = threshold)
                IRs.append(IR)
                eval_out = dict(src_corr_pts = data['src_corr_pts'], 
                                tgt_corr_pts = data['tgt_corr_pts'], 
                                confidence = data['confidence'], 
                                rot = batch["tsfm"][:3,:3].cpu(),
                                trans = batch["tsfm"][:3,3:].cpu())
                try:
                    # This cache might not exist, so put in try catch block
                    eval_cache = f"eval_cache/{model.signature}"
                    import os
                    os.makedirs(eval_cache, exist_ok = True)
                    if False:
                        torch.save(eval_out, f'{eval_cache}/{idx}.pth')
                except:
                    pass
            overlaps.append(batch["overlap"][0])
        low_overlap = torch.tensor(overlaps) < 0.3
        mean_num_matches = torch.tensor(nmatches).float().mean().item()
        IRs = torch.tensor(IRs)
        LO_FMR = compute_feature_matching_recall(IRs[low_overlap])
        LO_IR = IRs[low_overlap].mean().item()
        FMR = compute_feature_matching_recall(IRs[~low_overlap])
        IR = IRs[~low_overlap].mean().item()
        if colabsfm.LOGGER is not None:
            colabsfm.LOGGER.add_scalars('data/3dmatch_benchmark', 
                                       dict(LO_FMR = LO_FMR, LO_IR = LO_IR, FMR = FMR, IR = IR, mean_num_matches = mean_num_matches), 
                                       global_step = colabsfm.GLOBAL_STEP)
        else:
            print(f"{mean_num_matches=}")
            print(f"{LO_FMR=}")
            print(f"{LO_IR=}")
            print(f"{FMR=}")
            print(f"{IR=}")