import subprocess
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("experiment")
parser.add_argument("--task", default="train", required=False)
parser.add_argument("--sim3", action='store_true')

args = parser.parse_args()
checkpoint_path = "/proj/inter-op-slam/home/ejodhes/checkpoints/y24w7_predator_finetune_sim3/"
data_path = "/proj/inter-op-slam/home/ejodhes/data"
try:
    # If we can copy to a local drive we do it,
    shutil.copytree(f"{data_path}/colabsfm/pointclouds_megadepth", "/tmp/colabsfm/pointclouds_megadepth")
    # data_path = "/tmp"
    data_path = "/tmp/colabsfm/pointclouds_megadepth/"
except Exception as e:
    print(e)
    # Else we don't
    pass
log_path = "/proj/inter-op-slam/home/ejodhes/logs"

if args.sim3:
    subprocess.call(["python", args.experiment, "colabsfm/OverlapPredator/configs/"+args.task+"/colabsfm.yaml", "--data_root", data_path, "--checkpoint_dir", checkpoint_path, "--log_dir", log_path, "--sim3"])
else:
    subprocess.call(["python", args.experiment, "colabsfm/OverlapPredator/configs/"+args.task+"/colabsfm.yaml", "--data_root", data_path, "--checkpoint_dir", checkpoint_path, "--log_dir", log_path])