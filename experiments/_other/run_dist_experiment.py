import subprocess
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("experiment")
parser.add_argument("num_gpus")
args = parser.parse_args()
checkpoint_path = "/proj/inter-op-slam/home/ejodhes/checkpoints"
data_path = "/proj/inter-op-slam/home/ejodhes/data"
try:
    # If we can copy to a local drive we do it,
    shutil.copytree(f"{data_path}/indoor", "/tmp/indoor")
    shutil.copytree(f"{data_path}/indoor_normals", "/tmp/indoor_normals")
    data_path = "/tmp"
except Exception as e:
    print(e)
    # Else we don't
    pass
log_path = "/proj/inter-op-slam/home/ejodhes/logs"


subprocess.call(["torchrun",f"--nproc_per_node={args.num_gpus}", "--nnodes=1", "--rdzv_backend=c10d", 
                 args.experiment, "colabsfm/configs/train/tdmatch.yaml", 
                 "--data_root", data_path, "--checkpoint_dir", checkpoint_path, "--log_dir", log_path])