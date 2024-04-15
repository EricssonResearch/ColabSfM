# SfMReg Models
Maintainer: Johan Edstedt
Contact: johan.edstedt@liu.se

Here we provide instructions for training sfmreg registration models.

## Namings

I've used sfmreg, omnireg, and recently SfMReger

## Prepare data

Install hloc

## Creating SfMReg from Megadepth

1. Folder structure should be datasets/megadepth/MegaDepth_v1_SfM/... and make sure to have 
2. Run ```python sfmreg_pipeline/sfmreg_trajectory_pipeline_sift.py --output_path datasets/sfmreg```, this creates the sift triangulations. This may take a while (like 2-3 days), but can be speeded up by just running multiple copies of the same script in parallel.
4. (Optional) We can create multiple such datasets, which creates more dataaugmentation optionally
3. Run ```python create_combined_sfmreg.py --output_path datasets/sfmreg```
5. We use the format used in RoITr with "info" files for the pair sets


## Cambridge

The Cambridge scenes are downloaded and preprocessed, available at ```/media/fractal/Storage/datasets/cambridge```
I've used the retriangulation from Torsten Sattler instead of the original for the ground truth poses and intrinsics. (it's the standard in HLoc)
His retriangulation uses 1024 longer size, so we reconvert back to original image size and split the images. 
This was done with the script ```cambridge_pipeline/cambridge_partition_reconstructions.py```.
I then ran the script ```cambridge_pipeline/cambridge_pipeline_sift_sattler.py```
Then the script ```cambridge_pipeline/create_cambridge_bench.py```

## Quad6k

Available in [Quad](Quad)

## TUM

Available in [tum_pipeline](tum_pipeline)


## Install

1. Create a new conda env:
```conda create -n omnireg python=3.10```
```conda activate omnireg```
2. pip install the requirments in development mode
```pip install -e .```
3. Install cuda-toolkit that matches whatever is used by pytorch. You can try something like:
```conda install -c "nvidia/label/cuda-12.2.0" cuda``` (should work for latest pytorch)
4. We need to build the pointops library. This can be done by 
```bash
cd third_party/pointops
pip install .
```
We can optionally make it into a wheel (which is easy to then use in a docker container)
```bash
cd third_party/pointops
pip wheel .
pip install pointops-0.0.0-cp310-cp310-linux_x86_64.whl # wheel should have a name similar to (cp310 means c-python 3.10, and linux_x86_64 means which architectures it is compatible with)
```
If you're unlucky, this wheel might not work on other devices. You can then try ```auditwheel``` (https://pypi.org/project/auditwheel/).
5. (Optional) Install the cpp_wrappers for predator
```bash
cd omnireg/OverlapPredator/cpp_wrappers
sh compile_wrappers.sh
cd ../../..
```


## Code Structure

Most code is in the [omnireg](omnireg) folder. We mainly use RoITr as backbone, and have therefore fully integrated it into the codebase in the  [omnireg/roitr](omnireg/roitr) folder.

The final experiments used to produce tables etc can be found in [experiments/final_experiments](experiments/final_experiments).

The pretrained weights should be in the pretrained folder. This might be empty for you, I've put a copy of the weights here: [/media/fractal/Storage/ejodhes/pretrained](/media/fractal/Storage/ejodhes/pretrained).

Results of finetuning runs are assumed to be in the "workspace" folder. This might be empty for you, I've put a copy of some experiments here: [/media/fractal/Storage/ejodhes/workspace](/media/fractal/Storage/ejodhes/workspace).

## Finetuning on SfMReg

Run 
```bash
python experiments/final_experiments/y23w47_sfmreg_combined_lo.py
```

## Evaluating
The evaluation is still quite manual. Running the experiments on SfMReg should just be running, e.g.

```bash
# SFMReger
python experiments/final_experiments/sfmreger_eval_finetuned.py omnireg/roitr/configs/val/sfmreg.yaml
python experiments/final_experiments/sfmreger_eval_non_finetuned.py omnireg/roitr/configs/val/sfmreg.yaml
# RoiTr
python experiments/final_experiments/roi_tr_eval_sfmreg.py omnireg/roitr/configs/val/sfmreg.yaml --finetuned
python experiments/final_experiments/roi_tr_eval_sfmreg.py omnireg/roitr/configs/val/sfmreg.yaml 
# GeoTransformer
python experiments/final_experiments/geotransformer_eval_sfmreg.py omnireg/roitr/configs/val/sfmreg.yaml --backbone=3dmatch
# OverlapPredator
python experiments/final_experiments/predator_eval_sfmreg.py omnireg/OverlapPredator/configs/test/sfmreg.yaml
```

On Cambridge I've made some things kind of automatic, for example our finetuned method:
```bash
# SFMReger
python experiments/final_experiments/sfmreger_match_finetuned_cambridge.py
python experiments/final_experiments/sfmreger_match_non_finetuned_cambridge.py
# RoiTr
...
# GeoTransformer
python experiments/final_experiments/geotransformer_eval_cambridge.py
# OverlapPredator
python experiments/final_experiments/predator_eval_cambridge.py
```

On Quad6k:
```bash
# SFMReger
python experiments/final_experiments/sfmreger_eval_finetuned.py omnireg/roitr/configs/val/quad.yaml --data_root=data/sfmreg/Quad/ArtsQuad_dataset/pointclouds/
python experiments/final_experiments/sfmreger_eval_non_finetuned.py omnireg/roitr/configs/val/quad.yaml --data_root=data/sfmreg/Quad/ArtsQuad_dataset/pointclouds
# RoiTr
python experiments/final_experiments/roi_tr_eval_sfmreg.py omnireg/roitr/configs/val/quad.yaml --data_root=data/sfmreg/Quad/ArtsQuad_dataset/pointclouds --finetuned
python experiments/final_experiments/roi_tr_eval_sfmreg.py omnireg/roitr/configs/val/quad.yaml --data_root=data/sfmreg/Quad/ArtsQuad_dataset/pointclouds 
# GeoTransformer
python experiments/final_experiments/geotransformer_eval_sfmreg.py omnireg/roitr/configs/val/quad.yaml --backbone=3dmatch --data_root=data/sfmreg/Quad/ArtsQuad_dataset/pointclouds/
# OverlapPredator
python experiments/final_experiments/predator_eval_sfmreg.py omnireg/OverlapPredator/configs/test/quad.yaml --data_root=data/sfmreg/Quad/ArtsQuad_dataset/pointclouds/
```


However for RoITr or non-finetuned SfMReger things are still manual, but could be made more automatic.

## TODOS

1. Started on evaluating GeoTransformer, however there seems to be some bug in my wrapper [experiments/final_experiments/geotransformer_eval_sfmreg.py](experiments/final_experiments/geotransformer_eval_sfmreg.py), things run very slowly and run out of memory.
2. Evaluate OverlapPredator (see https://github.com/prs-eth/OverlapPredator/blob/770c3063399f08b3836935212ab4c84d355b4704/scripts/demo.py)
3. Sim3 Cambridge, i.e., don't assume shared scale. 
4. Find some more suitable benchmarks. So far LAMAR has failed. Perhaps we can find something else?

## API

We provide a simply API to use the model for inference in [api](api), code example below:

```python
import pycolmap
from omnireg.api import SfMReger

model_A = pycolmap.Reconstruction(...)
model_B = pycolmap.Reconstruction(...)

weights_path = ... # string path, probably in workspace/pretrained
sfmreger = SfMReger(weights_path, mode = "se3")
registration_results = sfmreger.register_reconstructions(model_A, model_B)
print(registration_results)
```

