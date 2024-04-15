import copy
import os, argparse, json, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3" # for Pytorch DistributedDataParallel(DDP) training
import torch
from torch import optim
from torch.utils.data.distributed import DistributedSampler # for Pytorch DistbutedDataParallel(DDP) training

from omnireg.roitr.lib.utils import setup_seed
from omnireg.roitr.configs.utils import load_config
from easydict import EasyDict as edict
from omnireg.roitr.dataset.dataloader import get_dataset, get_dataloader
from omnireg.roitr.model.RIGA_v2 import create_model
from omnireg.roitr.lib.loss import OverallLoss, Evaluator, EvaluatorRegistration
from omnireg.roitr.lib.tester import get_trainer
from omnireg.roitr.registration.benchmark_utils import ransac_pose_estimation_correspondences

from tqdm import tqdm
import numpy as np

from omnireg.models.pred_wrapper import PreadtorWrap, calibrate_neighbors
import sys
sys.path.append("omnireg/OverlapPredator")
from models.architectures import KPFCNN
from configs.models import architectures

if __name__ == '__main__':
    # set up validation
    config = load_config("omnireg/roitr/configs/val/sfmreg.yaml")
    config['root'] = "data/sfmreg/megadepth/pointclouds_megadepth"
    config['device'] = torch.device('cuda', 0)
    ##########################################################
    setup_seed(42) # fix the seed

    config_sfmreg = edict(config)
    # config_sfmreg.pretrain = "workspace/y23w47_sfmreg_combined_lo/y23w47_sfmreg_combined_lo/model_best_PIR.pth"
    config_sfmreg.pretrain = 'workspace/y24w7_sfmreg_only/model_best_PIR.pth'
    experiment_conf = edict(
        name = 'omniglue',  # just for interfacing
        descriptor_dim = 256,
        lr = 16e-5,
        refiner_num_heads = 1, 
        refiner_n_layers = 4, 
        refiner_descriptor_dim = 64,
        glue_detach = True,
        glue_refine = True,
        flash = True,
        fine_loss_use_mnn = True,
        iter_size = 4,
        low_overlap = False,
        # sfmreg_mode = "sim3",
    )
    config_sfmreg.update(experiment_conf)

    ##################################################################
    # create model
    model = create_model(config_sfmreg).to(config_sfmreg.device)
    state = torch.load(config_sfmreg.pretrain)
    model.load_state_dict(state['state_dict'])
    
    ##################################################################
    # create model roitr    
    config_roitr = edict(config)
    config_roitr.pretrain = "pretrained/model_3dmatch.pth"
    ######## Experiment settings ###########
    experiment_conf = edict(
        lr = 4e-5,
        use_glue = False,
        glue_refine = False,
    )
    config_roitr.update(experiment_conf)
    roitr_model = create_model(config_roitr).to(config_roitr.device)
    state = torch.load(config_roitr.pretrain)
    roitr_model.load_state_dict({k.replace('module.', ''): v for k, v in state['state_dict'].items()})
    
    #############################################
    # create Predator model
    config = load_config("omnireg/OverlapPredator/configs/val/sfmreg.yaml")
    config['device'] = torch.device('cuda', 0)
    config_predator = edict(config)
    config_predator.pretrain = "pretrained/predator-3dmatch.pth"
    architecture = "indoor"
    config_predator.architecture = architectures[architecture]
    model_predator = PreadtorWrap(KPFCNN(config_predator).to(config_roitr.device), config_predator)
    config_predator.neighborhood_limits = [225, 31, 32, 30]
    
    # get datset
    train_set, val_set, benchmark_set = get_dataset(config_sfmreg)
    val_loader = get_dataloader(val_set,
                                sampler=None,
                                batch_size=1,
                                num_workers=config_sfmreg.num_workers,
                                shuffle=False,
                                drop_last=False)
    evaluator = EvaluatorRegistration(config_sfmreg,True)
    # evaluator = Evaluator(config_sfmreg)
    print('start to evaluate on validation sets...')


    num_iter = int(len(val_loader))
    c_loader_iter = val_loader.__iter__()
    model.eval()
    for c_iter in tqdm(range(num_iter)):
        torch.cuda.synchronize()
        inputs = next(c_loader_iter)#.next()
        #######################################
        # Load inputs to device
        for k, v in inputs.items():
            if type(v) == list:
                inputs[k] = [item.to(config_roitr.device) for item in v]
            elif v is None or isinstance(v, str) :
                inputs[k] = v
            else:
                inputs[k] = v.to(config_roitr.device)

        
        with torch.no_grad():
            rot, trans = inputs['rot'][0], inputs['trans'][0]
            rot_np = rot.cpu().numpy()
            # print('rot mat stas',np.linalg.det(rot_np),np.cbrt(np.linalg.det(rot_np)))
            src_pcd, tgt_pcd = inputs['src_points'].contiguous(), inputs['tgt_points'].contiguous()
            src_normals, tgt_normals = inputs['src_normals'].contiguous(), inputs['tgt_normals'].contiguous()
            src_feats, tgt_feats = inputs['src_feats'].contiguous(), inputs['tgt_feats'].contiguous()
            src_raw_pcd = inputs['raw_src_pcd'].contiguous()
            
            outputs = model.forward(src_pcd, tgt_pcd, src_feats, tgt_feats, src_normals, tgt_normals, rot, trans, src_raw_pcd)
            outputs_roitr = roitr_model.forward(src_pcd, tgt_pcd, src_feats, tgt_feats, src_normals, tgt_normals, rot, trans, src_raw_pcd)
            outputs_predator = model_predator.forward(src_pcd, tgt_pcd, src_feats, tgt_feats, src_normals, tgt_normals, rot, trans, src_raw_pcd)
            evaluator_stats = evaluator(outputs, inputs)
            evaluator_stats_roitr = evaluator(outputs_roitr, inputs)
            evaluator_stats_predator = evaluator(outputs_predator, inputs)
            # print(c_iter,"Ev. stats", evaluator_stats)

            # save stuff
            torch.save(inputs, 'results_supp/model_input'+str(c_iter)+'.pth')
            torch.save(outputs, 'results_supp/model_output'+str(c_iter)+'.pth')
            torch.save(evaluator_stats, 'results_supp/model_stats'+str(c_iter)+'.pth')
            torch.save(outputs_roitr, 'results_supp/roitr_model_output'+str(c_iter)+'.pth')
            torch.save(evaluator_stats_roitr, 'results_supp/roitr_model_stats'+str(c_iter)+'.pth')
            torch.save(outputs_predator, 'results_supp/predator_model_output'+str(c_iter)+'.pth')
            torch.save(evaluator_stats_predator, 'results_supp/predator_model_stats'+str(c_iter)+'.pth')
