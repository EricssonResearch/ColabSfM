import os, gc, json
import torch
from tensorboardX import SummaryWriter
from .utils import AverageMeter, Logger, to_o3d_pcd
from tqdm import tqdm
import open3d as o3d


class Trainer(object):
    '''
    Class Trainer
    '''

    def __init__(self, config):
        self.config = config
        #############################
        # hyper-parameters
        #############################
        self.verbose = config.verbose
        self.verbose_freq = config.verbose_freq
        self.start_epoch = 1
        self.max_epoch = config.max_epoch
        self.training_max_iter = config.training_max_iter
        self.val_max_iter = config.val_max_iter
        self.device = config.device

        self.output_metric_path = config.output_metric_path
        self.saved_stats_dict = None
        if self.output_metric_path is not None:
            if os.path.exists(config.output_metric_path):
                with open(self.output_metric_path, "r+") as f:
                    self.saved_stats_dict = json.load(f)
        self.debug = config.debug

        self.best_loss = self.best_c_loss = self.best_f_loss = self.best_o_loss = 1e5
        self.best_PIR = self.best_IR = -1.

        self.save_dir = config.save_dir
        self.snapshot_dir = config.snapshot_dir

        self.model = config.model.to(self.device)
        self.local_rank = config.local_rank
        self.optimizer = config.optimizer
        self.scheduler = config.scheduler
        self.scheduler_interval = config.scheduler_interval
        self.snapshot_interval = config.snapshot_interval
        self.iter_size = config.iter_size

        if self.local_rank <= 0:
            self.writer = SummaryWriter(logdir=config.tboard_dir)
            self.logger = Logger(self.snapshot_dir)
            self.logger.write(f'#parameters {sum([x.nelement() for x in self.model.parameters()]) / 1000000.} M\n')
            with open(f'{config.snapshot_dir}/model.log', 'w') as f:
                f.write(str(self.model))

            f.close()
        else:
            self.writer = None
            self.logger = None

        if config.pretrain != '':
            self._load_pretrain(config.pretrain)

        self.loader = dict()

        self.loader['train'] = config.train_loader
        self.loader['val'] = config.val_loader
        self.loader['test'] = config.test_loader

        self.loss_func = config.loss_func
        self.evaluator = config.evaluator

    def _snapshot(self, epoch, name=None):
        '''
        Save a trained model
        :param epoch:  epoch of current model
        :param name: path to save current model
        :return: None
        '''
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_c_loss': self.best_c_loss,
            'best_f_loss': self.best_f_loss,
            'best_o_loss': self.best_o_loss,
            'best_PIR': self.best_PIR,
            'best_IR': self.best_IR,
        }

        if name is None:
            filename = os.path.join(self.save_dir, f'model_{epoch}.pth')
        else:
            filename = os.path.join(self.save_dir, f'model_{name}.pth')

        print(f'Save model to {filename}')
        self.logger.write(f'Save model to {filename}\n')
        torch.save(state, filename)

    def _load_pretrain(self, resume):
        '''
        Load a pretrained model
        :param resume: the path to the pretrained model
        :return: None
        '''
        if os.path.isfile(resume):
            print(f'=> loading checkpoint {resume}')
            state = torch.load(resume)
            self.start_epoch = state['epoch']
            if self.local_rank < 0:
                #print(state['state_dict'].keys())
                model_key = 'state_dict' if 'state_dict' in state else 'model' # hacky way to load both geotransformer and roitr
                model_key_key = list(state[model_key].keys())
                if model_key_key[0].split('.')[0] == 'model':
                    pretrained_dict = {k.replace('model.', ''): v for k, v in state[model_key].items()}
                else:
                    pretrained_dict = {k.replace('module.', ''): v for k, v in state[model_key].items()}
                if self.config.non_strict_loading:
                    model_dict = self.model.state_dict()
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                        (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
                    missing_keys, unexpected_keys = self.model.load_state_dict(pretrained_dict, strict = False)
                    self.logger.write(f"{missing_keys=}" + "\n")
                    self.logger.write(f"{unexpected_keys=}" + "\n")
                else:
                    self.model.load_state_dict(pretrained_dict, strict = True)
                #self.optimizer.load_state_dict(state['optimizer'])
                #self.scheduler.load_state_dict(state['scheduler'])
            else:
                self.model.load_state_dict(state['state_dict'])
                self.optimizer.load_state_dict(state['optimizer'])
                self.scheduler.load_state_dict(state['scheduler'])

            if self.local_rank <= 0 and self.config.dataset not in ["colabsfm", "quad"]:
                self.best_loss = state['best_loss']
                self.best_c_loss = state['best_c_loss']
                self.best_f_loss = state['best_f_loss']
                self.best_o_loss = state['best_o_loss']
                self.best_PIR = state['best_PIR']
                self.best_IR = state['best_IR']
                self.logger.write(f'Successfully load pretrained model from {resume}!\n')
                self.logger.write(f'Current best loss {self.best_loss}\n')
                self.logger.write(f'Current best c_loss {self.best_c_loss}\n')
                self.logger.write(f'Current best f_loss {self.best_f_loss}\n')

                self.logger.write(f'Current best PIR {self.best_PIR}\n')
                self.logger.write(f'Current best IR {self.best_IR}\n')
            else:
                self.logger.write(f"Will not compare previous best loss when finetuning on colabsfm")

        else:
            raise ValueError(f'=> no checkpoint found at {resume}')

    def _get_lr(self, group=0):
        '''
        Get current learning rate
        :param group:
        :return:
        '''
        return self.optimizer.param_groups[group]['lr']

    def stats_dict(self):
        '''
        Create the dictionary consisting of all the metrics
        :return: as described
        '''
        stats = dict()
        stats['loss'] = 0.
        stats['c_loss'] = 0.
        stats['f_loss'] = 0.
        stats['o_loss'] = 0.
        stats['PIR'] = 0.
        stats['IR'] = 0.
        stats['FMR'] = 0.
        stats['num_matches'] = 0.
        stats['trans_error'] = 0.
        stats['rot_error'] = 0.
        stats['RR'] = 0.
        # stats['scale_error'] = 0.
        # stats['inlier_rmse'] = 0.
        stats['RR_rmse'] = 0.
        # stats['RR_in_rmse'] = 0.
        stats['rmse'] = 0.
        
        '''
        to be added
        '''
        return stats

    def stats_meter(self):
        '''
        For each metric in stats dict, create an AverageMeter class for updating
        :return: as described
        '''
        meters = dict()
        stats = self.stats_dict()
        for key, _ in stats.items():
            meters[key] = AverageMeter()
        return meters

    def inference_one_batch(self, inputs, phase, idx=None):
        '''
        Inference for a single batch data
        :param inputs: the dictionary consisting of all the input data
        :param phase: train, validation or test
        :return: dictionary consisting of losses and metrics
        '''
        assert phase in ['train', 'val']
        #########################################
        # training
        #########################################
        if phase == 'train':
            self.model.train()

            ##################
            # forward pass
            ##################
            rot, trans = inputs['rot'][0], inputs['trans'][0]
            src_pcd, tgt_pcd = inputs['src_points'].contiguous(), inputs['tgt_points'].contiguous()
            src_normals, tgt_normals = inputs['src_normals'].contiguous(), inputs['tgt_normals'].contiguous()
            src_feats, tgt_feats = inputs['src_feats'].contiguous(), inputs['tgt_feats'].contiguous()
            src_raw_pcd = inputs['raw_src_pcd'].contiguous()
            try:
                outputs = self.model.forward(src_pcd, tgt_pcd, src_feats, tgt_feats, src_normals, tgt_normals, rot, trans, src_raw_pcd)
            except Exception as e:
                print("Failed to run forward during training with exception ->")
                print(e)
                print("Culprit pair:")
                print(inputs['src_paths'])
                print(inputs['tgt_paths'])
                raise e
            if not outputs:
                return {}
            
            stats = self.loss_func(outputs, inputs)
            evaluator_stats = self.evaluator(outputs, inputs)

            stats.update(evaluator_stats)

            loss = stats['loss']
            loss.backward()

        else:
            self.model.eval()
            with torch.no_grad():
                rot, trans = inputs['rot'][0], inputs['trans'][0]
                src_pcd, tgt_pcd = inputs['src_points'].contiguous(), inputs['tgt_points'].contiguous()
                src_normals, tgt_normals = inputs['src_normals'].contiguous(), inputs['tgt_normals'].contiguous()
                src_feats, tgt_feats = inputs['src_feats'].contiguous(), inputs['tgt_feats'].contiguous()
                src_raw_pcd = inputs['raw_src_pcd'].contiguous()

                outputs = self.model.forward(src_pcd, tgt_pcd, src_feats, tgt_feats, src_normals, tgt_normals, rot, trans, src_raw_pcd)
                if not outputs:
                    return {}
                gc.collect()
                stats = self.loss_func(outputs, inputs)
                evaluator_stats = self.evaluator(outputs, inputs)
                if self.debug: print("Ev. stats", evaluator_stats)
                
                # if evaluator_stats["IR"] > .48 and evaluator_stats["num_matches"] > 1000:
                #     print(evaluator_stats)
                #     torch.save(outputs, 'model_output.pth')
                #     torch.save(inputs, 'model_input.pth')
                #     print("saved output")
                #     exit()
                stats.update(evaluator_stats)


        ########################################
        # re-organize dictionary stats
        ########################################
        stats['loss'] = float(stats['loss'].detach())
        stats['c_loss'] = float(stats['c_loss'].detach())
        stats['f_loss'] = float(stats['f_loss'].detach())
        stats['o_loss'] = float(stats['o_loss'].detach())
        return stats

    def inference_one_epoch(self, epoch, phase):
        '''
        Inference for an epoch
        :param epoch: current epoch
        :param phase: current phase of training
        :return:
        '''
        gc.collect()
        assert phase in ['train', 'val']

        #init stats meter
        stats_meter = self.stats_meter()
        if phase == "val":
            if self.saved_stats_dict is not None:
                print("Found a metric dict saved in %s." % self.output_metric_path)
                print("\tContinuing from step %d" % len(self.saved_stats_dict["IR"]))
            elif self.output_metric_path is not None:
                print("Creating metric dic in %s" % self.output_metric_path)
                self.saved_stats_dict = dict()
                stats = self.stats_dict()
                for key, _ in stats.items():
                    self.saved_stats_dict[key] = []

        num_iter = int(len(self.loader[phase]))
        c_loader_iter = self.loader[phase].__iter__()

        self.optimizer.zero_grad()
        idx = 0
        for c_iter in tqdm(range(num_iter)):
            inputs = next(c_loader_iter)#.next()
            # if c_iter in [1627,3703,4035,4423,5038,5202,5964,6081,6360,7113,9253,9914,10810,11708]: continue
            # if c_iter in [3687,5369,16564]: continue
            # if c_iter in [667, 717, 721, 797, 810, 991, 994, 995, 1065, 1067, 1356, 1368]: continue
            if self.saved_stats_dict is not None and c_iter < len(self.saved_stats_dict["IR"]): continue
            for k, v in inputs.items():
                if type(v) == list:
                    inputs[k] = [item.to(self.device) for item in v]
                elif v is None or isinstance(v, str) :
                    inputs[k] = v
                else:
                    inputs[k] = v.to(self.device)

            ######################
            # forward pass
            ######################
            stats = self.inference_one_batch(inputs, phase, idx=idx)
            if not stats:
                continue
            idx += 1

            ######################
            # run optimization
            ######################
            if (c_iter + 1) % self.iter_size == 0 and phase == 'train':
                self.optimizer.step()
                self.optimizer.zero_grad()

            ########################
            # update to stats_meter
            ########################
            for key, value in stats.items():
                stats_meter[key].update(value)
                if self.saved_stats_dict is not None:
                    if isinstance(value, torch.Tensor): self.saved_stats_dict[key].append(value.item())
                    else: self.saved_stats_dict[key].append(value)
            if self.output_metric_path is not None:
                with open(self.output_metric_path, "w") as f:
                    json.dump(self.saved_stats_dict, f, indent=4)

            torch.cuda.empty_cache()

            if self.local_rank <= 0 and self.verbose and (c_iter + 1) % self.verbose_freq == 0:
                cur_iter = num_iter * (epoch - 1) + c_iter
                for key, value in stats_meter.items():
                    self.writer.add_scalar(f'{phase}/{key}', value.avg, cur_iter)

                message = f'{phase} Epoch: {epoch} [{c_iter + 1:4d}/{num_iter}] '
                print(stats_meter)
                for key, value in stats_meter.items():
                    print(key, value)
                    message += f'{key}:{value.avg:.2f}\t'

                self.logger.write(message + '\n')
        if self.local_rank <= 0:
            message = f'{phase} Epoch: {epoch} '
            for key, value in stats_meter.items():
                message += f'{key}: {value.avg:.4f}\t'

            self.logger.write(message + '\n')
        return stats_meter

    def train(self):
        '''
        Train
        :return:
        '''
        print('start training...')
        for epoch in range(self.start_epoch, self.max_epoch):
            if self.local_rank > -1:
                self.loader['train'].sampler.set_epoch(epoch)

            self.inference_one_epoch(epoch, 'train')
            self.scheduler.step()
            stats_meter = self.inference_one_epoch(epoch, 'val')
            if self.local_rank <= 0:
                if stats_meter['loss'].avg < self.best_loss:
                    self.best_loss = stats_meter['loss'].avg
                    self._snapshot(epoch, 'best_loss')

                if stats_meter['c_loss'].avg < self.best_c_loss:
                    self.best_c_loss = stats_meter['c_loss'].avg
                    self._snapshot(epoch, 'best_c_loss')

                if stats_meter['f_loss'].avg < self.best_f_loss:
                    self.best_f_loss = stats_meter['f_loss'].avg
                    self._snapshot(epoch, 'best_f_loss')
                if stats_meter['o_loss'].avg < self.best_o_loss:
                    self.best_o_loss = stats_meter['o_loss'].avg
                    self._snapshot(epoch, 'best_o_loss')

                if stats_meter['PIR'].avg > self.best_PIR:
                    self.best_PIR = stats_meter['PIR'].avg
                    self._snapshot(epoch, 'best_PIR')

                if stats_meter['IR'].avg > self.best_IR:
                    self.best_IR = stats_meter['IR'].avg
                    self._snapshot(epoch, 'best_IR')


        print('training finish!')

    def eval(self):
        '''
        Evaluation
        :return:
        '''
        print('start to evaluate on validation sets...')
        stats_meter = self.inference_one_epoch(0, 'val')

        for key, value in stats_meter.items():
            print(f'{key}: {value.avg}')
