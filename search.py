import argparse
import os
import random
import itertools

import yaml
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import models
import utils
from statistics import mean


import pdb

from ZeroShotProxy import compute_zico_score # compute_naswot, compute_zen_score

torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# config로 불러와야 하는 내용
# scoring
# - zeroshot proxy
# - search space
# sam-adapter
# - dataset
# - sam pretrained model

def generate_architecture_candidate_at_once(config):

    search_space = config['search']['search_space']

    all_values = {key: value for key, value in search_space.items()}

    all_layernums= []
    for prompt_num in all_values['prompt_num']:
        possible_layernums = list(itertools.combinations(range(1, 13), prompt_num))
        all_layernums.append(possible_layernums)
    
    arch_configs = []
    for combination in itertools.product(*all_values.values()):
        scale_factor, prompt_num, prompt_activation = combination
        possible_layernums = all_layernums[all_values['prompt_num'].index(prompt_num)]
        for layernum in possible_layernums:
            arch_config = {
                'scale_factor': scale_factor,
                'prompt_num': prompt_num,
                'prompt_layernum': layernum,
                'prompt_activation': prompt_activation
            }
            arch_configs.append(arch_config)
    

    return arch_configs

def prepare_training(arch_config, config):
    # config merge
    config['model']['args']['encoder_mode'].update(arch_config)

    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(∂
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler

def compute_nas_score(arch_config, model=None, search_proxy='zico', train_loader=None, lossfunc=None):
    mlp_arch = arch_config
    sam_model = model
    search_proxy = search_proxy


    if search_proxy == 'zico':
        nas_score =  compute_zico_score.getzico(sam_model, train_loader, lossfunction)
    elif search_proxy == 'zen':
        nas_score = 0 #compute_zen_score.compute_nas_score(mlp_arch, )
    elif search_proxy == 'naswot':
        nas_score = 0 #compute_naswot_score.compute()

    torch.cuda.empty_cache()

    return nas_score

def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    search_epoch = config['search']['epoch_search_max']
    search_population = config['search']['population_size']
    search_patient = config['search']['patient']
    max_patient = config['search']['patient']
    patient = 0
    
    # pdb.set_trace()
    candidate_configs = generate_architecture_candidate_at_once(config)
    
    structure_list = []
    best_score = 0
    best_arch = []

    for i in range(len(candidate_configs)):
        # random architecture search
        arch_config = candidate_configs[i]
        
        # SAM architecture construct 
        model, optimizer, epoch_start, lr_scheduler = prepare_training(arch_config, config)

        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids= [args.local_rank],
            output_device = args.local_rank,
            find_unused_parameters = True,
            broadcast_buffers = False,
        )
        model = model.module

        sam_checkpoint = torch.load(config['sam_checkpoint'])
        model.load_state_dict(sam_checkpoint, strict=False)

        for name, para in model.named_parameters():
            if "image_encoder" in name and "prompt_generator" not in name:
                para.requires_grad_(False)
        if local_rank == 0:
            model_total_params = sum(p.numel() for p in model.parameters())
            model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))
        
        # compute nas score
        nas_score = compute_nas_score(arch_config, model, search_proxy)

        # score comparison
        if nas_score > best_score:
            best_score = score
            best_arch = arch_config
            # patient = 0
            structure_list.append({'arch':best_arch, 'score': best_score})
            structure_list = sorted(structure_list, key=lambda x: x['score'], reverse=True)
        else:
            # patient += 1
        
        # patient comparison
        # if patient >= max_patient:
        #     print("training stopped due to no improvement in NAS score")
        #     break
        
    return structure_list
            



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default = './configs/search_demo.yaml')
    parser.add_argument('--name', default = None)
    parser.add_argument('--tag', default = None)
    parser.add_argument('--local_rank', type=int, default=-1, help="")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded')

    save_name = args.name
    if save_name is None: 
        save_name = '_'+ args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, args=args)