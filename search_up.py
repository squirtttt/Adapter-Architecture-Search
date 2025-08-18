import argparse
import os
import random
import itertools

import yaml
from tqdm import tqdm
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

import datasets
import models
import utils
from statistics import mean

import pdb


torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

## dataloader build

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True, sampler=sampler)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


## zico scoring method

def compute_nas_score(arch_config, model=None, search_proxy='zico', train_loader=None, lossfunc=None):
    mlp_arch = arch_config
    sam_model = model
    search_proxy = search_proxy


    if search_proxy == 'zico':
        nas_score = getzico(sam_model, train_loader, arch_config)
    elif search_proxy == 'zen':
        nas_score = 0 #compute_zen_score.compute_nas_score(mlp_arch, )
    elif search_proxy == 'naswot':
        nas_score = 0 #compute_naswot_score.compute()

    torch.cuda.empty_cache()

    return nas_score

def getgrad(model:torch.nn.Module, grad_dict:dict, step_iter=0):
    if step_iter==0:
        for name,mod in model.named_modules():
            #if isinstance(mod, nn.Linear): # isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
            if 'prompt_generator' in name:
                if 'lightweight_mlp_' in name and name.endswith('.0') and isinstance(mod, torch.nn.Linear):
                    # pdb.set_trace()
                    # print(mod.weight.grad.data.size())
                    # print(mod.weight.data.size())
                    name = [s for s in name.split('.') if s.startswith('lightweight_mlp_')]
                    short_name = name[0]
                    grad_dict[short_name]=[mod.weight.grad.data.cpu().reshape(-1).numpy()]
    else:
        for name,mod in model.named_modules():
            if "prompt_generator" in name:
            #if isinstance(mod, nn.Linear): # isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                if 'lightweight_mlp_' in name and name.endswith('.0') and isinstance(mod, torch.nn.Linear):
                    name = [s for s in name.split('.') if s.startswith('lightweight_mlp_')]
                    short_name = name[0]
                    grad_dict[short_name].append(mod.weight.grad.data.cpu().reshape( -1).numpy())
    return grad_dict

def caculate_zico(grad_dict, arch_config):
    allgrad_array=None
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname]= np.array(grad_dict[modname])
    nsr_mean_sum = 0
    nsr_mean_sum_abs = 0
    nsr_mean_avg = 0
    nsr_mean_avg_abs = 0
    score_dict = {}
    scale_factor = arch_config['model']['args']['encoder_mode']['scale_factor']
    dim = arch_config['model']['args']['encoder_mode']['embed_dim']//arch_config['model']['args']['encoder_mode']['scale_factor']
    for j, modname in enumerate(grad_dict.keys()):
        # pdb.set_trace()
        nsr_std = np.std(grad_dict[modname], axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        tmpsum = np.sum(nsr_mean_abs[nonzero_idx]/nsr_std[nonzero_idx])
        if tmpsum==0:
            pass
        else:
            # tmp_sum = (np.log(tmpsum))/dim
            d = grad_dict[modname].shape[1]
            tmp_sum = np.log(1+tmpsum/d) #*scale_factor
            nsr_mean_sum_abs += tmp_sum
            nsr_mean_avg_abs += np.log(np.mean(nsr_mean_abs[nonzero_idx]/nsr_std[nonzero_idx]))
            score_dict[modname] = tmp_sum
    score = nsr_mean_sum_abs/len(grad_dict)
    return score, score_dict # nsr_mean_sum_abs


def getzico(model, train_loader, arch_config):
    grad_dict= {}
    model.train()
    loss_list = []
    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    for i, batch in enumerate(train_loader):
        for k, v in batch.items():
            batch[k] = v.to(device)
        model.zero_grad()
        inp = batch['inp']
        gt = batch['gt']
        model.set_input(inp, gt)
        model.search_backward()
        batch_loss = [torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, model.loss_G)
        loss_list.extend(batch_loss)
        if pbar is not None:
            pbar.update(1)
        
        grad_dict= getgrad(model, grad_dict,i)
    
    #pdb.set_trace()    
    res, score_dict = caculate_zico(grad_dict, arch_config)
    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    return mean(loss), res, score_dict

## architecture candidate generation

def generate_architecture_candidate(config): # architecture candidate

    search_space = config['search']['search_space']

    scale_factors = search_space.get('scale_factor', [])
    prompt_activations = search_space.get('prompt_activation', [])

    if not scale_factors or not prompt_activations:
        raise ValueError("Empty scale factor or prompt activation function")
    
    alpha_init = config['search'].get('alpha', [0.0] * 12)

    arch_configs = []
    for scale_factor, prompt_activation in itertools.product(scale_factors, prompt_activations):
        arch_configs.append({
            'scale_factor': scale_factor,
            'prompt_activation': prompt_activation,
            'alpha': torch.tensor(alpha_init, dtype=torch.float32)
        })

    return arch_configs

def prepare_training(arch_config, config): # architecture construct
    # config merge
    config['model']['args']['encoder_mode'].update(arch_config)

    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    if local_rank == 0:
        # log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
        log(f'arch configuration: {arch_config}')
    return model, optimizer, epoch_start, lr_scheduler

## search - main
def main(config_, save_path, args):

    # init data, config
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }


    # search hyper-parameter
    patience = config['search']['patient'] # early stopping patience
    num_iter = config['search']['iteration'] # search iteration for each arch candidate
    K = config['search']['sample_size'] # sample size of perturbation
    epsilon = config['search']['epsilon'] # perturbation size
    tau = config['search']['tau'] # softmax temperature
    min_delta = config['search']['delta'] # minimum score difference
    threshold = config['search']['threshold']

    arch_candidate = generate_architecture_candidate(config) # only scale_factor, activation function


    best_score = float('inf')
    best_adapter_idx = None
    best_alpha = None
    best_arch = {
        'scale_factor': None,
        'prompt_activation': None,
        'alpha': None
    }

    for idx, adapter in enumerate(arch_candidate):
        alpha = nn.Parameter(torch.abs(torch.randn(12))) # parameters for each layer
        best_arch_score = float('inf')
        no_improve_counter = 0

        for step in range(num_iter): # sampling for each arch config
            delta_lst = [torch.abs(torch.randn_like(alpha)*epsilon) for _ in range(K)]
            nas_scores = []

            for delta in delta_lst:
                alpha_perturbed = alpha+delta
                # alpha -> configuration update
                adapter['alpha'] = alpha_perturbed.detach().clone()
            
                model, optimizer, epoch_start, lr_scheduler = prepare_training(adapter, config)
                model.optimizer = optimizer

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
                        # print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

                # compute nas score
                loss, nas_score, score_dict = compute_nas_score(arch_config = config, model=model, search_proxy='zico',train_loader=train_loader)
                nas_scores.append(nas_score)
                if local_rank == 0:
                    # log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
                    log(f"current zico: {nas_score}")
            
            z = torch.tensor(nas_scores, device=alpha.device, dtype=torch.float32)
            weights = torch.softmax((z-z.max())/tau, dim=0)
            direction = sum(w*d for w, d in zip(weights, delta_lst))
            alpha.data.add_(direction)

            current_best = float(torch.max(z))
            if (current_best - best_score) > min_delta:
                best_score = current_best
                no_improve_counter = 0
            else:
                no_improve_counter += 1
            if local_rank == 0:
                # log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
                log(f"[Adapter {idx+1} | Step {step}] ZICO: {current_best:.4f} | Best: {best_score:.4f} | Patience: {no_improve_counter}")
    
            if no_improve_counter >= patience:
                print(f"Adapter {idx+1}: Early stopping triggered at step {step}.")
                break

        # comparison with each best arch candidate
        # alpha multiplied with hard-label
        with torch.no_grad():
            probs = torch.sigmoid(alpha) # sigmoid는 값의 범위에 따라 적용 여부가 달라짐
            alpha_hard = (probs > threshold).float()
        adapter['alpha'] = alpha_hard.detach().clone()
        
        final_model = prepare_training(adapter, config)
        model.optimizer = optimizer

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
                        

        # compute nas score
        loss, final_score, score_dict = compute_nas_score(arch_config = config, model=model, search_proxy='zico',train_loader=train_loader)


        if final_score < best_score:
            best_score = final_score
            best_adapter_idx = idx
            best_alpha = alpha_hard.detach().clone()
            best_arch.update({
                'scale_factor': adapter['scale_factor'],
                'prompt_activation': adapter['prompt_activation'],
                'alpha': best_alpha
            })

    # end search        
    print(f"Best Adapter: Adapter{best_adapter_idx + 1}")
    print(f"Inclusion mask (sigmoid): {torch.sigmoid(best_alpha)}")

    if local_rank == 0:
        config['model']['args']['encoder_mode'].update(best_arch)
        save_path = save_name+"/best_arch.yaml"
        with open(save_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        print(f'best architecture config saved at: {save_path}')

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
    