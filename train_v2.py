import argparse
import os

import torch
import torch.distributed as dist
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from statistics import mean

import datasets
import models
import models.repeated_adapter_sam  # registers repeated_adapter_sam
import utils


torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


def make_data_loader(spec, tag=""):
    if spec is None:
        return None
    dataset = datasets.make(spec["dataset"])
    dataset = datasets.make(spec["wrapper"], args={"dataset": dataset})
    if local_rank == 0:
        log(f"{tag} dataset: size={len(dataset)}")
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    return DataLoader(dataset, batch_size=spec["batch_size"], shuffle=False, num_workers=8, pin_memory=True, sampler=sampler)


def make_data_loaders():
    return make_data_loader(config.get("train_dataset"), "train"), make_data_loader(config.get("val_dataset"), "val")


def freeze_sam_backbone_enable_adapter_training(model):
    for _, param in model.named_parameters():
        param.requires_grad_(False)
    for name, param in model.named_parameters():
        if "extended_adapters" in name:
            param.requires_grad_(True)


def eval_psnr(loader, model, eval_type=None):
    model.eval()
    if eval_type == "cod":
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = "sm", "em", "wfm", "mae"
    elif eval_type == "polyp":
        metric_fn = utils.calc_polyp
        metric1, metric2, metric3, metric4 = "mdice", "miou", "none", "none"
    else:
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = "f_mea", "mae", "none", "none"

    pbar = tqdm(total=len(loader), leave=False, desc="val") if local_rank == 0 else None
    pred_list, gt_list = [], []
    with torch.inference_mode():
        for batch in loader:
            for k, v in batch.items():
                batch[k] = v.cuda()
            pred = torch.sigmoid(model.infer(batch["inp"]))
            batch_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
            batch_gt = [torch.zeros_like(batch["gt"]) for _ in range(dist.get_world_size())]
            dist.all_gather(batch_pred, pred)
            dist.all_gather(batch_gt, batch["gt"])
            pred_list.extend(item.cpu() for item in batch_pred)
            gt_list.extend(item.cpu() for item in batch_gt)
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()
    result1, result2, result3, result4 = metric_fn(torch.cat(pred_list, 1), torch.cat(gt_list, 1))
    return result1, result2, result3, result4, metric1, metric2, metric3, metric4


def prepare_training():
    model = models.make(config["model"]).cuda()
    optimizer = utils.make_optimizer(model.parameters(), config["optimizer"])
    epoch_start = config.get("resume", 0) + 1 if config.get("resume") is not None else 1
    lr_scheduler = CosineAnnealingLR(optimizer, config.get("epoch_max"), eta_min=config.get("lr_min"))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model):
    model.train()
    pbar = tqdm(total=len(train_loader), leave=False, desc="train") if local_rank == 0 else None
    loss_list = []
    for batch in train_loader:
        for k, v in batch.items():
            batch[k] = v.to(device)
        model.set_input(batch["inp"], batch["gt"])
        model.optimize_parameters()
        batch_loss = [torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, model.loss_G)
        loss_list.extend(batch_loss)
        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()
    return mean([i.item() for i in loss_list])


def save(model, save_path, name):
    torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


def main(config_, save_path, args):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, "config.yaml"), "w") as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer
    model = torch.nn.parallel.DistributedDataParallel(
        model.cuda(),
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False,
    ).module

    sam_checkpoint = torch.load(config["sam_checkpoint"], map_location="cpu")
    model.load_state_dict(sam_checkpoint, strict=False)
    freeze_sam_backbone_enable_adapter_training(model)

    if local_rank == 0:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("model_grad_params:" + str(trainable), "\nmodel_total_params:" + str(total))

    max_val_v = -1e18
    for epoch in range(epoch_start, config["epoch_max"] + 1):
        train_loader.sampler.set_epoch(epoch)
        train_loss = train(train_loader, model)
        lr_scheduler.step()

        if local_rank == 0:
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalars("loss", {"train G": train_loss}, epoch)
            save(model, save_path, "last")

        if config.get("epoch_val") is not None and epoch % config["epoch_val"] == 0:
            result1, result2, result3, result4, metric1, metric2, metric3, metric4 = eval_psnr(
                val_loader, model, eval_type=config.get("eval_type")
            )
            if local_rank == 0:
                log(f"epoch {epoch}/{config['epoch_max']}, train G: loss={train_loss:.4f}, val: {metric1}={result1:.4f}")
                if result1 > max_val_v:
                    max_val_v = result1
                    save(model, save_path, "best")
                writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./save/_search_demo_v2/best_arch_hierarchical_v2.yaml")
    parser.add_argument("--name", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    save_name = args.name or "_" + args.config.split("/")[-1][:-len(".yaml")]
    if args.tag is not None:
        save_name += "_" + args.tag
    main(config, os.path.join("./save", save_name), args)
