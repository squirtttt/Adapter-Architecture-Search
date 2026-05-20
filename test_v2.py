import argparse

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import models.repeated_adapter_sam  # registers repeated_adapter_sam
import models.sam2_external  # registers optional SAM2 models
import utils


def eval_loader(loader, model, device, eval_type=None, verbose=False):
    model.eval()
    if eval_type == "f1":
        metric_fn = utils.calc_f1
        metric_names = ("f1", "auc", "none", "none")
    elif eval_type == "fmeasure":
        metric_fn = utils.calc_fmeasure
        metric_names = ("f_mea", "mae", "none", "none")
    elif eval_type == "ber":
        metric_fn = utils.calc_ber
        metric_names = ("shadow", "non_shadow", "ber", "none")
    elif eval_type == "polyp":
        metric_fn = utils.calc_polyp
        metric_names = ("mdice", "miou", "none", "none")
    else:
        metric_fn = utils.calc_cod
        metric_names = ("sm", "em", "wfm", "mae")

    meters = [utils.Averager() for _ in range(4)]
    pbar = tqdm(loader, leave=False, desc="eval")
    with torch.no_grad():
        for batch in pbar:
            for k, v in batch.items():
                batch[k] = v.to(device)
            pred = torch.sigmoid(model.infer(batch["inp"]))
            results = metric_fn(pred, batch["gt"])
            for meter, value in zip(meters, results):
                meter.add(value.item(), batch["inp"].shape[0])
            if verbose:
                pbar.set_description(
                    "eval " + ", ".join(f"{name}={meter.item():.4f}" for name, meter in zip(metric_names, meters))
                )
    return metric_names, [meter.item() for meter in meters]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset_key", default="test_dataset")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_key = args.dataset_key
    if dataset_key not in config:
        dataset_key = "val_dataset"
        print(f"{args.dataset_key} not found in config; falling back to val_dataset")

    spec = config[dataset_key]
    dataset = datasets.make(spec["dataset"])
    dataset = datasets.make(spec["wrapper"], args={"dataset": dataset})
    loader = DataLoader(dataset, batch_size=spec["batch_size"], num_workers=args.num_workers)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = models.make(config["model"]).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint, strict=True)

    metric_names, metric_values = eval_loader(loader, model, device, eval_type=config.get("eval_type"), verbose=args.verbose)
    for name, value in zip(metric_names, metric_values):
        print(f"{name}: {value:.4f}")
