"""
generate.py
==========

Generate samples from a trained sampler model and calculate FID score with the generated samples and the training set.

Example:
    torchrun --nproc_per_node=4 generate.py --log_dir results/cifar10/gcdv3_vi/vanilla --batchsize 100 -n 50000
"""
import argparse
import os
import random

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from tqdm import trange

import cmd_utils as cmd
from pytorch_fid.fid_score import (calculate_fid_given_paths,
                                   calculate_fid_given_paths_cache)
from utils import mkdir_p, print0


def print_size(net):
    """
    Print the number of parameters of a network
    """
    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print(
            "{} Parameters: {:.6f}M".format(net.__class__.__name__, params / 1e6),
            flush=True,
        )


def rescale(X, batch=True):
    return (X - (-1)) / (2)


if __name__ == "__main__":
    """
    Example usage: torchrun --nproc_per_node=4 generate.py --log_dir results/cifar10/gcdv3_svi/test
    Input: log directory (ex. results/cifar10/gcdv3_svi/test)
    sampler.pth, config.yaml must be present in logdir
    sampler model is loaded from sampler.pth, sampling parameters and dataset name is inferred from config.yaml
    n_generate samples are saved in logdir/generated and FID score is calculated with the generated samples and the training set
    generated images will be saved in logdir/generated, unless --save_images is set to False
    n_generate should be a multiple of batchsize
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="path to log_dir")
    parser.add_argument(
        "--batchsize", type=int, default=100, help="batch size to generate samples"
    )
    parser.add_argument(
        "-n",
        "--n_generate",
        type=int,
        help="number of samples to generate",
        default=50000,
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--epoch",
        type=str,
        default="best",
        help='which check point to load. can be "best" or "last"',
    )
    parser.add_argument(
        "-save",
        "--save_images",
        type=bool,
        default=True,
        help="Whether to retain the generated images after calculating FID",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Value guidance scale, 0.0 for no guidance",
    )
    parser.add_argument(
        "--stat",
        type=str,
        default=None,
        help="""path to precalculated statistics for FID calculation. Only used for pytorch_fid computation.
            Example: datasets/cifar10_train_fid_stats.pt""",
    )

    # parse command line arguments
    args, unknown = parser.parse_known_args()

    # set random seed
    # setting seeds
    torch.backends.cudnn.deterministic = True
    local_rank = int(os.environ["LOCAL_RANK"])
    device = "cuda:{}".format(local_rank)
    seed = args.seed
    torch.cuda.set_device(device)
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed + local_rank)
    random.seed(seed + local_rank)
    os.environ["PYTHONHASHSEED"] = str(seed + local_rank)

    assert (
        args.n_generate % args.batchsize == 0
    ), "n_generate must be a multiple of batchsize"

    config_path = os.path.join(args.log_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise ValueError("Config not found at {}".format(config_path))
    run_config = OmegaConf.load(config_path)

    data_path = os.path.join(
        "datasets", f"{run_config.data.name}_train_png"
    )  # ex) datasets/cifar10_train_png
    if not os.path.exists(data_path):
        raise ValueError("Dataset not found at {}".format(data_path))

    if args.guidance_scale is not None:
        output_path = os.path.join(args.log_dir, f"generated_{args.guidance_scale}")
    else:
        output_path = os.path.join(args.log_dir, "generated")
    mkdir_p(output_path)

    sampler_path = os.path.join(args.log_dir, f"sampler_{args.epoch}.pth")
    old_sampler_path = os.path.join(args.log_dir, "sampler.pth")
    if not os.path.exists(sampler_path) and os.path.exists(
        old_sampler_path
    ):  # for backward compatibility
        sampler_path = old_sampler_path
    if not os.path.exists(sampler_path):
        raise ValueError("Sampler not found at {}".format(sampler_path))

    # load sampler
    net = instantiate(run_config.sampler_net)
    print_size(net)
    sample_shape = run_config.sampler.sample_shape
    sampler = instantiate(run_config.sampler, net=net).to(device)

    checkpoint = torch.load(sampler_path, map_location=device)
    sampler.net.load_state_dict(checkpoint["state_dict"])
    print0("Loaded sampler from {}".format(sampler_path))
    epoch_info = checkpoint["epoch"]
    fid_info = checkpoint["fid"]
    print0(
        "Model was trained for {} epochs, FID score evaluated with 10000 samples are: {}".format(
            epoch_info, fid_info
        )
    )
    sampler.eval()

    if args.guidance_scale is not None:
        print0("Loading value function for guidance")
        value_path = os.path.join(args.log_dir, "value_best.pth")
        if not os.path.exists(value_path):
            raise ValueError("Value ftn not found at {}".format(value_path))
        v = instantiate(run_config.value).to(device)
        checkpoint = torch.load(value_path, map_location=device)
        v.load_state_dict(checkpoint["state_dict"])
        v.eval()
    else:
        v = None

    ngpus = torch.cuda.device_count()
    if ngpus >= 1:
        print0(f"Using distributed training on {ngpus} gpus.")
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        sampler.net = DDP(
            sampler.net, device_ids=[local_rank], output_device=local_rank
        )
        if v is not None:
            v = DDP(v, device_ids=[local_rank], output_device=local_rank)

    if args.guidance_scale is not None:
        trainer = instantiate(run_config.trainer, batchsize=args.batchsize)
        trainer.set_models(
            f=None,
            v=v,
            sampler=sampler,
            optimizer=None,
            optimizer_fstar=None,
            optimizer_v=None,
        )

    n_sample_to_generate = args.n_generate / args.batchsize / ngpus
    i_img = 0
    for i in trange(int(n_sample_to_generate), ncols=80):
        with torch.no_grad():
            if args.guidance_scale is not None:
                d_sample = trainer.sample_guidance(
                    n_sample=args.batchsize,
                    device=device,
                    guidance_scale=args.guidance_scale,
                )
            else:
                d_sample = sampler.sample(args.batchsize, device=device)
            Xi = d_sample["sample"].detach().cpu()
        sample = rescale(Xi).clamp(0, 1).detach().cpu()
        for s in sample:
            save_image(s, os.path.join(output_path, f"{local_rank}_{i_img}.png"))
            i_img += 1

    torch.distributed.barrier()  # make sure all files are generated
    print0(f"Generated {args.n_generate} samples at {output_path}")

    if local_rank <= 0:
        print("Calculating FID score")
        paths = [output_path, data_path]
        kwargs = {"batch_size": args.batchsize, "device": device, "dims": 2048}
        if args.stat is None:
            fid_score = calculate_fid_given_paths(paths, **kwargs)
        else:
            print(f"Loading precomputed statistics from {args.stat}")
            d_fid_stats = torch.load(args.stat)
            m2, s2 = d_fid_stats["m2"], d_fid_stats["s2"]
            paths = [output_path, data_path]
            fid_score, _, _ = calculate_fid_given_paths_cache(
                paths=paths, m2=m2, s2=s2, **kwargs
            )
        print(f"FID score: {fid_score}")

        if not args.save_images:
            import shutil

            shutil.rmtree(output_path)
