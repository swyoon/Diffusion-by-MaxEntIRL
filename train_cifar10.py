"""
Run DxMI training for CIFAR-10.
Example command:
$ CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_cifar10.py \
    --config configs/cifar10/T10.yaml --dataset configs/cifar10/cifar10.yaml


$ CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_cifar10.py \
    --config configs/cifar10/T4_ddgan.yaml --dataset configs/cifar10/cifar10.yaml
"""
import argparse
import os

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

import random

import wandb

wandb.require("core")

from hydra.utils import instantiate
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from tqdm import trange

import cmd_utils as cmd
import loader
from models.DxMI.trainer import append_buffer, reset_buffer
from models.logger import BaseLogger
from utils import (fix_legacy_dict, mkdir_p, print0,
                   weight_norm)


def rescale(X, batch=True):
    return (X - (-1)) / (2)


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


def save_model(trainer, logdir, postfix, d_other_info=None):
    """
    Save the model and other stuff
    """
    if d_other_info is None:
        d_other_info = {}
    d_sampler = {"state_dict": trainer.sampler.net.module.state_dict()}
    d_sampler.update(d_other_info)
    torch.save(
        d_sampler, os.path.join(logdir, f"sampler_{postfix}.pth")
    )  # Save the model
    if trainer.f is not None:
        torch.save(
            {"state_dict": trainer.f.module.state_dict()},
            os.path.join(logdir, f"energy_{postfix}.pth"),
        )  # Save the model
    if trainer.v is not None:
        torch.save(
            {"state_dict": trainer.v.module.state_dict()},
            os.path.join(logdir, f"value_{postfix}.pth"),
        )  # Save the model


def calculate_fid(trainer, sampler, logdir, i_iter, device, local_rank, ngpus, cfg, logger, best_fid, m2, s2):
    output_path = os.path.join(logdir, f"img_i_iter_{i_iter}")
    mkdir_p(output_path)
    data_path = os.path.join("datasets", f"cifar10_train_png")
    if not os.path.exists(data_path):
        print0("dataset not found")
        return best_fid, m2, s2

    print0("generating images for FID evaluation")
    sampler.eval()
    n_sample_to_generate = (
        cfg.training.n_fid_samples / cfg.training.sampling_batchsize / ngpus
    )
    i_img = 0
    for i in trange(int(n_sample_to_generate), ncols=80):
        with torch.no_grad():
            d_sample = sampler.sample(
                cfg.training.sampling_batchsize, device=device
            )
            Xi = d_sample["sample"].detach().cpu()
        sample = rescale(Xi).clamp(0, 1).detach().cpu()
        for s in sample:
            save_image(
                s, os.path.join(output_path, f"{local_rank}_{i_img}.png")
            )
            i_img += 1

    torch.distributed.barrier()  # make sure all files are generated
    print0("calculating FID score")
    if local_rank <= 0:
        from pytorch_fid.fid_score import calculate_fid_given_paths_cache

        kwargs = {"batch_size": 50, "device": device, "dims": 2048}
        paths = [output_path, data_path]
        fid, m2, s2 = calculate_fid_given_paths_cache(
            paths=paths, m2=m2, s2=s2, **kwargs
        )
        torch.save({"m2": m2, "s2": s2}, os.path.join(logdir, "fid_stats.pt"))

        if fid < best_fid:  # Check if the current FID score is better than the best FID score
            best_fid = fid  # Update the best FID score
            sampler_ckpt_path = os.path.join(logdir, "sampler.pth")
            print0(f"best FID: sampler saved at {sampler_ckpt_path}")
            torch.save(
                {
                    "state_dict": trainer.sampler.net.module.state_dict(),
                    "fid": fid,
                    "epoch": epoch,
                },
                sampler_ckpt_path,
            )  # Save the model
            torch.save(
                {"state_dict": trainer.v.module.state_dict()},
                os.path.join(logdir, "value.pth"),
            )  # Save the model
        print0(f"FID score: {fid}")
        logger.log({"FID_": fid, "Best_FID_": best_fid}, i_iter)

    return best_fid, m2, s2

def train_one_epoch(
    trainer,
    net,
    dataloader,
    optimizer,
    f,
    v,
    optimizer_fstar,
    optimizer_v,
    n_critic,
    n_generator,
):
    global i_iter, local_rank, batchsize
    global sampler
    global device, logger, cfg
    global best_fid, m2, s2
    # use_guidance = hasattr(trainer, 'guidance_scale') and trainer.guidance_scale
    guidance_scale = cfg.training.get("guidance_scale", None)
    if guidance_scale == 0:
        guidance_scale = None
    state_dict = reset_buffer(device)
    for step, (images, labels) in enumerate(tqdm(dataloader, ncols=80)):
        assert (images.max().item() <= 1) and (0 <= images.min().item())

        # FID calculation
        if cfg.training.fid_every is not None and i_iter % cfg.training.fid_every == 0:
            best_fid, m2, s2 = calculate_fid(trainer, sampler, logdir, i_iter, device, local_rank, ngpus, cfg, logger, best_fid, m2, s2)

        sampler.eval()
        images = (2 * images - 1).to(device)
        if guidance_scale is not None:
            g = torch.rand(1, device=device) * guidance_scale
            d_sample_off = trainer.sample_guidance(
                n_sample=len(images), device=device, guidance_scale=g
            )
            append_buffer(state_dict, d_sample_off)
            d_energy = trainer.update_f_v(images, d_sample_off, state_dict)
        else:
            d_sample = sampler.sample(len(images), device=device)
            append_buffer(state_dict, d_sample)
            d_energy = trainer.update_f_v(images, d_sample, state_dict)
        if (step + 1) % n_critic == 0:
            if cfg.training.get("fresh_sample", False):  # for SGD training
                fresh_sample_grad = cfg.training.get("fresh_sample_grad", False)
                d_sample = sampler.sample(
                    len(images), device=device, enable_grad=fresh_sample_grad
                )
            else:
                d_sample = None
            d_sampler = trainer.update_sampler(
                state_dict, n_generator, d_sample=d_sample
            )
            state_dict = reset_buffer(device)

            if (step + 1) % cfg.training.log_every == 0 and local_rank == 0:
                # also log weight norms
                v_norm = weight_norm(trainer.v)
                s_norm = weight_norm(trainer.sampler.net)
                d_weight = {
                    "weight_norm/sampler_": s_norm,
                    "weight_norm/value_": v_norm,
                }
                logger.log({**d_energy, **d_sampler, **d_weight}, i_iter)

        i_iter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset and model
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to config file. ex) configs/cifar10/T4_ddgan.yaml",
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset config file", required=True
    )
    parser.add_argument("--run", type=str)

    args, unknown = parser.parse_known_args()
    d_cmd_cfg = cmd.parse_unknown_args(unknown)
    d_cmd_cfg = cmd.parse_nested_args(d_cmd_cfg)
    print0("Overriding", d_cmd_cfg)

    # load configs
    cfg = OmegaConf.load(args.config)
    data_cfg = OmegaConf.load(args.dataset)
    cfg = {**cfg, **data_cfg}
    cfg = OmegaConf.merge(
        cfg, OmegaConf.create(d_cmd_cfg)
    )  # override with command line arguments
    print0(OmegaConf.to_yaml(cfg))

    # setting seeds
    torch.backends.cudnn.deterministic = True
    local_rank = int(os.environ["LOCAL_RANK"])
    device = "cuda:{}".format(local_rank)
    seed = cfg.training.seed
    torch.cuda.set_device(device)
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed + local_rank)
    random.seed(seed + local_rank)
    os.environ["PYTHONHASHSEED"] = str(seed + local_rank)

    cfg.training.fid_epoch = cfg.training.get("fid_epoch", None)
    cfg.training.fid_every = cfg.training.get("fid_every", None)
    assert (
        cfg.training.fid_epoch == None or cfg.training.fid_every == None
    ), "cannot set both fid_epoch and fid_every"
    # predefine model
    net = instantiate(cfg.sampler_net)
    print_size(net)
    sample_shape = cfg.sampler.sample_shape
    sampler = instantiate(cfg.sampler, net=net).to(device)
    f = instantiate(cfg.energy).to(device) if cfg.energy is not None else None
    # load checkpoint
    if cfg.training.sampler_ckpt:
        try:
            checkpoint = torch.load(cfg.training.sampler_ckpt, map_location="cpu")
            net.load_state_dict(fix_legacy_dict(checkpoint), strict=False)
            print0(f"Sampler checkpoint loaded from {cfg.training.sampler_ckpt}")
        except:
            raise ValueError("Failed to load valid model checkpoint")
    net = net.to(device)
    net.eval()
    if cfg.value is not None:
        v = instantiate(cfg.value)
        if cfg.training.value_ckpt is not None:
            ckpt = torch.load(cfg.training.value_ckpt, map_location="cpu")
            v.load_pretrained(ckpt)
            print0(
                f"value checkpoint successfully loaded from {cfg.training.value_ckpt}"
            )
        v.to(device)
    else:
        v = None

    ##  optimizer
    tune_beta = hasattr(sampler, "trainable_beta") and sampler.trainable_beta and cfg.training.get("beta_lr") is not None

    if tune_beta:
        params_not_beta = [param for name, param in net.named_parameters() if "log_betas" not in name]
        optimizer = torch.optim.Adam([
            {"params": net.log_betas, "lr": cfg.training.beta_lr},
            {"params": params_not_beta, "lr": cfg.training.lr}
        ])
        print0(f"[Optimizer] Training beta with lr={cfg.training.beta_lr}")
        print0(f"[Optimizer] Training the rest with lr={cfg.training.lr}")
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.training.lr)

    optimizer_v = torch.optim.Adam(v.parameters(), lr=cfg.training.v_lr)

    ngpus = torch.cuda.device_count()
    if ngpus >= 1:
        print0(f"Using distributed training on {ngpus} gpus.")
        batchsize = cfg.training.batchsize // ngpus
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        sampler.net = DDP(
            sampler.net, device_ids=[local_rank], output_device=local_rank
        )
        if f is not None:
            f = DDP(f, device_ids=[local_rank], output_device=local_rank)
        if v is not None:
            v = DDP(v, device_ids=[local_rank], output_device=local_rank)
    # sampler = VARSampler(net, S, sample_shape)

    # train loader
    train_set = loader.get_dataset(cfg.data.name, cfg.data.data_dir)
    # train_set = torch.utils.data.Subset(train_set, np.arange(0, 150))  # useful for debugging
    ds_sampler = DistributedSampler(train_set) if ngpus > 1 else None
    train_loader = DataLoader(
        train_set,
        batch_size=batchsize,
        shuffle=ds_sampler is None,
        sampler=ds_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # trainer
    trainer = instantiate(cfg.trainer, batchsize=batchsize)
    trainer.set_models(
        f=f,
        v=v,
        sampler=sampler,
        optimizer=optimizer,
        optimizer_fstar=None,
        optimizer_v=optimizer_v,
    )

    # train
    model_cfg_name = os.path.basename(args.config).split(".")[0]
    logdir = os.path.join(f"results/{cfg.data.name}/{model_cfg_name}", args.run)
    if local_rank == 0:
        writer = SummaryWriter(logdir=logdir)
        OmegaConf.save(cfg, os.path.join(logdir, "config.yaml"))
        print0(f'Current config file saved to {os.path.join(logdir, "config.yaml")}')
        logger = BaseLogger(writer, use_wandb=True)

        if cfg.data.name != "cifar10":
            raise NotImplementedError

        proj_name = "dxmi_cifar10_ddgan" if "ddgan" in model_cfg_name else f"dxmi_cifar10_T{trainer.sampler.n_timesteps}"
        wandb.init(
            project=proj_name,
            name=f"{model_cfg_name}_{args.run}",
            dir=logdir,
            config=OmegaConf.to_container(cfg),
        )
    i_iter = 0

    ## FID score-related initialization
    best_fid = float("inf")  # Initialize the best FID score
    if cfg.data.name == "cifar10":
        d_fid_stats = torch.load("datasets/cifar10_train_fid_stats.pt")
    else:
        raise NotImplementedError
    m2, s2 = d_fid_stats["m2"], d_fid_stats["s2"]

    ############################
    # Main Training Loop
    ############################
    for epoch in range(cfg.training.n_epochs):
        if local_rank == 0:
            trainer.sampler.eval()
            with torch.no_grad():
                d_sample = sampler.sample(64, device=device)
            Xi = d_sample["sample"].detach().cpu()
            img = make_grid(rescale(Xi), value_range=(0, 1))
            img_norm = Xi.reshape(Xi.shape[0], -1).norm(dim=1).mean().item()
            if epoch == 0:
                logger.log({"sample_init@": img, "sample_norm_": img_norm}, 0)
            else:
                logger.log({"sample@": img, "sample_norm_": img_norm}, i_iter)
            print0("epoch", epoch)

        ############################
        # Compute FID Score
        ############################
        if cfg.training.fid_epoch is not None and epoch % cfg.training.fid_epoch == 0:
            output_path = os.path.join(logdir, f"img_epoch_{epoch}")
            mkdir_p(output_path)
            data_path = os.path.join("datasets", f"cifar10_train_png")
            if not os.path.exists(data_path):
                print0("dataset not found")
                continue

            print0("generating images for FID evaluation")
            net.eval()
            n_sample_to_generate = (
                cfg.training.n_fid_samples / cfg.training.sampling_batchsize / ngpus
            )
            i_img = 0
            for i in trange(int(n_sample_to_generate), ncols=80):
                with torch.no_grad():
                    d_sample = sampler.sample(
                        cfg.training.sampling_batchsize, device=device
                    )
                    Xi = d_sample["sample"].detach().cpu()
                sample = rescale(Xi).clamp(0, 1).detach().cpu()
                for s in sample:
                    save_image(
                        s, os.path.join(output_path, f"{local_rank}_{i_img}.png")
                    )
                    i_img += 1

            torch.distributed.barrier()  # make sure all files are generated
            print0("calculating FID score")
            if local_rank <= 0:
                from pytorch_fid.fid_score import \
                    calculate_fid_given_paths_cache

                # pip install pytorch-fid
                kwargs = {"batch_size": 50, "device": device, "dims": 2048}
                paths = [output_path, data_path]
                fid, m2, s2 = calculate_fid_given_paths_cache(
                    paths=paths, m2=m2, s2=s2, **kwargs
                )
                torch.save({"m2": m2, "s2": s2}, os.path.join(logdir, "fid_stats.pt"))
                # fid = calculate_fid_given_paths(paths=paths, **kwargs)

                if (
                    fid < best_fid
                ):  # Check if the current FID score is better than the best FID score
                    best_fid = fid  # Update the best FID score
                    save_model(
                        trainer,
                        logdir,
                        "best",
                        d_other_info={"fid": fid, "epoch": epoch, "iter": i_iter},
                    )
                    sampler_ckpt_path = os.path.join(logdir, "sampler.pth")
                    print0(f"best FID: sampler saved at {sampler_ckpt_path}")

                print0(f"FID score: {fid}")
                logger.log({"FID_": fid, "Best_FID_": best_fid}, i_iter)

        ############################
        # Actual Training
        ############################
        train_one_epoch(
            trainer=trainer,
            net=net,
            dataloader=train_loader,
            optimizer=optimizer,
            f=f,
            v=v,
            optimizer_fstar=None,
            optimizer_v=optimizer_v,
            n_critic=cfg.training.n_critic,
            n_generator=cfg.training.n_generator,
        )
    if local_rank == 0:  # save the last model
        save_model(
            trainer, logdir, "last", d_other_info={"epoch": epoch, "iter": i_iter}
        )
