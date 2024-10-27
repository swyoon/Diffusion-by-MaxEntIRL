"""
Run DxMI training for large datasets (ImageNet, LSUN, ...).
Example command:
$ CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_image_large.py \
    --config configs/imagenet64/T10.yaml --dataset configs/imagenet64/imagenet64.yaml

$ CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_image_large.py \
    --config configs/imagenet64/T4.yaml --dataset configs/imagenet64/imagenet64.yaml

$ CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_image_large.py \
    --config configs/lsun/T4.yaml --dataset configs/lsun/bedroom.yaml
"""
import argparse
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim import RAdam, Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm, trange
from hydra.utils import instantiate
from tensorboardX import SummaryWriter
from models.logger import BaseLogger
import wandb
wandb.require("core")

import cmd_utils as cmd
from models.cm.fp16_util import MixedPrecisionTrainer
from models.cm.dxmi_util import infinite_loader, load_data
from models.cm.script_util import create_model_and_diffusion
from models.DxMI.openai_diffusion import OpenAIDiffusion
from models.DxMI.trainer import append_buffer, reset_buffer
from models.utils import print0, weight_norm

def generate():
    n_batch_to_generate = cfg.training.n_fid_samples / cfg.training.sampling_batchsize / ngpus 
    l_sample = []
    sampler.eval()
    for i_batch in trange(int(n_batch_to_generate), ncols=80):
        d_sample = sampler.sample(cfg.training.sampling_batchsize, device=device, 
                i_class=None, enable_grad=False)
        sample = d_sample['sample']
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        l_sample.append(sample)
    samples = torch.cat(l_sample)
    all_samples = [torch.zeros_like(samples) for _ in range(ngpus)]
    torch.distributed.all_gather(all_samples, samples)
    all_samples = torch.cat([t.detach().cpu() for t in all_samples])
    return all_samples


def fid():
    global best_fid 
    all_samples = generate()

    split_samples = all_samples[local_rank::ngpus]
    split_samples = (split_samples / 255.).float()
    act = get_activations_from_tensor(split_samples, extractor, batch_size=50, dims=dims, device=device)
    ## all gather act
    all_act = [torch.zeros_like(act) for _ in range(ngpus)]
    torch.distributed.all_gather(all_act, act)
    if local_rank <= 0:
        act = torch.cat([a.detach().cpu() for a in all_act]).detach().numpy()
        m1, s1 = np.mean(act, axis=0), np.cov(act, rowvar=False)
        fid  = calculate_frechet_distance(mu1=m1, sigma1=s1, mu2=m2, sigma2=s2)
        print(f'FID: {fid}')
    
        logger.log({'fid': fid}, i_iter)
        if fid < best_fid:
            best_fid = fid
            np.savez(os.path.join(logdir, f'best_samples.npz'), 
                     all_samples.permute(0,2,3,1).cpu().numpy())
            sampler_ckpt_path = os.path.join(logdir, 'sampler.pth')
            print0(f'best FID: sampler saved at {sampler_ckpt_path}')
            torch.save({'state_dict': trainer.sampler.net.module.state_dict(),
                        'fid': fid,
                        'i_iter': i_iter}, 
                    sampler_ckpt_path)  # Save the model
            torch.save({'state_dict': trainer.v.module.state_dict()}, 
                        os.path.join(logdir, 'value.pth'))  # Save the model

            print(f'Best model saved to {os.path.join(logdir, "best_model.pth")}')
        logger.log({'FID_': fid, 'Best_FID_': best_fid}, i_iter)
    torch.distributed.barrier()  # make sure all files are generated


local_rank = int(os.environ.get("LOCAL_RANK", 0))

########### Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='path to config file. ex) configs/cifar10/gcd.yaml')
parser.add_argument("--dataset", type=str, help="Dataset config file", required=True)
parser.add_argument('--run', type=str, required=True)

args, unknown = parser.parse_known_args()
d_cmd_cfg = cmd.parse_unknown_args(unknown)
d_cmd_cfg = cmd.parse_nested_args(d_cmd_cfg)
if local_rank == 0:
    print("Overriding", d_cmd_cfg)

############ load configs
cfg = OmegaConf.load(args.config)
data_cfg = OmegaConf.load(args.dataset)
cfg = {**cfg, **data_cfg}
cfg = OmegaConf.merge(
    cfg, OmegaConf.create(d_cmd_cfg)
)  # override with command line arguments
if local_rank == 0:
    print(OmegaConf.to_yaml(cfg))


############# setting seeds
torch.backends.cudnn.deterministic = True
device = "cuda:{}".format(local_rank)
seed = cfg.training.seed
torch.cuda.set_device(device)
torch.manual_seed(seed + local_rank)
np.random.seed(seed + local_rank)
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(seed + local_rank)
random.seed(seed + local_rank)
os.environ['PYTHONHASHSEED'] = str(seed + local_rank)


############# Model
unet, diffusion = create_model_and_diffusion(
    **cfg.diffusion,
)
# diffusion -> KarrasDenoiser

unet.load_state_dict(torch.load(cfg.training.pretrained_path, map_location="cpu"))
sampler = OpenAIDiffusion(unet, diffusion, **cfg.sampler)
unet.to(device)

if cfg.diffusion.use_fp16:
    unet.convert_to_fp16()

if cfg.value is not None:
    v = instantiate(cfg.value) 
    if cfg.training.value_ckpt is not None:
        ckpt = torch.load(cfg.training.value_ckpt, map_location='cpu')
        v.load_pretrained(ckpt)
        print0(f'value checkpoint successfully loaded from {cfg.training.value_ckpt}')
    v.to(device)
else:
    v = None

############# Optimizer
print0('using MixedPrecisionTrainer')

if cfg.training.get('beta_lr', None) is not None:
    print0(f'using beta_lr: {cfg.training.beta_lr}')
    mp_trainer = MixedPrecisionTrainer(model=unet, use_fp16=cfg.diffusion.use_fp16, 
                                        initial_lg_loss_scale=cfg.training.initial_log_loss_scale,
                                        special_key='log_betas')
    opt = RAdam([{'params': mp_trainer.master_params[1:], 'lr': cfg.training.lr},
                {'params': mp_trainer.master_params[0:1], 'lr': cfg.training.beta_lr}], 
                    weight_decay=cfg.training.weight_decay)
else:
    mp_trainer = MixedPrecisionTrainer(model=unet, use_fp16=cfg.diffusion.use_fp16, 
                                        initial_lg_loss_scale=cfg.training.initial_log_loss_scale)
    opt = RAdam(
        mp_trainer.master_params, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
opt_v = Adam(v.parameters(), lr=cfg.training.v_lr)


############# DDP 
torch.distributed.init_process_group(backend="nccl", init_method="env://")
sampler.net = DDP(unet, device_ids=[local_rank], output_device=local_rank)
v = DDP(v, device_ids=[local_rank], output_device=local_rank) if v is not None else None

############# load data
ds = load_data(data_dir=cfg.data.data_dir, 
               cachefile=cfg.data.cachefile,
               batch_size=cfg.training.batchsize, 
               image_size=cfg.data.image_size, 
               class_cond=cfg.data.class_cond, 
               deterministic=cfg.data.deterministic,
               random_crop=False, random_flip=True)
# ds = torch.utils.data.Subset(ds, np.arange(0, 150))  # useful for debugging

class_cond = cfg.data.class_cond
n_class = cfg.data.n_class 
if local_rank == 0: print(f'Classes {n_class}')
ngpus = torch.cuda.device_count()
print(f'ngpus: {ngpus}')
ds_sampler = DistributedSampler(ds) if ngpus > 1 else None
batchsize = cfg.training.batchsize // ngpus
train_loader = DataLoader(
    ds,
    batch_size=batchsize,
    shuffle=ds_sampler is None,
    sampler=ds_sampler,
    num_workers=4,
    pin_memory=False,
    drop_last=True,
)

loader = infinite_loader(train_loader)

############# Directory and wandb setting
model_cfg_name = os.path.basename(args.config).split(".")[0]
logdir = os.path.join(f"results/{cfg.data.name}/{model_cfg_name}", args.run)
os.environ["OPENAI_LOGDIR"] = logdir

best_fid = float('inf')
if local_rank == 0:
    writer = SummaryWriter(logdir=logdir)
    OmegaConf.save(cfg, os.path.join(logdir, 'config.yaml'))
    print(f'Config saved to {os.path.join(logdir, "config.yaml")}')
    logger = BaseLogger(writer, use_wandb=True)
    
    dataset_name = cfg.data.name
    proj_name = f'dxmi_{dataset_name}_T{cfg.sampler.n_timesteps}'
    
    wandb.init(project=proj_name, name=f'{model_cfg_name}_{args.run}', dir=logdir,
               config=OmegaConf.to_container(cfg))
    
    if dataset_name == 'imagenet64':
        fid_file = 'datasets/VIRTUAL_imagenet64_labeled.npz'
    elif dataset_name == 'lsun_bedroom':
        fid_file = 'datasets/VIRTUAL_lsun_bedroom256.npz'
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset_name}")
    
    d_fid_stats = np.load(fid_file)
    m2, s2 = torch.tensor(d_fid_stats['mu']), torch.tensor(d_fid_stats['sigma'])

############ Trainer
trainer = instantiate(cfg.trainer, batchsize=batchsize)
trainer.set_models(v=v, sampler=sampler, optimizer=opt, optimizer_v=opt_v)


############# FID
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import get_activations_from_tensor, calculate_frechet_distance
dims = 2048
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
extractor = InceptionV3([block_idx]).to(device)
extractor.eval()


############# iterate over data
state_dict = reset_buffer(device)
for i_iter in tqdm(range(cfg.training.n_iter)):
    data, cond = next(loader)
    data = data.to(device)
    cond = {k: v.to(device) for k, v in cond.items()}
    y = cond.get('y', None)

    if cfg.training.fid_every is not None and i_iter % cfg.training.fid_every == 0:
        fid()

    d_sample = sampler.sample(len(data), device=device, i_class=y)
    append_buffer(state_dict, d_sample)
    d_energy = trainer.update_f_v(data, d_sample, state_dict, y=y)
    d_sampler = trainer.update_sampler_mixed_precision(state_dict, mp_trainer=mp_trainer, d_sample=d_sample)
    state_dict = reset_buffer(device)

    if (i_iter+1) % cfg.training.log_every == 0 and local_rank == 0:
        v_norm = weight_norm(trainer.v)
        s_norm = weight_norm(trainer.sampler.net)
        d_weight = {'weight_norm/sampler_': s_norm, 'weight_norm/value_': v_norm} 
        logger.log({**d_energy, **d_sampler, **d_weight}, i_iter)
