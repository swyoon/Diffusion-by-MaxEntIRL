"""
diffusion model training for large datasets (ImageNet, LSUN, ...)
"""
import argparse
import os
import pickle
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm, trange

from models.cm.script_util import create_model_and_diffusion
from models.DxMI.openai_diffusion import OpenAIDiffusion

from utils import print0, mkdir_p

def generate():
    global output_path, local_rank
    n_batch_to_generate = args.n_sample / args.batchsize / ngpus 
    l_sample = []
    sampler.eval()
    i_img = 0
    for i_batch in trange(int(n_batch_to_generate), ncols=80):
        if args.guidance_scale is not None:
            d_sample = trainer.sample_guidance(n_sample=args.batchsize, device=device, guidance_scale=args.guidance_scale)
        else:
            d_sample = sampler.sample(args.batchsize, device=device, 
                    i_class=None, enable_grad=False)
        sample = d_sample['sample']
        
        if args.skip_fid:
            sample = ((sample + 1)/2).clamp(0, 1)
            for s in sample:
                save_image(s, os.path.join(output_path, f"{local_rank}_{i_img}.png"))
                i_img += 1
            l_sample.append(sample.detach().cpu())
        else:
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            l_sample.append(sample)

    samples = torch.cat(l_sample)
    
    if not args.skip_fid:
        all_samples = [torch.zeros_like(samples) for _ in range(ngpus)]
        torch.distributed.all_gather(all_samples, samples)
        all_samples = torch.cat([t.detach().cpu() for t in all_samples])
        return all_samples
    else:
        return samples


def fid():
    all_samples = generate()
    print(f'generated {len(all_samples)}')

    if not args.skip_fid:
        split_samples = all_samples[local_rank::ngpus]
        split_samples = (split_samples / 255.).float()
        act = get_activations_from_tensor(split_samples, extractor, batch_size=50, dims=dims, device=device)
        all_act = [torch.zeros_like(act) for _ in range(ngpus)]
        torch.distributed.all_gather(all_act, act)
        if local_rank <= 0:
            act = torch.cat([a.detach().cpu() for a in all_act]).detach().numpy()
            m1, s1 = np.mean(act, axis=0), np.cov(act, rowvar=False)
            fid  = calculate_frechet_distance(mu1=m1, sigma1=s1, mu2=m2, sigma2=s2)
            print(f'FID from {len(all_samples)} samples: {fid}')
        
            if args.guidance_scale is not None:
                np.savez(os.path.join(args.log_dir, f'samples_{len(all_samples)}_{args.guidance_scale}.npz'), 
                         all_samples.permute(0,2,3,1).cpu().numpy())
            else:
                np.savez(os.path.join(args.log_dir, f'samples_{len(all_samples)}.npz'), 
                         all_samples.permute(0,2,3,1).cpu().numpy())

    return all_samples


local_rank = int(os.environ.get("LOCAL_RANK", 0))

########### Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, required=True, help='path to logdir')
parser.add_argument('--n_sample', type=int, required=True)
parser.add_argument('--batchsize', type=int, default=100)
parser.add_argument('--guidance_scale', type=float, default=None, help='Value guidance scale, 0.0 for no guidance')
parser.add_argument('--skip_fid', action='store_true', 
                    help='''Do not calculate FID. Instead, save individual images as they are generated. 
                    This option avoids memory issues with large images, such as LSUN''')

args, unknown = parser.parse_known_args()

############ load configs
cfg = OmegaConf.load(os.path.join(args.log_dir, 'config.yaml'))
print(OmegaConf.to_yaml(cfg))

ngpus = torch.cuda.device_count()
print(f'using {ngpus} GPUs')

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
unet, diffusion = create_model_and_diffusion(**cfg.diffusion)
ckpt_path = os.path.join(args.log_dir, 'sampler.pth')
output_path = os.path.join(args.log_dir, f'generated')
mkdir_p(output_path)
ckpt_dict = torch.load(ckpt_path)
print0(f'checkpoint loaded from {ckpt_path}')
print0(f'Best FID: {ckpt_dict["fid"]}')
print0(f'iter: {ckpt_dict["i_iter"]}')
sampler = OpenAIDiffusion(unet, diffusion, **cfg.sampler)
sampler.net.load_state_dict(ckpt_dict['state_dict'])
sampler.net.to(device)

if cfg.diffusion.use_fp16:
    unet.convert_to_fp16()

if args.guidance_scale is not None:
    print("Loading value function for guidance")
    value_path = os.path.join(args.log_dir, 'value.pth')
    if not os.path.exists(value_path):
        raise ValueError("Value ftn not found at {}".format(value_path))
    v = instantiate(cfg.value).to(device)
    checkpoint = torch.load(value_path, map_location=device)
    v.load_state_dict(checkpoint['state_dict'])
    v.eval()
else:
    v = None

if args.guidance_scale is not None:
    trainer = instantiate(cfg.trainer, batchsize=args.batchsize)
    trainer.set_models(f=None, v=v, sampler=sampler, optimizer=None, 
                            optimizer_fstar=None, optimizer_v=None)

############# DDP 
torch.distributed.init_process_group(backend="nccl", init_method="env://")
sampler.net = DDP(unet, device_ids=[local_rank], output_device=local_rank)

############# FID
if not args.skip_fid:
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import get_activations_from_tensor, calculate_frechet_distance
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    extractor = InceptionV3([block_idx]).to(device)
    extractor.eval()

    if cfg.data.name == 'imagenet64':
        d_fid_stats = np.load('datasets/VIRTUAL_imagenet64_labeled.npz')
    elif cfg.data.name == 'lsun_bedroom':
        d_fid_stats = np.load('datasets/VIRTUAL_lsun_bedroom256.npz')
    else:
        raise NotImplementedError
    m2, s2 = torch.tensor(d_fid_stats['mu']), torch.tensor(d_fid_stats['sigma'])

samples = fid()