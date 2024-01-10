import os
import torch

from diffusers import AutoencoderKL

torch.manual_seed(123456)
x = torch.randn(1, 3, 512, 512)

# Mist

# from omegaconf import OmegaConf
# from mist_v3 import load_model_from_config

# device = torch.device('cpu')

# ckpt = 'models/ldm/stable-diffusion-v1/model.ckpt'
# base = 'configs/stable-diffusion/v1-inference-attack.yaml'

# config_path = os.path.join(os.getcwd(), base)
# config = OmegaConf.load(config_path)

# ckpt_path = os.path.join(os.getcwd(), ckpt)
# model = load_model_from_config(config, ckpt_path).to(device)


# z1 = model.encode_first_stage(x).mean
# print(z1)


# SD 2.1

model2 = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae")

print("z2", model2.encode(x).latent_dist.mean)
