from ControlNet.cldm.model import create_model, load_state_dict
from ControlNet.cldm.ddim_hacked import DDIMSampler

import torch

import custom_api

model = create_model('/workspace/ControlNet_jh/ControlNet/models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('/workspace/ControlNet_jh/ControlNet/models/control_sd15_openpose.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

# load trained SD
sd_config = '/data3/model_checkpoints/DIFFUSION_DB/Diffusion_models/configs/v1-inference.yaml'
sd_ckpt = '/data3/model_checkpoints/DIFFUSION_DB/Diffusion_models/diffusers_trained/exp_v1.5/razaras_superschool/1200/model.ckpt'
custom_api.load_sd(model,sd_config,sd_ckpt)

# model save 
out_path = '/workspace/ControlNet_jh/models/controlnet/pose_razaras-superschool-1200.pth'
torch.save(model.state_dict(),out_path)