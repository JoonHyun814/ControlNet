import torch
from ControlNet.ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import os

import PIL
from PIL import Image
import numpy as np
import random


def resize_padding(image, out_size = 512):
    old_size = image.size
    
    ratio = float(out_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    
    image = image.resize(new_size, Image.ANTIALIAS)
    out_image = Image.new("RGB", (out_size, out_size))
    
    out_image.paste(image, ((out_size-new_size[0])//2,(out_size-new_size[1])//2))
    return out_image


def add_token(embed_path,tokens,model=None,tokenizer=None,text_encoder=None):
    # Load tokenizer and text encoder
    if model:
        tokenizer = model.cond_stage_model.tokenizer
        text_encoder = model.cond_stage_model.transformer

    # add tokens to tokenizer
    for token in tokens:
        num_added_tokens = tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")

    # resize token embeddings of text encoder
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get token embeddings
    token_embeds = text_encoder.get_input_embeddings().weight.data

    # insert new token embeddings
    for token in tokens:
        # token id and path
        token_id = tokenizer.convert_tokens_to_ids(token)
        token_path = os.path.join(embed_path, token + '.pt')

        # load token pt file
        ptfile = torch.load(token_path)

        # insert token embedding
        new_embedding = ptfile['string_to_param']['*'][0].cpu()
        token_embeds[token_id] = new_embedding
        print(token, "added to CLIP tokenizer")
        
        
def load_sd(model,sd_config,sd_ckpt):
    config = OmegaConf.load(sd_config)
    pl_sd = torch.load(sd_ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    sd_model = instantiate_from_config(config.model)
    sd_model.load_state_dict(sd, strict=False)
    
    model.model.load_state_dict(sd_model.model.state_dict())
    model.first_stage_model.load_state_dict(sd_model.first_stage_model.state_dict())
    model.cond_stage_model.load_state_dict(sd_model.cond_stage_model.state_dict())
    
    
def load_img(np_img):
    image = Image.fromarray(np_img).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore