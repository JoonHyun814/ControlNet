import os
import re
import tempfile
from typing import Optional

import requests
import torch
from transformers import (
    AutoFeatureExtractor,
    BertTokenizerFast,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LDMTextToImagePipeline,
    LMSDiscreteScheduler,
    PNDMScheduler,
    PriorTransformer,
    StableDiffusionInpaintPipeline,
    StableUnCLIPImg2ImgPipeline,
    StableUnCLIPPipeline,
    UnCLIPScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.paint_by_example import PaintByExamplePipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    create_unet_diffusers_config, 
    convert_ldm_unet_checkpoint, 
    create_vae_diffusers_config, 
    convert_ldm_vae_checkpoint,
    stable_unclip_image_noising_components,
    convert_open_clip_checkpoint,
    stable_unclip_image_encoder,
    convert_paint_by_example_checkpoint,
    convert_ldm_clip_checkpoint,
    create_ldm_bert_config,
    convert_ldm_bert_checkpoint,
    logger
    )

def load_pipeline_inpainting_stable_diffusion_ckpt(
    checkpoint_path: str,
    original_config_file: str = None,
    image_size: int = 512,
    prediction_type: str = None,
    model_type: str = None,
    extract_ema: bool = False,
    scheduler_type: str = "pndm",
    num_in_channels: Optional[int] = None,
    upcast_attention: Optional[bool] = None,
    device: str = None,
    from_safetensors: bool = False,
    stable_unclip: Optional[str] = None,
    stable_unclip_prior: Optional[str] = None,
    clip_stats_path: Optional[str] = None,
) -> StableDiffusionInpaintPipeline:
    """
    Load a Stable Diffusion pipeline object from a CompVis-style `.ckpt`/`.safetensors` file and (ideally) a `.yaml`
    config file.

    Although many of the arguments can be automatically inferred, some of these rely on brittle checks against the
    global step count, which will likely fail for models that have undergone further fine-tuning. Therefore, it is
    recommended that you override the default values and/or supply an `original_config_file` wherever possible.

    Args:
        checkpoint_path (`str`): Path to `.ckpt` file.
        original_config_file (`str`):
            Path to `.yaml` config file corresponding to the original architecture. If `None`, will be automatically
            inferred by looking for a key that only exists in SD2.0 models.
        image_size (`int`, *optional*, defaults to 512):
            The image size that the model was trained on. Use 512 for Stable Diffusion v1.X and Stable Diffusion v2
            Base. Use 768 for Stable Diffusion v2.
        prediction_type (`str`, *optional*):
            The prediction type that the model was trained on. Use `'epsilon'` for Stable Diffusion v1.X and Stable
            Diffusion v2 Base. Use `'v_prediction'` for Stable Diffusion v2.
        num_in_channels (`int`, *optional*, defaults to None):
            The number of input channels. If `None`, it will be automatically inferred.
        scheduler_type (`str`, *optional*, defaults to 'pndm'):
            Type of scheduler to use. Should be one of `["pndm", "lms", "heun", "euler", "euler-ancestral", "dpm",
            "ddim"]`.
        model_type (`str`, *optional*, defaults to `None`):
            The pipeline type. `None` to automatically infer, or one of `["FrozenOpenCLIPEmbedder",
            "FrozenCLIPEmbedder", "PaintByExample"]`.
        extract_ema (`bool`, *optional*, defaults to `False`): Only relevant for
            checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights or not. Defaults to
            `False`. Pass `True` to extract the EMA weights. EMA weights usually yield higher quality images for
            inference. Non-EMA weights are usually better to continue fine-tuning.
        upcast_attention (`bool`, *optional*, defaults to `None`):
            Whether the attention computation should always be upcasted. This is necessary when running stable
            diffusion 2.1.
        device (`str`, *optional*, defaults to `None`):
            The device to use. Pass `None` to determine automatically. :param from_safetensors: If `checkpoint_path` is
            in `safetensors` format, load checkpoint with safetensors instead of PyTorch. :return: A
            StableDiffusionInpaintPipeline object representing the passed-in `.ckpt`/`.safetensors` file.
    """
    if prediction_type == "v-prediction":
        prediction_type = "v_prediction"

    # if not is_omegaconf_available():
    #     raise ValueError(BACKENDS_MAPPING["omegaconf"][1])

    from omegaconf import OmegaConf

    # if from_safetensors:
    #     if not is_safetensors_available():
    #         raise ValueError(BACKENDS_MAPPING["safetensors"][1])

    #     from safetensors import safe_open

    #     checkpoint = {}
    #     with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
    #         for key in f.keys():
    #             checkpoint[key] = f.get_tensor(key)
    # else:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    # Sometimes models don't have the global_step item
    if "global_step" in checkpoint:
        global_step = checkpoint["global_step"]
    else:
        print("global_step key not found in model")
        global_step = None

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    with tempfile.TemporaryDirectory() as tmpdir:
        if original_config_file is None:
            key_name = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"

            original_config_file = os.path.join(tmpdir, "inference.yaml")
            if key_name in checkpoint and checkpoint[key_name].shape[-1] == 1024:
                if not os.path.isfile("v2-inference-v.yaml"):
                    # model_type = "v2"
                    r = requests.get(
                        " https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml"
                    )
                    open(original_config_file, "wb").write(r.content)

                if global_step == 110000:
                    # v2.1 needs to upcast attention
                    upcast_attention = True
            else:
                if not os.path.isfile("v1-inference.yaml"):
                    # model_type = "v1"
                    r = requests.get(
                        " https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
                    )
                    open(original_config_file, "wb").write(r.content)

        original_config = OmegaConf.load(original_config_file)

    if num_in_channels is not None:
        original_config["model"]["params"]["unet_config"]["params"]["in_channels"] = num_in_channels

    if (
        "parameterization" in original_config["model"]["params"]
        and original_config["model"]["params"]["parameterization"] == "v"
    ):
        if prediction_type is None:
            # NOTE: For stable diffusion 2 base it is recommended to pass `prediction_type=="epsilon"`
            # as it relies on a brittle global step parameter here
            prediction_type = "epsilon" if global_step == 875000 else "v_prediction"
        if image_size is None:
            # NOTE: For stable diffusion 2 base one has to pass `image_size==512`
            # as it relies on a brittle global step parameter here
            image_size = 512 if global_step == 875000 else 768
    else:
        if prediction_type is None:
            prediction_type = "epsilon"
        if image_size is None:
            image_size = 512

    num_train_timesteps = original_config.model.params.timesteps
    beta_start = original_config.model.params.linear_start
    beta_end = original_config.model.params.linear_end

    scheduler = DDIMScheduler(
        beta_end=beta_end,
        beta_schedule="scaled_linear",
        beta_start=beta_start,
        num_train_timesteps=num_train_timesteps,
        steps_offset=1,
        clip_sample=False,
        set_alpha_to_one=False,
        prediction_type=prediction_type,
    )
    # make sure scheduler works correctly with DDIM
    scheduler.register_to_config(clip_sample=False)

    if scheduler_type == "pndm":
        config = dict(scheduler.config)
        config["skip_prk_steps"] = True
        scheduler = PNDMScheduler.from_config(config)
    elif scheduler_type == "lms":
        scheduler = LMSDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "heun":
        scheduler = HeunDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "euler":
        scheduler = EulerDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "euler-ancestral":
        scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "dpm":
        scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
    elif scheduler_type == "ddim":
        scheduler = scheduler
    else:
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")

    # Convert the UNet2DConditionModel model.
    unet_config = create_unet_diffusers_config(original_config, image_size=image_size)
    unet_config["upcast_attention"] = upcast_attention
    unet = UNet2DConditionModel(**unet_config)

    converted_unet_checkpoint = convert_ldm_unet_checkpoint(
        checkpoint, unet_config, path=checkpoint_path, extract_ema=extract_ema
    )

    unet.load_state_dict(converted_unet_checkpoint)

    # Convert the VAE model.
    vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)

    # Convert the text model.
    if model_type is None:
        model_type = original_config.model.params.cond_stage_config.target.split(".")[-1]
        logger.debug(f"no `model_type` given, `model_type` inferred as: {model_type}")

    if model_type == "FrozenOpenCLIPEmbedder":
        text_model = convert_open_clip_checkpoint(checkpoint)
        tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2", subfolder="tokenizer")

        if stable_unclip is None:
            pipe = StableDiffusionInpaintPipeline(
                vae=vae,
                text_encoder=text_model,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
        else:
            image_normalizer, image_noising_scheduler = stable_unclip_image_noising_components(
                original_config, clip_stats_path=clip_stats_path, device=device
            )

            if stable_unclip == "img2img":
                feature_extractor, image_encoder = stable_unclip_image_encoder(original_config)

                pipe = StableUnCLIPImg2ImgPipeline(
                    # image encoding components
                    feature_extractor=feature_extractor,
                    image_encoder=image_encoder,
                    # image noising components
                    image_normalizer=image_normalizer,
                    image_noising_scheduler=image_noising_scheduler,
                    # regular denoising components
                    tokenizer=tokenizer,
                    text_encoder=text_model,
                    unet=unet,
                    scheduler=scheduler,
                    # vae
                    vae=vae,
                )
            elif stable_unclip == "txt2img":
                if stable_unclip_prior is None or stable_unclip_prior == "karlo":
                    karlo_model = "kakaobrain/karlo-v1-alpha"
                    prior = PriorTransformer.from_pretrained(karlo_model, subfolder="prior")

                    prior_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
                    prior_text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")

                    prior_scheduler = UnCLIPScheduler.from_pretrained(karlo_model, subfolder="prior_scheduler")
                    prior_scheduler = DDPMScheduler.from_config(prior_scheduler.config)
                else:
                    raise NotImplementedError(f"unknown prior for stable unclip model: {stable_unclip_prior}")

                pipe = StableUnCLIPPipeline(
                    # prior components
                    prior_tokenizer=prior_tokenizer,
                    prior_text_encoder=prior_text_model,
                    prior=prior,
                    prior_scheduler=prior_scheduler,
                    # image noising components
                    image_normalizer=image_normalizer,
                    image_noising_scheduler=image_noising_scheduler,
                    # regular denoising components
                    tokenizer=tokenizer,
                    text_encoder=text_model,
                    unet=unet,
                    scheduler=scheduler,
                    # vae
                    vae=vae,
                )
            else:
                raise NotImplementedError(f"unknown `stable_unclip` type: {stable_unclip}")
    elif model_type == "PaintByExample":
        vision_model = convert_paint_by_example_checkpoint(checkpoint)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        feature_extractor = AutoFeatureExtractor.from_pretrained("CompVis/stable-diffusion-safety-checker")
        pipe = PaintByExamplePipeline(
            vae=vae,
            image_encoder=vision_model,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=feature_extractor,
        )
    elif model_type == "FrozenCLIPEmbedder":
        text_model = convert_ldm_clip_checkpoint(checkpoint)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
        feature_extractor = AutoFeatureExtractor.from_pretrained("CompVis/stable-diffusion-safety-checker")
        pipe = StableDiffusionInpaintPipeline(
            vae=vae,
            text_encoder=text_model,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
    else:
        text_config = create_ldm_bert_config(original_config)
        text_model = convert_ldm_bert_checkpoint(checkpoint, text_config)
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        pipe = LDMTextToImagePipeline(vqvae=vae, bert=text_model, tokenizer=tokenizer, unet=unet, scheduler=scheduler)

    return pipe


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="dir path to model ckpt"
    )
    parser.add_argument(
        "--original_config_file",
        type=str,
        default=None,
        help="dir path to model config"
    )
    parser.add_argument(
        "--dump_path",
        type=str,
        default=None,
        help="dir path to saved model"
    )
    
    args = parser.parse_args()

    pipe = load_pipeline_inpainting_stable_diffusion_ckpt(
        checkpoint_path = args.checkpoint_path,
        original_config_file = args.original_config_file
    )

    pipe.save_pretrained(args.dump_path)