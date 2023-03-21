# ControlNet


## Setting
```bash
git clone https://github.com/lllyasviel/ControlNet.git

git clone diffusers
cd diffusers
git checkout 9a37409663a53f775fa380db332d37d7ea75c915
pip install .
cd ..
```

## Convert ControlNet.ckpt to diffusers format
```bash
python ./diffusers/scripts/convert_controlnet_to_diffusers.py \
    --checkpoint_path $CONTROLNET_CKPT_PATH\
    --original_config_file $CONTROLNET_CONFIG_PATH\
    --dump_path $OUT_PATH
```

## Convert SD-inpainting.ckpt to diffusers format
```bash
python ./convert_inpainting_to_diffusers.py \
    --checkpoint_path $SD_INPAINTING_CKPT_PATH \
    --original_config_file $SD_INPAINTING_CONFIG_PATH\
    --dump_path $OUT_PATH
```

## Convert SD.ckpt to diffusers format
```bash
python ./diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py \
    --checkpoint_path $SD_CKPT_PATH \
    --original_config_file $SD_CONFIG_PATH\
    --dump_path $OUT_PATH
```

## Convert SD.ckpt to ControlNet.ckpt
```bash
python ./tool_transfer_control.py \
    --controlnet_path $CONTROLNET_CKPT_PATH \
    --sd_path $SD_CKPT_PATH \
    --dump_path $OUT_PATH
```