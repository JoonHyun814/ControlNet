# ControlNet_diffusers


## Setting

docker image: nvidia/cuda:11.3.0-base-ubuntu20.04

python version: 3.8


```bash
git clone https://github.com/JoonHyun814/ControlNet_diffusers.git
cd ControlNet_diffusers

git clone https://github.com/lllyasviel/ControlNet.git

git clone diffusers
cd diffusers
git checkout 9a37409663a53f775fa380db332d37d7ea75c915
pip install .
cd ..

apt-get install libglib2.0-0
apt-get install -y libsm6 libxext6 libxrender-dev
apt-get -y install libgl1-mesa-glx

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## download model

huggingface: lllyasviel/ControlNet/models




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
