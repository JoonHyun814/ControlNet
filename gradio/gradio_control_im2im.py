# add system environment variable
import sys
sys.path.append('/workspace/ControlNet_jh')
sys.path.append('/workspace/ControlNet_jh/ControlNet')

from customs.pipes.control_img2img_pipe import ControlNet_img2img_Pipe
from customs import custom_api
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler

from PIL import Image
import numpy as np
from ControlNet.annotator.canny import CannyDetector
from ControlNet.annotator.openpose import OpenposeDetector
from ControlNet.annotator.util import resize_image, HWC3

import gradio as gr

pretrained_model_name_or_path = "/data3/model_checkpoints/DIFFUSION_DB/ControlNet/diffusers/pose_Realistic_Vision_V1.3"
embed_path = '/data3/model_checkpoints/DIFFUSION_DB/embeddings/v15'
embed_list = ['bad-hands','real-korean-style']


# load tokenizer
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
)
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, 
    subfolder="text_encoder", 
    torch_dtype=torch.float16
)

# add tokens
custom_api.add_token(embed_path=embed_path,tokens=embed_list,tokenizer=tokenizer,text_encoder=text_encoder)

# load scheduler
# scheduler = EulerAncestralDiscreteScheduler()

# load controlNet img2img model
pipe = ControlNet_img2img_Pipe.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    text_encoder=text_encoder,
    # scheduler = scheduler,
    tokenizer=tokenizer
    ).to("cuda")


def inference(prompt,negative_prompt,image,model_type,num_images_per_prompt,strength,num_inference_steps,guidance_scale,seed):
    # load init image
    # image preprocess
    image = custom_api.resize_padding(Image.fromarray(image))
    image = np.asanyarray(image)
    image = resize_image(HWC3(image), 512)


    # detect condition
    if model_type == 'canny':
        canny_detector = CannyDetector()
        detected_map = canny_detector(image, 100, 200)
    elif model_type == 'pose':
        apply_openpose = OpenposeDetector()
        detected_map, _ = apply_openpose(image)

    # inputs to PIL Image
    detected_map = Image.fromarray(detected_map)
    image = Image.fromarray(image)

    # inference
    custom_api.seed_everything(int(seed))
    image_out = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        image = image,
        controlnet_hint=detected_map,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt
        ).images
    
    image_out.insert(0,detected_map)
    
    return image_out

with gr.Blocks() as demo:
    inputs = {}
    with gr.Row():
        with gr.Column():
            inputs['prompt'] = gr.Text(label='prompt')
            inputs['negative_prompt'] = gr.Text(label='negative_prompt')
            inputs['image'] = gr.Image(label='image')
            inputs['model_type'] = gr.Dropdown(choices=['pose','canny'],value='pose',label='model_type')
            masking_button = gr.Button("generate")
            inputs['num_images_per_prompt'] = gr.Slider(minimum=1,maximum=8,value=1,step=1,label='num_images_per_prompt')
            inputs['strength'] = gr.Slider(minimum=0.01,maximum=1,value=0.7,step=0.01,label='strength')
            inputs['num_inference_steps'] = gr.Slider(minimum=1,maximum=100,value=20,step=1,label='num_inference_steps')
            inputs['guidance_scale'] = gr.Slider(minimum=1,maximum=20,value=7,step=1,label='guidance_scale')
            inputs['seed'] = gr.Number(value=42,step=1,label='seed')
        masking_output = gr.Gallery()
            
    masking_button.click(
        inference,
        inputs=[inputs[k] for k in list(inputs.keys())],
        outputs=masking_output
    )
    
demo.launch(server_name='0.0.0.0',server_port=5050)