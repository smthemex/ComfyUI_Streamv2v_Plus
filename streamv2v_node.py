# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import hashlib
import random
import re
import time
from typing import Literal, Dict, Optional
import fire
from torchvision.io import read_video, write_video
from tqdm import tqdm
from .utils.wrapper import StreamV2VWrapper
import torch
import os
from PIL import Image
import numpy as np
import sys
import folder_paths
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
path_dir = os.path.dirname(CURRENT_DIR)
file_path = os.path.dirname(path_dir)
BIGMAX = (2 ** 53 - 1)
input_path = folder_paths.get_input_directory()

paths = []
for search_path in folder_paths.get_folder_paths("diffusers"):
    if os.path.exists(search_path):
        for root, subdir, files in os.walk(search_path, followlinks=True):
            if "model_index.json" in files:
                paths.append(os.path.relpath(root, start=search_path))

if paths != []:
    paths = ["none"] + [x for x in paths if x]
else:
    paths = ["none",]


def tensor_to_image(tensor):
    #tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image


def get_video_img(tensor):
    outputs = []
    for x in tensor:
        x=tensor_to_image(x)
        outputs.append(x)
    return outputs

def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def narry_list(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        modified_value = phi2narry(value)
        list_in[i] = modified_value
    return list_in


def get_local_path(file_path, model_path):
    path = os.path.join(file_path, "models", "diffusers", model_path)
    model_path = os.path.normpath(path)
    if sys.platform == 'win32':
        model_path = model_path.replace('\\', "/")
    return model_path


def get_instance_path(path):
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = instance_path.replace('\\', "/")
    return instance_path


def get_lora(lora):
    lora_path = folder_paths.get_full_path("loras", lora)
    lora_path = get_instance_path(lora_path)
    return lora_path


def instance_path(path, repo):
    if repo == "":
        if path == "none":
            repo = "none"
        else:
            model_path = get_local_path(file_path, path)
            repo = get_instance_path(model_path)
    return repo


class Video2Video:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "local_model": (paths,),
                "repo_id": ("STRING", {"default": "runwayml/stable-diffusion-v1-5"}),
                "input_video": ("STRING", {"forceInput": True}),
                "lora_list": (["none"] + folder_paths.get_filename_list("loras"),),
                "prompt": ("STRING", {"multiline": True, "default": "Claymation, a man is giving a talk"}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1, "round": 0.01}),
                "diffusion_steps": ("INT", {"default": 4, "min": 1, "max": 1000}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 1000}),
                "noise_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "seed": ("INT", {"default": 2, "min": 0, "max": 0xffffffffffffffff})
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "main_process"
    CATEGORY = "StreamV2V"

    def main_process(self,
                     local_model,repo_id,
                     input_video, lora_list,
                     prompt,
                     scale,
                     guidance_scale,
                     diffusion_steps,
                     num_inference_steps,
                     noise_strength,
                     seed,
                     ):

        # if not os.path.exists(output_dir):
        #     os.mkdir(output_dir)
        model_id=instance_path(local_model, repo_id)
        if model_id=="none":
            raise "need local model or repo_id"
        output_dir = os.path.join(file_path, "output")
        video_info = read_video(input_video)  # input path to the file name
        video = video_info[0] / 255
        fps = video_info[2]["video_fps"]
        height = int(video.shape[1] * scale)
        width = int(video.shape[2] * scale)

        init_step = int(50 * (1 - noise_strength))
        interval = int(50 * noise_strength) // diffusion_steps
        t_index_list = [init_step + i * interval for i in range(diffusion_steps)]

        stream = StreamV2VWrapper(
            model_id_or_path=model_id,
            mode="img2img",
            t_index_list=t_index_list,
            frame_buffer_size=1,
            width=width,
            height=height,
            warmup=10,
            acceleration="xformers",
            do_add_noise=True,
            output_type="pt",
            enable_similar_image_filter=False,
            similar_image_filter_threshold=0.98,
            use_denoising_batch=True,
            use_cached_attn=True,
            use_feature_injection=True,
            feature_injection_strength=0.8,
            feature_similarity_threshold=0.98,
            cache_interval=4,
            cache_maxframes=1,
            use_tome_cache=True,
            seed=seed,
        )
        stream.prepare(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        if lora_list=="none":
            pass
        # Specify LORAs
        else:
            pixelart = "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors$"
            lowpoly = "low_poly.safetensors$"
            claymation = "Claymation.safetensors$"
            crayons = "doodle.safetensors$"
            sketch = "Sketch_offcolor.safetensors$"
            oilpainting = "bichu-v0612.safetensors$"
            if lora_list == "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors" or re.search(pixelart, lora_list):
                lora_name = "pixelart"
            elif lora_list == "low_poly.safetensors" or re.search(lowpoly, lora_list):
                lora_name = "lowpoly"
            elif lora_list == "Claymation.safetensors" or re.search(claymation, lora_list):
                lora_name = "claymation"
            elif lora_list == "doodle.safetensors" or re.search(crayons, lora_list):
                lora_name = "crayons"
            elif lora_list == "Sketch_offcolor.safetensors" or re.search(sketch, lora_list):
                lora_name = "sketch"
            elif lora_list == "bichu-v0612.safetensors" or re.search(oilpainting, lora_list):
                lora_name = "oilpainting"
            else:
                lora_name = lora_list.rsplit(".", 1)[0]
                lora_name = lora_name.split("\\", 1)[-1]
            lora_path = get_lora(lora_list)
            stream.stream.load_lora(lora_path, adapter_name=lora_name)
            stream.stream.pipe.set_adapters(adapter_names=["lcm", lora_name], adapter_weights=[1.0, 1.0])
            print(f"Use LORA: {lora_name} in {lora_path}")

        video_result = torch.zeros(video.shape[0], height, width, 3)

        for _ in range(stream.batch_size):
            stream(image=video[0].permute(2, 0, 1))

        inference_time = []
        for i in tqdm(range(video.shape[0])):
            iteration_start_time = time.time()
            output_image = stream(video[i].permute(2, 0, 1))
            video_result[i] = output_image.permute(1, 2, 0)
            iteration_end_time = time.time()
            inference_time.append(iteration_end_time - iteration_start_time)
        print(f'Avg time: {sum(inference_time[20:]) / len(inference_time[20:])}')

        gen = get_video_img(video_result)
        gen = narry_list(gen)  # 列表排序
        images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
        return (images,)




class Text2Video:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "local_model": (paths,),
                "repo_id": ("STRING", {"default": "runwayml/stable-diffusion-v1-5"}),
                "vae_id": ("STRING", {"default": "madebyollin/taesd"}),
                "prompt": ("STRING", {"multiline": True, "default": "Claymation, a man is giving a talk"}),
                "width": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64, }),
                "height": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
                "seed": ("INT", {"default": 2, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "txt2img_process"
    CATEGORY = "StreamV2V"

    def txt2img_process(self,local_model,
                        repo_id,
                        vae_id,
                        prompt,
                        width,
                        height,
                        seed,
                        ):
        model_id = instance_path(local_model, repo_id)
        if model_id == "none":
            raise "need local model or repo_id"
        stream = StreamV2VWrapper(
            model_id_or_path=model_id,
            t_index_list=[0, 16, 32, 45],
            lora_dict=None,
            output_type="pil",
            mode="txt2img",
            lcm_lora_id=None,
            vae_id=vae_id,
            device="cuda",
            dtype=torch.float16,
            frame_buffer_size=1,
            warmup=10,
            width=width,
            height=height,
            acceleration="xformers",
            do_add_noise=True,
            use_denoising_batch=False,
            use_cached_attn=True,
            use_feature_injection=True,
            feature_injection_strength=0.8,
            feature_similarity_threshold=0.98,
            cache_interval=1,
            cache_maxframes=4,
            use_tome_cache=True,
            cfg_type="none",
            seed=seed,
            use_safety_checker= False
         )

        image=phi2narry(stream.txt2img(prompt))
        return (image,)



class WebCam2Video:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "local_model": (paths,),
                "repo_id": ("STRING", {"default": "runwayml/stable-diffusion-v1-5"}),
                "prompt": ("STRING", {"multiline": True, "default": " a man is giving a talk"}),
                "lora_list": (["none"]+folder_paths.get_filename_list("loras"),),
                "width": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64, }),
                "height": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
                "seed": ("INT", {"default": 2, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images",)
    FUNCTION = "webcam2img_process"
    CATEGORY = "StreamV2V"

    def webcam2img_process(self,
                        image,
                        local_model,
                        repo_id,
                        prompt,
                        lora_list,
                        width,
                        height,
                        seed,
                        ):
        model_id = instance_path(local_model, repo_id)
        if model_id == "none":
            raise "need local model or repo_id"
        stream = StreamV2VWrapper(
            model_id_or_path=model_id,
            mode="img2img",
            t_index_list=[30, 35, 40, 45],#
            frame_buffer_size=1,
            width=width,
            height=height,
            warmup=10,
            dtype=torch.float16,
            device="cuda",
            acceleration="xformers",
            do_add_noise=True,
            output_type="pil",
            use_denoising_batch=True,
            use_cached_attn=True,
            use_feature_injection=True,
            feature_injection_strength=0.8,
            feature_similarity_threshold=0.98,
            cache_interval=4,
            cache_maxframes=1,
            use_tome_cache=True,
            seed=seed,

        )

        if lora_list != "none":
            pixelart="PixelArtRedmond15V-PixelArt-PIXARFK.safetensors$"
            lowpoly="low_poly.safetensors$"
            claymation="Claymation.safetensors$"
            crayons="doodle.safetensors$"
            sketch="Sketch_offcolor.safetensors$"
            oilpainting="bichu-v0612.safetensors"
            if  lora_list == "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors" or re.search(pixelart,lora_list):
                lora_name = "pixelart"
            elif lora_list == "low_poly.safetensors" or re.search(lowpoly,lora_list):
                lora_name = "lowpoly"
            elif lora_list == "Claymation.safetensors" or re.search(claymation,lora_list):
                lora_name = "claymation"
            elif lora_list == "doodle.safetensors"or re.search(crayons,lora_list):
                lora_name = "crayons"
            elif lora_list == "Sketch_offcolor.safetensors"or re.search(sketch,lora_list):
                lora_name = "sketch"
            elif lora_list == "bichu-v0612.safetensors"or re.search(oilpainting,lora_list):
                lora_name = "oilpainting"
            else:
                lora_name = lora_list.rsplit(".",1)[0]
                lora_name =lora_name.split("\\",1)[-1]
            lora_path = get_lora(lora_list)
            stream.stream.load_lora(lora_path, adapter_name=lora_name)
            stream.stream.pipe.set_adapters(["lcm", lora_name], adapter_weights=[1.0, 1.0])
            print(f"Use LORA: {lora_name} in {lora_path}")


        image=tensor_to_image(image)
        image_tensor = stream.preprocess_image(image)
        output_image = stream(image=image_tensor, prompt=prompt)
        print(type(output_image))
        images=phi2narry(output_image)
        return images


class Load_Video:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        video_files = [f for f in os.listdir(input_path) if
                       os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["mp4", "webm", "mkv",
                                                                                            "avi"]]
        return {"required": {
            "video": (video_files,)
        }}

    # OUTPUT_NODE = False
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("input_video",)
    FUNCTION = "load_video"
    CATEGORY = "StreamV2V"

    def load_video(self, video):
        input_video = os.path.join(input_path, video)
        input_video = get_instance_path(input_video)
        return (input_video,)


NODE_CLASS_MAPPINGS = {
    "Video2Video": Video2Video,
    "Text2Video": Text2Video,
    "WebCam2Video": WebCam2Video,
    "Load_Video": Load_Video,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Video2Video": "Video2Video",
    "Text2Video": "Text2Video",
    "WebCam2Video":"WebCam2Video",
}
