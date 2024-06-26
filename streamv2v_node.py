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
    paths = ["none", ]


def tensor_to_image(tensor):
    # tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image


def get_video_img(tensor):
    outputs = []
    for x in tensor:
        x = tensor_to_image(x)
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

class Load_Stream:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        video_files = [f for f in os.listdir(input_path) if
                       os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["mp4", "webm", "mkv",
                                                                                            "avi"]]
        return {
            "required": {
                "video": (["none"]+video_files,),
                "local_model": (paths,),
                "repo_id": ("STRING", {"default": "runwayml/stable-diffusion-v1-5"}),
                "vae_id": ("STRING", {"default": "madebyollin/taesd"}),
                "lora": (["none"] + folder_paths.get_filename_list("loras"),),
                "lora_scale": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.1}),
                "trigger_word": ("STRING", {"default": "best quality"}),
                "sampler_type": (["txt2img", "vdieo2vdieo","WebCam2Video"],),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "width": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64, }),
                "height": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1, "round": 0.01}),
                "diffusion_steps": ("INT", {"default": 4, "min": 1, "max": 1000}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 1000}),
                "noise_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "seed": ("INT", {"default": 2, "min": 0, "max": 0xffffffffffffffff})}}

    RETURN_TYPES = ("MODEL","CONDITIONING","STRING")
    RETURN_NAMES = ("stream","video","info",)
    FUNCTION = "load_stream"
    CATEGORY = "StreamV2V_Plus"

    def load_stream(self, video,
                     local_model, repo_id,
                     vae_id,
                     lora,
                     lora_scale,
                     trigger_word,sampler_type,
                     scale,width,height,
                     guidance_scale,
                     diffusion_steps,
                     num_inference_steps,
                     noise_strength,
                     seed,
                     ):

        model_id = instance_path(local_model, repo_id)
        if model_id == "none":
            raise "need local model or repo_id"
        output_dir = os.path.join(file_path, "output")
        fps=25
        height_v=512
        width_v=512

        if sampler_type=="vdieo2vdieo":
            if video == "none":
                raise "need video input"
            input_video = get_instance_path(os.path.join(input_path, video))
            # if not os.path.exists(output_dir):
            #     os.mkdir(output_dir)
            video_info = read_video(input_video)  # input path to the file name
            video = video_info[0] / 255
            fps = video_info[2]["video_fps"]
            height_v = int(video.shape[1] * scale)
            width_v = int(video.shape[2] * scale)

            init_step = int(50 * (1 - noise_strength))
            interval = int(50 * noise_strength) // diffusion_steps
            t_index_list = [init_step + i * interval for i in range(diffusion_steps)]

            stream = StreamV2VWrapper(
                model_id_or_path=model_id,
                mode="img2img",
                t_index_list=t_index_list,
                frame_buffer_size=1,
                width=width_v,
                height=height_v,
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

            if lora == "none":
                raise "need style lora"
            lora_path = get_lora(lora)
            stream.stream.load_lora(lora_path, adapter_name=trigger_word)
            stream.stream.pipe.set_adapters(adapter_names=["lcm", trigger_word], adapter_weights=[1.0, lora_scale])
            print(f"Use LORA: {trigger_word} in {lora}")

        elif sampler_type=="txt2img":
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
                use_safety_checker=False
            )
        else:
            stream = StreamV2VWrapper(
                model_id_or_path=model_id,
                mode="img2img",
                t_index_list=[30, 35, 40, 45],
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
            if lora == "none":
                raise "need style lora"
            lora_path = get_lora(lora)
            stream.stream.load_lora(lora_path, adapter_name=trigger_word)
            stream.stream.pipe.set_adapters(adapter_names=["lcm", trigger_word], adapter_weights=[1.0, lora_scale])
            print(f"Use LORA: {trigger_word} in {lora}")

        info=";".join([str(fps),str(height_v), str(width_v),str(guidance_scale),str(num_inference_steps),trigger_word])
        return (stream,video,info)


class Video2Video:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stream": ("MODEL",),
                "video":("CONDITIONING",),
                "info":("STRING", {"forceInput": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": "Claymation, a man is giving a talk"}),
            }
        }

    RETURN_TYPES = ("IMAGE","FLOAT")
    RETURN_NAMES = ("image","frame_rate")
    FUNCTION = "main_process"
    CATEGORY = "StreamV2V_Plus"


    def main_process(self, stream,video,info,prompt,):

        fps,height, width,guidance_scale,num_inference_steps,trigger_word=info.split(";")
        fps=float(fps)
        height=int(height)
        width=int(width)
        guidance_scale=float(guidance_scale)
        num_inference_steps=int(num_inference_steps)

        prompt = prompt + " " + trigger_word + "style"
        stream.prepare(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

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
        frame_rate = float(fps)
        return (images, frame_rate)


class Text2IMG:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stream": ("MODEL",),
                "info": ("STRING", {"forceInput": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": "Claymation, a man is giving a talk"}),

            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "txt2img_process"
    CATEGORY = "StreamV2V_Plus"

    def txt2img_process(self, stream,info,prompt,):
        fps, height, width, guidance_scale, num_inference_steps, trigger_word = info.split(";")
        prompt = prompt + " " + trigger_word + "style"
        image = phi2narry(stream.txt2img(prompt))
        return (image,)


class WebCam2Video:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stream": ("MODEL",),
                "image": ("IMAGE",),
                "info": ("STRING", {"forceInput": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": "Claymation, a man is giving a talk"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "webcam2img_process"
    CATEGORY = "StreamV2V_Plus"

    def webcam2img_process(self,stream,image,info,prompt):
        fps, height, width, guidance_scale, num_inference_steps, trigger_word = info.split(";")
        prompt = prompt + " " + trigger_word + "style"
        image = tensor_to_image(image)
        output_image = stream(image=image, prompt=prompt)
        print(type(output_image))
        images = phi2narry(output_image)
        return (images,)


NODE_CLASS_MAPPINGS = {
    "Load_Stream":Load_Stream,
    "Video2Video": Video2Video,
    "Text2IMG": Text2IMG,
    "WebCam2Video": WebCam2Video
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load_Stream":"Load_Stream",
    "Video2Video": "Video2Video",
    "Text2IMG": "Text2IMG",
    "WebCam2Video": "WebCam2Video"
}
