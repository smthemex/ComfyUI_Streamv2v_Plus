# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import time
from torchvision.io import read_video
from tqdm import tqdm
from .utils.wrapper import StreamV2VWrapper
import torch
import os
from PIL import Image
import numpy as np
import sys
import folder_paths
from comfy.utils import common_upscale
import cv2
import diffusers

dif_version = str(diffusers.__version__)
dif_version_int = int(dif_version.split(".")[1])
from diffusers import (DDIMScheduler,
                       KDPM2AncestralDiscreteScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler,
                       DPMSolverSinglestepScheduler, AutoencoderTiny,
                       EulerDiscreteScheduler, HeunDiscreteScheduler, KDPM2DiscreteScheduler,
                       EulerAncestralDiscreteScheduler, UniPCMultistepScheduler, DiffusionPipeline,
                       DDPMScheduler, LCMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline, AutoencoderKL, )

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
path_dir = os.path.dirname(CURRENT_DIR)
file_path = os.path.dirname(path_dir)
BIGMAX = (2 ** 53 - 1)
input_path = folder_paths.get_input_directory()


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


def nomarl_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img_pil = tensor_to_image(samples)
    return img_pil


def tensor_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    return samples


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


scheduler_list = ["LCM", "DDIM",
                  "Euler",
                  "Euler a",
                  "DDPM",
                  "DPM++ 2M",
                  "DPM++ 2M Karras",
                  "DPM++ 2M SDE",
                  "DPM++ 2M SDE Karras",
                  "DPM++ SDE",
                  "DPM++ SDE Karras",
                  "DPM2",
                  "DPM2 Karras",
                  "DPM2 a",
                  "DPM2 a Karras",
                  "Heun",
                  "LMS",
                  "LMS Karras",
                  "UniPC",
                  ]


def get_sheduler(name):
    scheduler = False
    if name == "Euler":
        scheduler = EulerDiscreteScheduler()
    elif name == "Euler a":
        scheduler = EulerAncestralDiscreteScheduler()
    elif name == "DDIM":
        scheduler = DDIMScheduler()
    elif name == "DDPM":
        scheduler = DDPMScheduler()
    elif name == "DPM++ 2M":
        scheduler = DPMSolverMultistepScheduler()
    elif name == "DPM++ 2M Karras":
        scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True)
    elif name == "DPM++ 2M SDE":
        scheduler = DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++")
    elif name == "DPM++ 2M SDE Karras":
        scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    elif name == "DPM++ SDE":
        scheduler = DPMSolverSinglestepScheduler()
    elif name == "DPM++ SDE Karras":
        scheduler = DPMSolverSinglestepScheduler(use_karras_sigmas=True)
    elif name == "DPM2":
        scheduler = KDPM2DiscreteScheduler()
    elif name == "DPM2 Karras":
        scheduler = KDPM2DiscreteScheduler(use_karras_sigmas=True)
    elif name == "DPM2 a":
        scheduler = KDPM2AncestralDiscreteScheduler()
    elif name == "DPM2 a Karras":
        scheduler = KDPM2AncestralDiscreteScheduler(use_karras_sigmas=True)
    elif name == "Heun":
        scheduler = HeunDiscreteScheduler()
    elif name == "LCM":
        scheduler = LCMScheduler()
    elif name == "LMS":
        scheduler = LMSDiscreteScheduler()
    elif name == "LMS Karras":
        scheduler = LMSDiscreteScheduler(use_karras_sigmas=True)
    elif name == "UniPC":
        scheduler = UniPCMultistepScheduler()
    return scheduler


class Stream_Model_Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "scheduler": (scheduler_list,),
                "use_sdxl": ("BOOLEAN", {"default": False},),
                "vae_id": ("STRING", {"default": ""}),
                "lcm_lora": (["none"] + folder_paths.get_filename_list("loras"),),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("pipe", "info")
    FUNCTION = "main"
    CATEGORY = "StreamV2V_Plus"

    def main(self, ckpt_name, scheduler, use_sdxl, vae_id, lcm_lora, ):
        # 加载pipe
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        scheduler_used = get_sheduler(scheduler)

        if use_sdxl:
            original_config_file = os.path.join(CURRENT_DIR, "models", "sd_xl_base.yaml")
            if dif_version_int >= 28:
                pipe = StableDiffusionXLPipeline.from_single_file(
                    ckpt_path, original_config=original_config_file, torch_dtype=torch.float16).to("cuda")
            else:
                pipe = StableDiffusionXLPipeline.from_single_file(
                    ckpt_path, original_config_file=original_config_file, torch_dtype=torch.float16).to("cuda")
            info = "sdxl"
        else:
            original_config_file = os.path.join(folder_paths.models_dir, "configs", "v1-inference.yaml")
            if dif_version_int >= 28:
                pipe = StableDiffusionPipeline.from_single_file(
                    ckpt_path, original_config=original_config_file, torch_dtype=torch.float16).to("cuda")
            else:
                pipe = StableDiffusionPipeline.from_single_file(
                    ckpt_path, original_config_file=original_config_file, torch_dtype=torch.float16).to("cuda")
            info = "sd15"
        if vae_id != "":
            pipe.vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16).to("cuda")
        pipe.scheduler = scheduler_used.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_vae_tiling()
        lcm_lora_path = get_lora(lcm_lora)
        pipe.load_lora_weights(lcm_lora_path, adapter_name="lcm", )
        info = ";".join([info, ckpt_name])
        return (pipe, info,)


class Stream_Lora_Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("MODEL",),
                "info": ("STRING", {"forceInput": True, "default": ""}),
                "lora": (["none"] + folder_paths.get_filename_list("loras"),),
                "lora_scale": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.1}),
                "trigger_word": ("STRING", {"default": "best quality"}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING",)
    RETURN_NAMES = ("pipe", "info",)
    FUNCTION = "main"
    CATEGORY = "StreamV2V_Plus"

    def main(self, pipe, info, lora, lora_scale, trigger_word):
        info, ckpt_name = info.split(";")
        lora_path = get_lora(lora)
        if lora == "none":
            raise "you need a style lora"
        pipe.load_lora_weights(lora_path, adapter_name=trigger_word, )

        pipe.set_adapters(adapter_names=["lcm", trigger_word], adapter_weights=[1.0, lora_scale])
        pipe.fuse_lora()
        print(f"Use style lora: {trigger_word} in {lora}")
        info = ";".join([info, trigger_word, ckpt_name])
        return (pipe, info,)


class Stream_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        video_files = [f for f in os.listdir(input_path) if
                       os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["mp4", "webm", "mkv",
                                                                                            "avi"]]
        return {
            "required": {
                "pipe": ("MODEL",),
                "info": ("STRING", {"forceInput": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": "Claymation, a man is giving a talk"}),
                "video": (["none"] + video_files,),
                "sampler_type": (["txt2img", "vdieo2vdieo", "WebCam2Video"],),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1, "round": 0.01}),
                "diffusion_steps": ("INT", {"default": 4, "min": 1, "max": 1000}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 1000}),
                "noise_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "seed": ("INT", {"default": 2, "min": 0, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64, }),
                "height": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
                "acceleration": (["xformers", "tensorrt", ],),

            },
            "optional": {"image": ("IMAGE",),
                         }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT",)
    RETURN_NAMES = ("image", "audio", "fps",)
    FUNCTION = "main_stream"
    CATEGORY = "StreamV2V_Plus"

    def main_stream(self, pipe, info, prompt, video, sampler_type, guidance_scale, diffusion_steps, num_inference_steps,
                    noise_strength, seed, width, height, acceleration, **kwargs):
        model_type, trigger_word, ckpt_name = info.split(";")
        if model_type == "sdxl":
            using_sdxl = True
        else:
            using_sdxl = False
        image = kwargs.get("image")
        if sampler_type == "vdieo2vdieo":
            if video == "none":
                raise "need video input"
            input_video = get_instance_path(os.path.join(input_path, video))
            video_info = read_video(
                input_video)  # input path to the file name  "THWC") -> tuple[Tensor, Tensor, dict[str, Any]]
            video = video_info[0] / 255  # Tensor THWC
            fps = video_info[2]["video_fps"]  # dict[str, Any]

            height_v = int(video.shape[1])
            width_v = int(video.shape[2])
            waveform = video_info[1]
            if waveform.size(1) == 0:
                audio = None
                print("no audio in input video!")
            else:
                sample_rate = video_info[2]["audio_fps"]
                print("get audio from video")
                audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
            init_step = int(50 * (1 - noise_strength))
            interval = int(50 * noise_strength) // diffusion_steps
            t_index_list = [init_step + i * interval for i in range(diffusion_steps)]

            stream = StreamV2VWrapper(
                pipe,
                mode="img2img",
                t_index_list=t_index_list,
                frame_buffer_size=1,
                width=width_v,
                height=height_v,
                warmup=10,
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
                using_sdxl=using_sdxl
            )

            prompt = prompt + " " + trigger_word + "style"
            stream.stream.prepare(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

            video_result = torch.zeros(video.shape[0], height_v, width_v, 3)

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
            gen = narry_list(gen)
            images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height_v, width_v, 3)))))
            return (images, audio, fps)

        elif sampler_type == "txt2img":
            stream = StreamV2VWrapper(
                pipe,
                t_index_list=[0, 16, 32, 45],
                output_type="pil",
                mode="txt2img",
                device="cuda",
                dtype=torch.float16,
                frame_buffer_size=1,
                warmup=10,
                width=width,
                height=height,
                do_add_noise=True,
                use_denoising_batch=False,
                use_cached_attn=True,
                use_feature_injection=True,
                feature_injection_strength=0.8,
                feature_similarity_threshold=0.98,
                cache_interval=1,
                cache_maxframes=4,
                use_tome_cache=True,
                seed=seed,
                use_safety_checker=False,
                using_sdxl=using_sdxl
            )
            prompt = prompt + " " + trigger_word + "style"
            stream.stream.prepare(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            output_img = stream(prompt=prompt)
            images = phi2narry(output_img)
            return (images,)

        else:
            stream = StreamV2VWrapper(
                pipe,
                mode="img2img",
                t_index_list=[32, 45],
                frame_buffer_size=1,
                width=width,
                height=height,
                warmup=10,
                dtype=torch.float16,
                device="cuda",
                do_add_noise=True,
                output_type="pil",
                use_denoising_batch=True,
                use_cached_attn=False,
                use_feature_injection=True,
                feature_injection_strength=0.8,
                feature_similarity_threshold=0.98,
                cache_interval=4,
                cache_maxframes=1,
                use_tome_cache=True,
                seed=seed,
                acceleration=acceleration,
                using_sdxl=using_sdxl,
                model_id_or_path=ckpt_name
            )
            prompt = prompt + " " + trigger_word + "style"
            stream.stream.prepare(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )  # 预处理prompt
            image = nomarl_upscale(image, width, height)
            for _ in range(2):  # 预处理prompt+img
                stream(image)
            images = stream(image)
            images = phi2narry(images)
            return (images,)


NODE_CLASS_MAPPINGS = {
    "Stream_Model_Loader": Stream_Model_Loader,
    "Stream_Lora_Loader": Stream_Lora_Loader,
    "Stream_Sampler": Stream_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load_Stream": "Load_Stream",
    "Stream_Lora_Loader": "Stream_Lora_Loader",
    "Stream_Sampler": "Stream_Sampler",
}
