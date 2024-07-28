# ComfyUI_Streamv2v_Plus
You can using Streamv2v/StreamDiffusion in comfyui

Streamv2v  From: [Streamv2v](https://github.com/Jeff-LiangF/streamv2v)   
StreamDiffusion  From: [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion)

My ComfyUI node list：
-----
1、ParlerTTS node:[ComfyUI_ParlerTTS](https://github.com/smthemex/ComfyUI_ParlerTTS)     
2、Llama3_8B node:[ComfyUI_Llama3_8B](https://github.com/smthemex/ComfyUI_Llama3_8B)      
3、HiDiffusion node：[ComfyUI_HiDiffusion_Pro](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro)   
4、ID_Animator node： [ComfyUI_ID_Animator](https://github.com/smthemex/ComfyUI_ID_Animator)       
5、StoryDiffusion node：[ComfyUI_StoryDiffusion](https://github.com/smthemex/ComfyUI_StoryDiffusion)  
6、Pops node：[ComfyUI_Pops](https://github.com/smthemex/ComfyUI_Pops)   
7、stable-audio-open-1.0 node ：[ComfyUI_StableAudio_Open](https://github.com/smthemex/ComfyUI_StableAudio_Open)        
8、GLM4 node：[ComfyUI_ChatGLM_API](https://github.com/smthemex/ComfyUI_ChatGLM_API)   
9、CustomNet node：[ComfyUI_CustomNet](https://github.com/smthemex/ComfyUI_CustomNet)           
10、Pipeline_Tool node :[ComfyUI_Pipeline_Tool](https://github.com/smthemex/ComfyUI_Pipeline_Tool)    
11、Pic2Story node :[ComfyUI_Pic2Story](https://github.com/smthemex/ComfyUI_Pic2Story)   
12、PBR_Maker node:[ComfyUI_PBR_Maker](https://github.com/smthemex/ComfyUI_PBR_Maker)      
13、ComfyUI_Streamv2v_Plus node:[ComfyUI_Streamv2v_Plus](https://github.com/smthemex/ComfyUI_Streamv2v_Plus)   
14、ComfyUI_MS_Diffusion node:[ComfyUI_MS_Diffusion](https://github.com/smthemex/ComfyUI_MS_Diffusion)   
15、ComfyUI_AnyDoor node: [ComfyUI_AnyDoor](https://github.com/smthemex/ComfyUI_AnyDoor)  
16、ComfyUI_Stable_Makeup node: [ComfyUI_Stable_Makeup](https://github.com/smthemex/ComfyUI_Stable_Makeup)  
17、ComfyUI_EchoMimic node:  [ComfyUI_EchoMimic](https://github.com/smthemex/ComfyUI_EchoMimic)   
18、ComfyUI_FollowYourEmoji node: [ComfyUI_FollowYourEmoji](https://github.com/smthemex/ComfyUI_FollowYourEmoji)   

1.Installation
-----
  In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_Streamv2v_Plus.git
```  
  
2.requirements  
----
```
pip install -r requirements.txt

```

（如果peft报错）if get error about ：PEFT backend is required for this method  
请按以下代码更新 PEFT和transformers 
pip install  -U PEFT  transformers 

tensorrt still has bug，need module below：       
tensorrt 暂时无法使用，以下是需求库，可以不装       
polygraphy      
onnx_graphsurgeon   
tensorrt   
cuda-python   


缺啥装啥。。。  
If the module is missing, , pip install  missing module.       

3 Need  model 
---
3.1 base model   
SDXL or sd1.5 pure单体模型，或者社区模型，部分模型没有vae会导致报错，可以在vae填入常规的vae repo，该地址栏默认是空，也就是使用社区模型内置的vae和encoder。  首次使用会下载config文件，注意连外网。。。  
SDXL or sd1.5 pure monolithic model, or community model,  

3.2 vae   
====注意，使用SDXL或者XL turbo时，vae必须使用  madebyollin/sdxl-vae-fp16-fix   
SD1.5可以不使用vae或者使用madebyollin/taesd   

3.3 lcm lora   lcm loras是必需的。   
latent-consistency/lcm-lora-sdv1-5  SD1.5  
latent-consistency/lcm-lora-sdxl    SDXL or XL turbo   

3.4 style lora 
choice which you like   选一个你喜欢的风格lora，注意匹配底模。   
when you changge a style lora，twigger word need change  so. 改变风格lora时，关键词需要跟着变。   

4 Function Description
--
4.1 txt2img   文生图   
4.2 webcam2img/img2img   摄像头生图  
4.3 video2video  视频转绘  

5.example 示例
----
sd1.5 txt2im/img2img/webcam2img/video2video  选择菜单使用不同的功能。  
![](https://github.com/smthemex/ComfyUI_Streamv2v_Plus/blob/main/example/sd15.png)

SDXL（turbo 1 step，XL 4 step） SDXL示例，选择菜单使用不同的功能。
![](https://github.com/smthemex/ComfyUI_Streamv2v_Plus/blob/main/example/sdxl.png)

cam2video or cam2img need more vr  摄像头生图得看配置，低的跑的慢。
![](https://github.com/smthemex/ComfyUI_Streamv2v_Plus/blob/main/example/cam.jpg)

6 Citation
------
streamdiffusion
``` python  
@article{kodaira2023streamdiffusion,
      title={StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation},
      author={Akio Kodaira and Chenfeng Xu and Toshiki Hazama and Takanori Yoshimoto and Kohei Ohno and Shogo Mitsuhori and Soichi Sugano and Hanying Cho and Zhijian Liu and Kurt Keutzer},
      year={2023},
      eprint={2312.12491},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
StreamingV2V
``` python  
@article{liang2024looking,
  title={Looking Backward: Streaming Video-to-Video Translation with Feature Banks},
  author={Liang, Feng and Kodaira, Akio and Xu, Chenfeng and Tomizuka, Masayoshi and Keutzer, Kurt and Marculescu, Diana},
  journal={arXiv preprint arXiv:2405.15757},
  year={2024}
}
```





