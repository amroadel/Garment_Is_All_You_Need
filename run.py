import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import cv2
from PIL import Image
import numpy as np
import argparse

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image, DDIMScheduler, AutoencoderKL
from huggingface_hub import hf_hub_download
# from transformers import AutoFeatureExtractor

from generate_body import generate_body
from generate_face import generate_face
from utils import save_data, image_grid, pad_face, load_images_from_file_or_folder, get_propmpts

from clip_interrogator import Config, Interrogator

caption_model_name = 'blip-base'
clip_model_name = "ViT-H-14/laion2b_s32b_b79k"
config = Config()
config.clip_model_name = clip_model_name
config.caption_model_name = caption_model_name
ci = Interrogator(config)

def image_to_prompt(image, mode):
    """
    Get a prompt from an image.
    """
    ci.config.chunk_size = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    ci.config.flavor_intermediate_count = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image)
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    elif mode == 'fast':
        return ci.interrogate_fast(image)
    elif mode == 'negative':
        return ci.interrogate_negative(image)
    
def load_pipeline(base_model_path, vae_model_path, device, USE_XL=True, from_file=False):
    """
    Load a diffusers pipeline.
    Args:
        base_model_path: Path to the base diffusion model
        vae_model_path: Path to the VAE model
        device: Device to use (e.g., cuda or cpu)
        USE_XL: Use the XL diffusion pipeline
        """
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    # safety_model_id = "CompVis/stable-diffusion-safety-checker"
    # safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
    # safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    pipeline_class = StableDiffusionXLPipeline if USE_XL else StableDiffusionPipeline
    if from_file: 
        pipe = pipeline_class.from_single_file(
            base_model_path,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            # feature_extractor=safety_feature_extractor,
            # safety_checker=safety_checker
        ).to(device)
    else:
        pipe = pipeline_class.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            # feature_extractor=safety_feature_extractor,
            # safety_checker=safety_checker
        ).to(device)
        
    pipe.enable_attention_slicing()
    return pipe

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate images using the given prompts.")
    parser.add_argument("--garment_path", type=str, required=True, help="Path to the garment image")
    parser.add_argument("--prompt_config", type=str, default="prompt_config.json", help="Path to the prompt configuration json file")
    parser.add_argument("--face_images_path", type=str, default=None, help="Path to existing face images (Optional)")
    parser.add_argument("--garment_type", type=str, choices=['top', 'bottom', 'dress'], default='top', help="Type of the garment")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of samples to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., cuda or cpu)")

    args = parser.parse_args()

    # Load garment image
    garment_image = Image.open(args.garment_path)
    garment_description = image_to_prompt(garment_image, mode='best')

    negative_prompt, face_prompt, body_prompt = get_propmpts(args.prompt_config, garment_description, garment_type=args.garment_type)

    if args.face_images_path is None:
        # Generate face images
        base_model_path="https://huggingface.co/KamCastle/kcup/realismEngineSDXL_v30VAE.safetensors"
        vae_model_path="stabilityai/sdxl-vae"
        face_pipe = load_pipeline(base_model_path, vae_model_path, args.device, USE_XL=True, from_file=True)
        face_images = generate_face(face_pipe, face_prompt, negative_prompt, args.num_inference_steps, args.num_samples)
        del face_pipe
        torch.cuda.empty_cache()
    else:
        # Load the face images
        face_images = load_images_from_file_or_folder(args.face_images_path)

    # Generate body images
    base_model_path = "SG161222/Realistic_Vision_V5.0_noVAE"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    body_pipe = load_pipeline(base_model_path, vae_model_path, args.device, USE_XL=False, from_file=False)
    ip_ckpt = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid-plusv2_sd15.bin", repo_type="model")
    body_images = generate_body(face_images, body_prompt, negative_prompt, face_strength=1.3, likeness_strength=1, 
                                ip_ckpt=ip_ckpt, body_pipe=body_pipe, USE_XL=False, preserve_face_structure=True, num_inference_steps=args.num_inference_steps, 
                                num_samples=args.num_samples, device=args.device)
    del body_images
    torch.cuda.empty_cache()
    
    # Save the generated images
    save_data(body_images, args.output_folder, args.image_name)