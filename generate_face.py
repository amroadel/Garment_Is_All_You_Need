import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import argparse

def generate_face(face_pipe, positive_prompt, negative_prompt, num_inference_steps=25, num_samples=2):
  """
    Generate face images using the given prompts. 
    Args:
      face_pipe: Face pipeline object
      positive_prompt: Positive prompt for the face generation
      negative_prompt: Negative prompt for the face generation
      num_inference_steps: Number of inference steps
      num_samples: Number of samples to generate
    Returns:
        images: Generated face images
    """
  torch.cuda.empty_cache()
  generator = torch.Generator(device="cpu").manual_seed(42)
  # torch.cuda.manual_seed_all(42)
  images = face_pipe(
              prompt=[positive_prompt]*num_samples,
              negative_prompt=[negative_prompt]*num_samples,
              generator=generator,
              num_inference_steps=num_inference_steps,
              seed = 42,
              num_samples=num_samples,
              guidance_scale=7,
        ).images
  return images
