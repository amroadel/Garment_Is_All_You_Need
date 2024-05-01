import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import cv2
from PIL import Image
import numpy as np
from utils import pad_face

# from transformers import AutoFeatureExtractor
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID, IPAdapterFaceIDPlus, IPAdapterFaceIDXL, IPAdapterFaceIDPlusXL
from insightface.app import FaceAnalysis
from insightface.utils import face_align


def get_face_embeddings(face_images, preserve_face_structure=True):
    """
    Get face embeddings from a list of face images.
    Args:
        face_images: List of face images (paths, PIL images, or OpenCV images)
        preserve_face_structure: Preserve face structure in the final image
    """
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    cv2.setNumThreads(1)

    faceid_all_embeds = []
    face_image = None
    first_iteration = True

    # Convert paths or PIL images to OpenCV images
    for image in face_images:
        if isinstance(image, str):  # Path to image
            face = Image.open(image)
            face = pad_face(face)
            face = np.array(face)
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        elif isinstance(image, Image.Image):  # PIL image
            face = pad_face(face)
            face = np.array(face)
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Unsupported image type. Please provide paths to images or PIL images.")

        faces = app.get(face)
        faceid_embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        faceid_all_embeds.append(faceid_embed)

        if first_iteration and preserve_face_structure:
            face_image = face_align.norm_crop(face, landmark=faces[0].kps, image_size=224)
            first_iteration = False

    average_embedding = torch.mean(torch.stack(faceid_all_embeds, dim=0), dim=0)

    return average_embedding, face_image



def generate_body(images, positive_prompt, negative_prompt, face_strength, likeness_strength, 
                  ip_ckpt, body_pipe, USE_XL=True, preserve_face_structure=True, 
                  num_inference_steps=25, num_samples=2, device='cuda'):
    
    """
    Generate body images using the given prompts.
    Args:
        images: List of face images (paths, PIL images, or OpenCV images)
        positive_prompt: Positive prompt for the body image
        negative_prompt: Negative prompt for the body image
        face_strength: Strength of the face in the final image
        likeness_strength: Strength of the likeness in the final image
        ip_ckpt: Path to the IP model checkpoint
        body_pipe: to the diffusion pipeline object
        USE_XL: Use the XL pipeline
        preserve_face_structure: Preserve face structure in the final image
        num_inference_steps: Number of inference steps
        num_samples: Number of samples to generate
        device: Device to use (e.g., cuda or cpu)
    """
    # Get face embeddings
    average_embedding, face_image = get_face_embeddings(images, preserve_face_structure)

     # Load the IP adapter
    if preserve_face_structure:
        if face_image is None:
            raise ValueError("Face image is required for preserving face structure.")
        ip_adapter = IPAdapterFaceIDPlusXL if USE_XL else IPAdapterFaceIDPlus
    else:
        ip_adapter = IPAdapterFaceIDXL if USE_XL else IPAdapterFaceID
    
    image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    ip_model = ip_adapter(body_pipe, image_encoder_path, ip_ckpt, device=device)

    
    # Generate body images
    body_images = ip_model.generate(
            prompt=positive_prompt, negative_prompt=negative_prompt, faceid_embeds=average_embedding,
            scale=likeness_strength, face_image=face_image, shortcut=True, s_scale=face_strength,
            width=768, height=1024, num_inference_steps=num_inference_steps, num_samples=num_samples
        )
    
    return body_images

