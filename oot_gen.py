from pathlib import Path
import sys
from PIL import Image
from OOTDiffusion.run.utils_ootd import get_mask_location

from OOTDiffusion.preprocess.openpose.run_openpose import OpenPose
from OOTDiffusion.preprocess.humanparsing.run_parsing import Parsing
from OOTDiffusion.ootd.inference_ootd_hd import OOTDiffusionHD
from OOTDiffusion.ootd.inference_ootd_dc import OOTDiffusionDC

def pad_image_to_aspect_ratio(image_path, target_width, target_height, background_color=(255, 255, 255)):
    """
    Pad an image to a specific aspect ratio and fill the background with a specified color.
    """
    # load the image
    img = Image.open(image_path)

    # calculate original and target aspect ratios
    original_width, original_height = img.size
    original_aspect = original_width / original_height
    target_aspect = target_width / target_height

    # determine where to pad (vertically or horizontally)
    if original_aspect > target_aspect:
        # if image is too wide
        new_width = original_width
        new_height = int(original_width / target_aspect)
    else:
        # if image is too tall
        new_height = original_height
        new_width = int(original_height * target_aspect)

    # new image with desired aspect ratio
    new_img = Image.new('RGB', (new_width, new_height), background_color)

    # calculate offsets
    pad_width = (new_width - original_width) // 2
    pad_height = (new_height - original_height) // 2

    # paste the original image into the new image
    new_img.paste(img, (pad_width, pad_height))

    #new_img.save(output_path)

    return new_img


def oot_gen(model_path, cloth_path, output_path = './ootd_images/', model_type = 'hd',\
            category: int = 0, scale: float = 2.0, n_steps: int = 20, n_samples: int = 4,\
            save_mask: bool = False, seed: int = 69, gpu_id: int = 0): 
         
    '''
    This function merges the clothes items with an image of a model using OOTDiffusion.
   
     Args:
    - model_path: path to the image of the model
    - cloth_path: path to the image of the clothes
    - output_path: path to save the generated image
    - model_type: must be 'hd' for upper body or 'dc' for full body
    - category: 0 = upperbody; 1 = lowerbody; 2 = dress
    - scale: guidance scale for the diffusion process
    - n_steps: number of steps for the diffusion process
    - s_samples: number of samples of the generated output images
    - save_mask: whether to generate the mask for the clothes
    - seed
    - gpu_id

    returns: path to dir with generated images
    '''

    openpose_model = OpenPose(gpu_id)
    parsing_model = Parsing(gpu_id)


    category_dict = ['upperbody', 'lowerbody', 'dress']
    category_dict_utils = ['upper_body', 'lower_body', 'dresses']

    if model_type == "hd":
        model = OOTDiffusionHD(gpu_id)
    elif model_type == "dc":
        model = OOTDiffusionDC(gpu_id)
    else:
        raise ValueError("model_type must be \'hd\' or \'dc\'!")
    

    if model_type == 'hd' and category != 0:
        raise ValueError("model_type \'hd\' requires category == 0 (upperbody)!")
    
    target_width, target_height = 768, 1024
    model_img = pad_image_to_aspect_ratio(model_path, target_width, target_height)
    cloth_img = pad_image_to_aspect_ratio(cloth_path, target_width, target_height)

    keypoints = openpose_model(model_img.resize((384, 512)))
    model_parse, _ = parsing_model(model_img.resize((384, 512)))

    mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
    mask = mask.resize((768, 1024), Image.NEAREST)
    mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
    
    masked_vton_img = Image.composite(mask_gray, model_img, mask)
    if save_mask:
        masked_vton_img.save(output_path + 'mask.jpg')

    images = model(
        model_type=model_type,
        category=category_dict[category],
        image_garm=cloth_img,
        image_vton=masked_vton_img,
        mask=mask,
        image_ori=model_img,
        num_samples=n_samples,
        num_steps=n_steps,
        image_scale=scale,
        seed=seed,
    )

    image_idx = 0
    for image in images:
        image.save(output_path + 'out_' + model_type + '_' + str(image_idx) + '.png')
        image_idx += 1
    
    return output_path
    



