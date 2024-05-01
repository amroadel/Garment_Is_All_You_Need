from PIL import Image, ImageOps
import os
import json

def pad_and_resize_garment(image_path, save_path):
    """
    Resize and pad a garment image to 768x1024 pixels.
    """
    # Open the image
    original_image = Image.open(image_path)

    # Desired dimensions
    desired_width = 768
    desired_height = 1024

    # Calculate aspect ratios
    aspect_ratio = original_image.width / original_image.height
    desired_aspect_ratio = desired_width / desired_height

    # Calculate new dimensions maintaining aspect ratio
    if aspect_ratio > desired_aspect_ratio:
        # Image is wider, calculate height based on width
        new_height = int(desired_width / aspect_ratio)
        new_size = (desired_width, new_height)
    else:
        # Image is taller or square, calculate width based on height
        new_width = int(desired_height * aspect_ratio)
        new_size = (new_width, desired_height)

    # Resize the image with aspect ratio maintained
    resized_image = original_image.resize(new_size, Image.ANTIALIAS)

    # Calculate padding
    width_padding = desired_width - resized_image.width
    height_padding = desired_height - resized_image.height

    # Calculate padding on each side
    left_padding = width_padding // 2
    right_padding = width_padding - left_padding
    top_padding = height_padding // 2
    bottom_padding = height_padding - top_padding

    # Add padding
    padded_image = ImageOps.expand(resized_image,
                                   (left_padding, top_padding, right_padding, bottom_padding),
                                   fill="white")
    padded_image.save(save_path)
    return padded_image

def image_grid(imgs, rows, cols):
    """
    Create a grid of images.
    """
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def pad_face(image):
    """
    Pad face image with white color.
    """
    # Set the desired padding size
    padding = (150, 150, 150, 150)  # (left, top, right, bottom)

    # Pad the image with white color
    padded_image = ImageOps.expand(image, padding, fill='white')
    return padded_image

def save_data(images, folder_name, image_name, json_data=None):
    """
    Save images and JSON data to a specified folder.

    Args:
    - images: PIL Image or list of PIL Images
    - folder_name: Name of the folder to save images and JSON file
    - image_name: Name of the image(s) to be saved
    - json_data: JSON data to be saved (optional)
    """
    # Create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Check if images is a list
    if isinstance(images, list):
        # Save each image in the list
        for idx, image in enumerate(images):
            image_path = os.path.join(folder_name, f"{image_name}_{idx+1}.png")
            image.save(image_path)
        print(f"{len(images)} images saved to {folder_name}")
    else:
        # Save the single image
        image_path = os.path.join(folder_name, f"{image_name}.png")
        images.save(image_path)
        print(f"Image saved to {folder_name}/{image_name}.png")

    # Save JSON data if provided
    if json_data:
        json_path = os.path.join(folder_name, f"{image_name}.json")
        with open(json_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
        print(f"JSON data saved to {folder_name}/{image_name}.json")


def load_images_from_file_or_folder(path):
    if os.path.isfile(path):  # Check if it's a file
        # Load a single image
        image = Image.open(path)
        return [image]
    elif os.path.isdir(path):  # Check if it's a directory
        # Load images from the folder
        images = []
        for filename in os.listdir(path):
            # Check if the file is an image file
            if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                # Load the image using PIL
                image_path = os.path.join(path, filename)
                image = Image.open(image_path)
                images.append(image)
        return images
    else:
        raise ValueError("Invalid path. It must be a path to an image file or a folder containing images.")


def get_propmpts(prompt_config_path, garment_description=None, garment_type=None):
    """
    Load prompts from a JSON file.
    Args:
        prompt_config_path: Path to the JSON file containing prompts
    Returns:
        positive_prompt: Positive prompt for the image generation
        negative_prompt: Negative prompt for the image generation
    """
    with open(prompt_config_path, 'r') as file:
        prompt_config = json.load(file)

    facial_features = prompt_config["facial_features"]
    body_features = prompt_config["body_features"]

    # Fill the face prompt template
    face_prompt = f'''
        hyperdetailed photography, soft light, head portrait, (white background:1.3), skin details, sharp and in focus, 
        {facial_features["gender"]} {facial_features["race"]} {facial_features["additional_details"]}, {facial_features["hair"]["length"]} 
        ({facial_features["hair"]["color"]}: 1.4) {facial_features["hair"]["style"]} hair, clear texture, {facial_features["eyes"]["description"]}, 
        {facial_features["additional_face_features"]}, cute, beautiful, 8k, professional, adult, age of 30, red shirt
    '''

    if garment_description:

        if garment_type == "top":
            clothing = " ".join(garment_description.split()[:10]) + " " + body_features["clothing"]["bottom"]
        elif garment_type == "bottom":
            clothing = body_features["clothing"]["top"] + " " + " ".join(garment_description.split()[:10])
        else:
            clothing = " ".join(garment_description.split()[:10])
        
    # Fill the body prompt template
    body_prompt = f'''
        a ({body_features["height"]} body: 1.4) fullbody photograph of a ({facial_features["gender"]}: 2.0) 
        fashion model wearing ({clothing}: 1.5), {facial_features["hair"]["length"]} ({facial_features["hair"]["color"]}: 1.4) {facial_features["hair"]["style"]} hair, 
        {body_features["pose"]} pose, {body_features["background"]} background, zoomout, below knee
    '''


    negative_prompt = prompt_config.get("negative_prompt")

    return negative_prompt, face_prompt, body_prompt