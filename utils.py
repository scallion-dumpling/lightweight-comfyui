import os
import requests
from tqdm import tqdm
import yaml


def download_from_civit(model_info):
    """
    Downloads a model file given a dictionary or tuple with link, model_name, and model_type.
    
    Parameters:
    model_info (dict or tuple): A dictionary or tuple containing 'link', 'model_name', and 'model_type'.
    """
    
    # Unpack the model information
    if isinstance(model_info, dict):
        link = model_info['link']
        model_name = model_info['model_name']
        model_type = model_info['model_type']
    elif isinstance(model_info, tuple):
        link, model_name, model_type = model_info
    else:
        raise ValueError("model_info must be a dict or tuple containing 'link', 'model_name', and 'model_type'")
    
    # Get the API token from the environment variable
    api_token = os.getenv('CIVIT_KEY')
    
    # Construct the download URL
    download_url = f"{link}&token={api_token}"
    
    # Determine the target directory based on model_type
    target_dirs = {
        "checkpoints": "/workspace/ComfyUI/models/checkpoints/",
        "controlnet": "/workspace/ComfyUI/models/controlnet/",
        "loras": "/workspace/ComfyUI/models/loras/",
        "upscale_models": "/workspace/ComfyUI/models/upscale_models/",
        "unet": "/workspace/ComfyUI/models/unet/",
        "vae": "/workspace/ComfyUI/models/vae/"
    }
    
    target_dir = target_dirs.get(model_type)
    
    if not target_dir:
        raise ValueError(f"Invalid model_type '{model_type}'. Expected one of {list(target_dirs.keys())}.")
    
    file_path = os.path.join(target_dir, model_name)

    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"File '{model_name}' already exists in '{target_dir}'. Skipping download.")
        return
    
    # Make the initial request to get the file size
    response = requests.get(download_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Download the file with a progress bar
    with open(file_path, 'wb') as file, tqdm(
        desc=f"Downloading {model_name}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    
    print(f"File downloaded and saved to {file_path}")
