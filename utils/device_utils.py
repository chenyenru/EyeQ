import os
import torch

def initialize_device() -> torch.device:
    """
    Initializes the device for computation and sets the environment variable for PyTorch.

    Code from github.com/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb

    Returns:
        torch.device: The device for computation.
    
    Raises:
        ValueError: If CUDA is not available.
    """
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # Use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # Turn on tfloat32 for Ampere GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    else:
        return ValueError("CUDA is not available. Please install CUDA to run this stack.")
       
    return device