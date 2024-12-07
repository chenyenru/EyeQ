from utils.img_utils import (
    show_points,
    show_box,
    show_masks,
    convert_to_base64,
    plt_img_base64,
)
from utils.device_utils import initialize_device

import os
from PIL import Image
import numpy as np
import cv2

from torch import device
import torch

from numpydantic import NDArray


from transformers import CLIPProcessor, CLIPModel

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


def object_detection(image_path: str):
    """
    Performs object detection on the given image.

    Args:
        image_path: The path to the image file.

    Returns:
        dict: The outputs from the object detection model, including predicted classes and bounding boxes.
    """
    # Assuming you have an object detection model loaded
    model_name = "facebook/detectron2"
    torch.cuda.empty_cache()
    # Load your object detection model here
    # For example, using a pre-trained model from torchvision
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    predictor = DefaultPredictor(cfg)
    im = cv2.imread(image_path)
    outputs = predictor(im)
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
    v = Visualizer(
        im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("outputs/output.jpg", out.get_image()[:, :, ::-1])

    return outputs


def clip_tool(image_path: str):
    """
    Uses CLIP to encode image and text and compute similarity.

    Args:
        image_path: The path to the image file.

    Returns:
        torch.Tensor: The probabilities of the image matching each text description.
    """
    model_name = "openai/clip-vit-base-patch32"
    torch.cuda.empty_cache()
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.to(device)

    image = Image.open(image_path)
    inputs = processor(
        text=["a photo of a cat", "a photo of a dog"],
        images=image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    return probs


def sam2(image_path: str, input_point: NDArray = None, input_label: NDArray = None):
    """
    Initializes SAM2 model and predicts segmentation mask for the given image.

    Args:
        image_path: The path to the image file.
        input_point: Coordinates of input points for segmentation. Defaults to None.
        input_label: Labels for the input points. Defaults to None.

    Returns:
        None
    """
    model_name = "facebook/sam2-hiera-small"
    device = "cuda"
    torch.cuda.empty_cache()
    predictor = SAM2ImagePredictor.from_pretrained(
        model_name, trust_remote_code=True, device=device
    )
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))

    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=input_point, point_labels=input_label, multimask_output=True
    )

    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    show_masks(
        image,
        masks,
        scores,
        point_coords=input_point,
        input_labels=input_label,
        borders=True,
    )


if __name__ == "__main__":
    initialize_device()
    test_img_path = "examples/Siberian-Husky-dog.jpg"
    sam2(test_img_path, input_point=np.array(
        [[400, 375]]), input_label=np.array([1]))
    print(clip_tool(test_img_path))
    object_detection(test_img_path)
