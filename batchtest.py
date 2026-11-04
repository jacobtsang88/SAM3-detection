import requests
import torch
from PIL import Image

from transformers import Sam3Model, Sam3Processor

#implementation from HF sam3 docs

def main():
    model = Sam3Model.from_pretrained("facebook/sam3", device_map="auto")
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    cat_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
    kitchen_url = "http://images.cocodataset.org/val2017/000000136466.jpg"
    images = [
        Image.open(requests.get(cat_url, stream=True).raw).convert("RGB"),
        Image.open(requests.get(kitchen_url, stream=True).raw).convert("RGB")
    ]

    # Different text prompt for each image
    text_prompts = ["nose", "dial"]

    inputs = processor(images=images, text=text_prompts, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results for both images
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )

    print(f"Image 1: {len(results[0]['masks'])} objects found")
    print(f"Image 2: {len(results[1]['masks'])} objects found")

if __name__ == "__main__"():
    main()