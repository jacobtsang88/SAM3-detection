import requests
import torch
from PIL import Image

from transformers import Sam3Model, Sam3Processor


#https://stackoverflow.com/questions/26392336/importing-images-from-a-directory-python-to-list-or-dictionary
from PIL import Image
import glob
image_list = []
for filename in sorted(glob.glob('test/*.png')):
    im = Image.open(filename)
    image_list.append(im)
#print(image_list)

def main():
    #implementation from HF sam3 docs
    model = Sam3Model.from_pretrained("facebook/sam3", device_map="auto")
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    '''
    cat_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
    kitchen_url = "http://images.cocodataset.org/val2017/000000136466.jpg"
    images = [
        Image.open(requests.get(cat_url, stream=True).raw).convert("RGB"),
        Image.open(requests.get(kitchen_url, stream=True).raw).convert("RGB")
    ]
    '''
    images = []
    for im in image_list:
        images.append(im.convert("RGB"))

    # Different text prompt for each image, now make it same for all images
    text_prompts = ["blue rectangular top of tent"] * len(images)

    prompt = "tent"

    for i, image in enumerate(image_list):
        image = image.convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.3,
            mask_threshold=0.3,
            target_sizes=inputs.get("original_sizes").tolist()
        )

        print(f"Image {i + 1}: {len(results[0]['masks'])} objects found")

    '''
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
    '''

if __name__ == "__main__":
    main()

