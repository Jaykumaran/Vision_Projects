import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


IMG_DIR = "/media/server40901/HDD4TB/Jaykumaran_4tb/1_GS/InvRGBL/dataset_custom/000/images"
OUT_DIR = "/media/server40901/HDD4TB/Jaykumaran_4tb/1_GS/InvRGBL/dataset_custom/000/sky_masks"
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load the model from HuggingFace (no mmcv)
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    use_safetensors=True
).to(device)
model.eval()

# Cityscapes 'Sky' class index is 10
SKY_CLASS_ID = 10

img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])

print(f"Extracting sky masks for {len(img_files)} images using SegFormer B5...")

with torch.no_grad():
    for img_name in tqdm(img_files):
        img_path = os.path.join(IMG_DIR, img_name)
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size


        inputs = processor(images=image, return_tensors="pt").to(device)
        

        outputs = model(**inputs)
        logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

        # Upscale to original image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        )


        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        # Create binary mask (255 for sky, 0 for others)
        sky_mask = (pred_seg == SKY_CLASS_ID).astype(np.uint8) * 255
        


        mask_name = img_name.replace(".jpg", ".png")
        Image.fromarray(sky_mask).save(os.path.join(OUT_DIR, mask_name))

print(f"Done! Sky masks saved to: {OUT_DIR}")
