import torch
import glob
import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from urllib.request import urlretrieve
from zipfile import ZipFile
from ultralytics import YOLO
from tqdm.auto import tqdm
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re



device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


model = YOLO('/home/jaykumaran/Vision_Projects/Vision_Projects/ANPR-OCR/YOLO11/runs/detect/yolo11m-license/weights/best.pt')


# We will use a non-downstreamed checkpoint i.e. TrOCR Large Stage 1 rather than printed or handwritten ckpt
# trocr_name = "microsoft/trocr-large-stage1"
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-stage1')
ocr_model = VisionEncoderDecoderModel.from_pretrained('lpr_ocr_base/').to(device)
count = 0


# def preprocess_ocr_image(image):
#     """Enhance contrast and sharpen image before OCR"""
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply Adaptive Histogram Equalization (CLAHE)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     enhanced = clahe.apply(gray)

#     # Apply sharpening filter
#     kernel = np.array([[0, -1, 0], 
#                        [-1, 5,-1], 
#                        [0, -1, 0]])
#     sharpened = cv2.filter2D(enhanced, -1, kernel)
#     sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
	
#     return sharpened_rgb
    
def validate_license_plate(text):
    """Validate if the text matches AA12AA1234 HWMVA format"""
    pattern = r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$"
    return bool(re.match(pattern, text))


def superRes(img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
 
    path = "EDSR_x4.pb"
    
    sr.readModel(path)
    
    sr.setModel("edsr",4)
    
    result = sr.upsample(img)
    
    # Resized image
    resized = cv2.resize(img,dsize=None,fx=2,fy=2)
    
    return resized




def ocr(image, processor, model, print_tokens = False):
    
    """image: PIL Image,
        print_tokens: Whether to print the generated integer tokens or not
        
        Returns:
            generated_text: OCR text string
    """
    global count
    # Perform ocr on detected and cropped images
    # image = cv2.resize(image, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
    # image = preprocess_ocr_image(image)
    # image = cv2.resize(image, (384,384), interpolation=cv2.INTER_CUBIC)
    
    # ****************************************************************************
    
    # image = superRes(image)
    cv2.imwrite(f"intermediate/intermediate_{count}.jpg",image)
    pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    if print_tokens:
        print(generated_ids)
    
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens = True
    )[0]
    count += 1    # increment globally
    
    # if validate_license_plate(generated_text):
    #     return generated_text 
    # else:
        
    #     return "" # Skip invalid results
    
    return generated_text

def draw_box(output, frame, processor, ocr_model, print_tokens = False, conf_threshold = 0.5):
    frame = np.array(frame[..., ::-1])
    line_width = max(round(sum(frame.shape) / 2 * 0.003), 2)
    font_thickness = max(line_width - 1, 1)
    
    for out in output:
        for (box, conf) in zip(out.boxes.xyxy, out.boxes.conf): 
            if conf > conf_threshold:   # confidence score  > 0.7
            
                point1 = (int(box[0]), int(box[1])) #tuple
                point2 = (int(box[2]), int(box[3])) #tuple
                
                #crop ROI and pass to ocr
                license_plate_roi = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                extracted_text = ocr(license_plate_roi, processor, ocr_model)
                
                cv2.rectangle(
                    frame,
                    point1, point2,
                    color = (0, 0, 255), # RED
                    thickness=3
                )
                
                w, h = cv2.getTextSize(
                    extracted_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = line_width /3,
                    thickness=font_thickness
                )[0]  #text width and height
                
                w = int(w - (0.20 * w))
                outside = point1[1] - h >= 3 
                
                point2 = point1[0] + w, point1[1] - h - 3 if outside else point1[1] + h + 3
                
                cv2.rectangle(
                    frame,
                    point1, point2,
                    color = (0, 0, 255),
                    thickness=-1,
                    lineType=cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    extracted_text,
                    (point1[0], point1[1] - 5 if outside else point1[1] + h + 2),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    color = (255, 255, 255),
                    fontScale= line_width / 3.8,
                    thickness=2,
                    lineType=cv2.LINE_AA
                    
                )
            
            
    return frame
        
def crop_and_ocr(all_images, processor, ocr_model, print_tokens = False):
    
    for image_name in all_images:
        image = cv2.imread(image_name)[..., ::-1]
        output = model.predict(image)
        frame = draw_box(output, image, processor, ocr_model, confidence_threshold = 0.5)
        return frame
        

def process_video(video_path, processor, ocr_model, output_path=None):
    if not os.path.exists(video_path):
        print("Error: Video file not found!")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file!")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 FPS if invalid
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) if output_path else None

    while True:
        ret, frame = cap.read()
        if not ret:  # If no frame is read, break the loop (video ended)
            print("✅ Video processing completed.")
            break
        
        if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            print("⚠️ Warning: Empty frame detected, skipping.")
            continue  # Skip empty frames
        
        output = model.predict(frame)  # Using processor instead of undefined 'model'
        frame = draw_box(output, frame, processor, ocr_model)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("License Plate Detection", frame)  # Display video
        if out:
            out.write(frame)  # Save output

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            print("🔴 Video processing interrupted by user.")
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

# Run the video processing
video_path = "/home/jaykumaran/Vision_Projects/ANPR-OCR/TrOCR/ViolationVideo/10.32.84.14Big-0159-1350-1359.MP4"
output_video_path = "non-regex_bangalore_license_ocr_output.mp4"
process_video(video_path, processor, ocr_model, output_video_path)

