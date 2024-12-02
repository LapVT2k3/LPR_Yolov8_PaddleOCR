import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
# import easyocr
from paddleocr import PaddleOCR, draw_ocr
import pandas as pd
from util import set_background
import uuid
import os
import re

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
os.environ['OMP_NUM_THREADS'] = '1'
font_path = "arial.ttf"
set_background("./imgs/background.png")

folder_path = "./licenses_plates_imgs_detected/"
LICENSE_MODEL_DETECTION_DIR = './models/best (3).pt'
COCO_MODEL_DIR = "./models/yolov8n.pt"

# reader = easyocr.Reader(['en'], gpu=False)
ocr = PaddleOCR(use_angle_cls = True, use_gpu = False, lang='en')

vehicles = [2]

header = st.container()
body = st.container()

coco_model = YOLO(COCO_MODEL_DIR)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

state = "Uploader"

if "state" not in st.session_state :
    st.session_state["state"] = "Uploader"



# def read_license_plate(license_plate_crop, img):
#     scores = 0
#     detections = reader.readtext(license_plate_crop)

#     width = img.shape[1]
#     height = img.shape[0]
    
#     if detections == [] :
#         return None, None

#     rectangle_size = license_plate_crop.shape[0]*license_plate_crop.shape[1]

#     plate = [] 

#     for result in detections:
#         length = np.sum(np.subtract(result[0][1], result[0][0]))
#         height = np.sum(np.subtract(result[0][2], result[0][1]))
        
#         if length*height / rectangle_size > 0.17:
#             bbox, text, score = result
#             text = result[1]
#             text = text.upper()
#             scores += score
#             plate.append(text)
    
#     if len(plate) != 0 : 
#         return " ".join(plate), scores/len(plate)
#     else :
#         return " ".join(plate), 0

# def read_license_plate(license_plate_crop):
#     global reader

#     height, width = license_plate_crop.shape[:2]

#     aspect_ratio = width / height

#     threshold_ratio = 2.5

#     if aspect_ratio > threshold_ratio:
#         detections = reader.readtext(license_plate_crop)
#         if len(detections) > 0:
#             _, text, score = detections[0]
#             return text.replace(" ", ""), score

#     else:
#         mid_height = height // 2
#         top_crop = license_plate_crop[0:mid_height, :]
#         bottom_crop = license_plate_crop[mid_height:, :]

#         text_top, score_top = None, 0
#         text_bottom, score_bottom = None, 0

#         if top_crop.size > 0:
#             detections_top = reader.readtext(top_crop)
#             if len(detections_top) > 0:
#                 _, text_top, score_top = detections_top[0]

#         if bottom_crop.size > 0:
#             detections_bottom = reader.readtext(bottom_crop)
#             if len(detections_bottom) > 0:
#                 _, text_bottom, score_bottom = detections_bottom[0]

#         if text_top and text_bottom:
#             full_text = text_top.replace(" ", "") + " " + text_bottom.replace(" ", "")
#             avg_score = (score_top + score_bottom) / 2
#             return full_text, avg_score
#         elif text_top:
#             return text_top.replace(" ", ""), score_top
#         elif text_bottom:
#             return text_bottom.replace(" ", ""), score_bottom

def paddle_ocr(frame, x1, y1, x2, y2, padding=10):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    x1 = max(x1 - padding, 0)
    y1 = max(y1 - padding, 0)
    x2 = min(x2 + padding, frame.shape[1])
    y2 = min(y2 + padding, frame.shape[0])
    
    original_height = abs(y2 - y1)
    
    crop = frame[y1:y2, x1: x2]
    result = ocr.ocr(crop) #
    
    gray = None
    binary = None
    # gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # result = ocr.ocr(binary)
    
    print(result)
    text = ""    
    annotated = None
    if result[0] != None:
        boxes = [res[0] for res in result[0]]
        texts = [res[1][0] for res in result[0]]
        scores = [res[1][1] for res in result[0]]
        
        texts_use = []
        # binary_3channel = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        # annotated = draw_ocr(binary_3channel, boxes, texts, scores, font_path=font_path)
        
        annotated = draw_ocr(crop, boxes, texts, scores, font_path=font_path)
        for r in result[0]:
            score = r[1][1]
            box = r[0]
            box_height = abs(box[2][1] - box[0][1])
            
            if np.isnan(score):
                score = 0
            else:
                score = int(score * 100)
            if score > 90 and box_height > 0.1 * original_height:
                pattern = re.compile('[\W]')
                text = pattern.sub('', r[1][0])
                text = text.replace("???", "")
                text = text.replace("O", "0")
                text = text.replace("粤", "")
                texts_use.append(text)
                
        texts_sorted = sorted(texts_use, key=len)
        text = ''.join(texts_sorted)
        
    return gray, binary, annotated, str(text)

def model_prediction(img):
    licenses_texts = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    object_detections = coco_model(img)[0]
    license_detections = license_plate_detector(img)[0]

    if len(object_detections.boxes.cls.tolist()) != 0 :
        for detection in object_detections.boxes.data.tolist() :
            xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection

            if int(class_id) in vehicles :
                cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)
    else :
            xcar1, ycar1, xcar2, ycar2 = 0, 0, 0, 0
            car_score = 0

    if len(license_detections.boxes.cls.tolist()) != 0 :
        license_plate_crops_total = []
        license_plate_gray_crops = []
        license_plate_thresh_crops = []
        license_plate_annotated_crops = []
        for license_plate in license_detections.boxes.data.tolist() :
            x1, y1, x2, y2, score, class_id = license_plate

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

            license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]

            img_name = '{}.jpg'.format(uuid.uuid1())
         
            cv2.imwrite(os.path.join(folder_path, img_name), license_plate_crop)
            
            # license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY) 
            # _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
            # licenses_texts.append(license_plate_text)
            
            # if license_plate_text is not None and license_plate_text_score is not None  :
            #     license_plate_crops_total.append(license_plate_crop)
            
            # cv2.imshow("thresh", license_plate_crop_thresh)
            # cv2.waitKey(0)
            
            license_plate_gray_crop, license_plate_thresh_crop, license_plate_annotated_crop, license_plate_text = paddle_ocr(img, x1, y1, x2, y2)
            licenses_texts.append(license_plate_text)

            if license_plate_text is not None:
                license_plate_crops_total.append(license_plate_crop)
                license_plate_gray_crops.append(license_plate_gray_crop)
                license_plate_thresh_crops.append(license_plate_thresh_crop)
                license_plate_annotated_crops.append(license_plate_annotated_crop)
          

        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        return [img_wth_box, licenses_texts, license_plate_crops_total, 
                license_plate_gray_crops, license_plate_thresh_crops, license_plate_annotated_crops]
    else : 
        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return [img_wth_box]
    

with header :
    col1 = st.columns([1])[0]
    col1.title("Nhận diện biển số xe - Nhóm 15")


with body :

    if st.session_state["state"] == "Uploader" :
        img = st.file_uploader("Upload a Car Image: ", type=["png", "jpg", "jpeg"])

    _, col2, _ = st.columns([0.3,1,0.2])

    _, col5, _ = st.columns([0.8,1,0.2])

    
    if img is not None:
        image = np.array(Image.open(img))    
        col2.image(image)

        if col5.button("Apply Detection"):
            results = model_prediction(image)

            if len(results) == 6 :
                prediction, texts, license_plate_crop, license_plate_gray_crop, \
                    license_plate_thresh_crop, license_plate_annotated_crop = results

                texts = [i for i in texts if i is not None]
                
                if len(texts) == 1 and len(license_plate_crop) :
                    _, col3, _ = st.columns([0.4, 1, 0.2])
                    col3.header("Detection Results ✅:")

                    _, col4, _ = st.columns([0.1, 1, 0.1])
                    col4.image(prediction)

                    _, col9, _ = st.columns([0.4, 1, 0.2])
                    col9.header("License Cropped ✅:")

                    _, col10, _ = st.columns([0.3, 1, 0.1])
                    col10.image(license_plate_crop[0], caption="Original Crop", width=500)

                    if license_plate_gray_crop[0] != None:
                        _, col12, _ = st.columns([0.3, 1, 0.1])
                        col12.image(license_plate_gray_crop[0], caption="Gray", width=500)

                    if license_plate_thresh_crop[0] != None:
                        _, col13, _ = st.columns([0.3, 1, 0.1])
                        col13.image(license_plate_thresh_crop[0], caption="Threshold", width=500)

                    _, col14, _ = st.columns([0.3, 1, 0.1])
                    col14.image(license_plate_annotated_crop[0], caption="Annotated")

                    _, col11, _ = st.columns([0.45, 1, 0.55])
                    col11.success(f"License Number: {texts[0]}")

                elif len(texts) > 1 and len(license_plate_crop) > 1  :
                    _, col3, _ = st.columns([0.4, 1, 0.2])
                    col3.header("Detection Results ✅:")

                    _, col4, _ = st.columns([0.1, 1, 0.1])
                    col4.image(prediction)

                    _, col9, _ = st.columns([0.4, 1, 0.2])
                    col9.header("License Cropped ✅:")

                    _, col6, _ = st.columns([0.1, 1, 0.1])
                    for i in range(len(license_plate_crop)):
                        col6.image(license_plate_crop[i], caption=f"Original Crop {i+1}", width=500)
                        if license_plate_gray_crop[i] != None:
                            col6.image(license_plate_gray_crop[i], caption=f"Gray {i+1}", width=500)
                        if license_plate_thresh_crop[i] != None:
                            col6.image(license_plate_thresh_crop[i], caption=f"Threshold {i+1}", width=500)
                        col6.image(license_plate_annotated_crop[i], caption=f"Annotated {i+1}")

                        col6.success(f"License Number {i+1}: {texts[i]}")

            else :
                prediction = results[0]
                _, col3, _ = st.columns([0.4,1,0.2])
                col3.header("Detection Results ✅:")

                _, col4, _ = st.columns([0.3,1,0.1])
                col4.image(prediction)




 