import easyocr
import base64
import streamlit as st

def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

reader = easyocr.Reader(['en'], gpu=False)

# def read_license_plate(license_plate_crop):
#     detections = reader.readtext(license_plate_crop)
#     print(detections)

#     if detections == [] :
#         return None, None

#     for detection in detections:
#         bbox, text, score = detection

#         text = text.upper()
#         print(text)

#         if text is not None and score is not None and bbox is not None and len(text) >= 6:
#             return text, score

#     return None, None

def read_license_plate(license_plate_crop):
    """
    Đọc số biển xe từ ảnh cắt, hỗ trợ cả biển số 1 hàng và 2 hàng dựa trên tỉ lệ chiều dài/chiều cao.
    
    Args:
        license_plate_crop (numpy.ndarray): Ảnh đã cắt chứa biển số xe.
        
    Returns:
        tuple: (text_bien_so, score_trung_binh) nếu đọc được,
               (None, None) nếu không đọc được.
    """
    global reader

    # Xác định kích thước ảnh
    height, width = license_plate_crop.shape[:2]

    # Tính tỉ lệ chiều dài / chiều cao
    aspect_ratio = width / height

    # Ngưỡng tỉ lệ để phân biệt biển số 1 hàng và 2 hàng (có thể điều chỉnh)
    threshold_ratio = 2.5  # Tùy chỉnh ngưỡng dựa trên kích thước thực tế

    if aspect_ratio > threshold_ratio:  # Biển số 1 hàng (tỉ lệ dài)
        detections = reader.readtext(license_plate_crop)
        if len(detections) > 0:
            _, text, score = detections[0]
            return text.replace(" ", ""), score

    else:  # Biển số 2 hàng (tỉ lệ nhỏ)
        # Chia ảnh biển số thành 2 phần
        mid_height = height // 2
        top_crop = license_plate_crop[0:mid_height, :]  # Hàng trên
        bottom_crop = license_plate_crop[mid_height:, :]  # Hàng dưới

        # Đọc từng hàng
        text_top, score_top = None, 0
        text_bottom, score_bottom = None, 0

        if top_crop.size > 0:
            detections_top = reader.readtext(top_crop)
            if len(detections_top) > 0:
                _, text_top, score_top = detections_top[0]

        if bottom_crop.size > 0:
            detections_bottom = reader.readtext(bottom_crop)
            if len(detections_bottom) > 0:
                _, text_bottom, score_bottom = detections_bottom[0]

        # Ghép kết quả
        if text_top and text_bottom:
            full_text = text_top.replace(" ", "") + " " + text_bottom.replace(" ", "")
            avg_score = (score_top + score_bottom) / 2
            return full_text, avg_score
        elif text_top:
            return text_top.replace(" ", ""), score_top
        elif text_bottom:
            return text_bottom.replace(" ", ""), score_bottom

    return None, None

