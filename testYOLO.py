from ultralytics import YOLO
import cv2

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


# Load YOLO model
model = YOLO('./models/best (3).pt')

# Mode selection: 'camera', 'video', or 'image'
mode = "image"  # Thay đổi giữa 'camera', 'video', 'image'

# Path for video or image (chỉ dùng khi mode là 'video' hoặc 'image')
video_path = "./data_test/video/1.mp4"  # Đường dẫn video
image_path = "./data_test/image/5.jpg"  # Đường dẫn ảnh

# Desired display size (width x height)
display_width = 1280
display_height = 960

if mode == "camera":
    cap = cv2.VideoCapture(0)  # Dùng camera máy tính
elif mode == "video":
    cap = cv2.VideoCapture(video_path)  # Dùng video
elif mode == "image":
    frame = cv2.imread(image_path)  # Đọc ảnh

# Process frames
frame_nmr = -1
ret = True

while ret:
    if mode in ["camera", "video"]:
        frame_nmr += 1
        ret, frame = cap.read()
        if not ret:
            break
    elif mode == "image":
        ret = False  # Chỉ xử lý một ảnh duy nhất

    # Detect license plates
    results = model(frame)[0]

    for license_plate in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Crop license plate
        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

        # Process license plate
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 120, 255, cv2.THRESH_BINARY_INV)

        # Read license plate number
        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

        # Display license plate text above the bounding box
        # cv2.putText(frame, license_plate_text, (int(x1), int(y1) - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # # Print license plate text
        # print(f"Detected License Plate: {license_plate_text} (Score: {license_plate_text_score})")

    # Resize frame for display
    resized_frame = cv2.resize(frame, (display_width, display_height))

    # Show video or image with bounding boxes
    cv2.imshow("License Plate Detection", resized_frame)

    # Break on pressing 'q'
    if mode in ["camera", "video"] and cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif mode == "image":
        cv2.waitKey(0)  # Đợi nhấn phím để đóng cửa sổ
        break

# Release video capture and close OpenCV window
if mode in ["camera", "video"]:
    cap.release()
cv2.destroyAllWindows()
