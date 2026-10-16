import cv2
import numpy as np
import math

def process_video(input_video, output_video, student_ids):
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(input_video)
    
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print(f"Error opening video file: {input_video}")
        return

    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
    fps = cap.get(cv2.CAP_PROP_FPS)
       
    size = (frame_width, frame_height) 
    result = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    print(f"Processing {input_video} | Resolution = {size}")

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == False: 
            break
            
        # Khử nhiễu nhẹ để màu sắc đồng đều hơn trước khi chuyển HSV
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) 
          
        # Threshold of red/blue in HSV space (Giữ chuẩn theo hướng dẫn của bạn)
        lower_red = np.array([159, 50, 70])
        upper_red = np.array([180, 255, 255])
        
        # Bổ sung dải đỏ dải thấp (vì màu đỏ trong HSV nối từ 159-180 vòng qua 0-10)
        lower_red_2 = np.array([0, 50, 70])
        upper_red_2 = np.array([10, 255, 255])
        
        lower_blue = np.array([105, 150, 70])
        upper_blue = np.array([135, 255, 255])
      
        # preparing the mask to overlay 
        mask_red1 = cv2.inRange(hsv, lower_red, upper_red)
        mask_red2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = cv2.bitwise_or(mask_red, mask_blue)
        
        # Dùng Morphology để xóa các chấm nhiễu nhỏ (noise) trên mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            # 1. Area filtering (Chặn các đốm sáng nhỏ hoặc mảng màu khổng lồ)
            area = cv2.contourArea(cnt)
            if not (600 < area < 25000):
                continue
                
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
                
            # 2. Width & height of the bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            
            # 3. Shape check: Circularity
            circularity = 4 * math.pi * (area / (perimeter * perimeter))
            
            # =====================================================================
            # BỘ LỌC CỐT LÕI ĐỂ CHỐNG NHẬN DIỆN NHẦM VẬT THỂ
            # - Biển báo tròn phải có aspect ratio (w/h) xấp xỉ 1 (khoảng 0.8 -> 1.2)
            # - Độ tròn (circularity) phải gần 1 (khoảng 0.75 -> 1.25)
            # =====================================================================
            if 0.75 < circularity < 1.25 and 0.8 < aspect_ratio < 1.2:
                
                # Check vị trí: Chỉ lấy biển báo ở Giữa hoặc Bên phải đường
                center_x = x + w // 2
                if center_x > frame_width // 3:
                    
                    # Vẽ Bounding Box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
        # Chèn Mã số sinh viên 
        cv2.putText(frame, f"IDs: {student_ids}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        result.write(frame)
        
    cap.release()
    result.release()
    print(f"Finished {output_video}\n")

if __name__ == "__main__":
    student_ids = "52400047_52300001_52300008"
    video_files = ["video1.mp4", "video2.mp4"]
    
    for vid in video_files:
        output_name = f"{student_ids}_{vid}" 
        process_video(vid, output_name, student_ids)