import cv2
import numpy as np
import math

# 1. Doc frame dau tien cua video
cap = cv2.VideoCapture("video1.mp4")

if not cap.isOpened():
    print("Khong the mo video!")
    exit()

ret, frame = cap.read()
cap.release()

if not ret:
    print("Khong doc duoc frame!")
    exit()

# 2. Chuyen BGR sang HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# 3. Loc mau do
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([179, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)


# 4. Tim contour
contours, hierarchy = cv2.findContours(
    red_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

result = frame.copy()

print("Tong contour tim duoc:", len(contours))

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        continue

    circularity = 4 * math.pi * area / (perimeter ** 2)

    x, y, w, h = cv2.boundingRect(contour)
    ratio = w / h if h > 0 else 0

    print(
        f"Contour {i}: area={area:.1f}, "
        f"do tron={circularity:.2f}, "
        f"rong={w}, cao={h}, ratio={ratio:.2f}"
    )

    # Loc ung vien gan hinh tron
    if area >= 100 and circularity > 0.65 and 0.7 < ratio < 1.3:
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

# 5. Hien thi ket qua
cv2.imshow("Original Frame", frame)
cv2.imshow("Red Mask", red_mask)
cv2.imshow("Circle Candidates", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
