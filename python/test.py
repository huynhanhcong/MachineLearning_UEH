import cv2
import numpy as np

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)

def region_of_interest(image):
    height, width = image.shape
    polygons = np.array([[(
        int(0.1 * width), height),
        (int(0.45 * width), int(0.6 * height)),
        (int(0.55 * width), int(0.6 * height)),
        (int(0.9 * width), height)
    ]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 6)
    return line_image

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None

    height, width, _ = image.shape

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue  # Tránh chia cho 0 (đường dọc)
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))

    lane_lines = []
    for fit in [left_fit, right_fit]:
        if len(fit) > 0:
            avg_slope, avg_intercept = np.mean(fit, axis=0)
            y1 = height
            y2 = int(height * 0.6)
            try:
                x1 = int((y1 - avg_intercept) / avg_slope)
                x2 = int((y2 - avg_intercept) / avg_slope)
                lane_lines.append((x1, y1, x2, y2))
            except ZeroDivisionError:
                continue

    return lane_lines

def process_video(video_path="", show_intermediate=False):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        canny_image = canny(frame)
        roi_image = region_of_interest(canny_image)

        lines = cv2.HoughLinesP(roi_image, rho=2, theta=np.pi / 180, threshold=60, minLineLength=20, maxLineGap=50)
        averaged_lines = average_slope_intercept(frame, lines)
        line_image = display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

        # Resize the output window (50%)
        scale_percent = 50
        width = int(combo_image.shape[1] * scale_percent / 100)
        height = int(combo_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_combo = cv2.resize(combo_image, dim, interpolation=cv2.INTER_AREA)

        if show_intermediate:
            resized_canny = cv2.resize(canny_image, dim, interpolation=cv2.INTER_AREA)
            resized_roi = cv2.resize(roi_image, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow("Canny", resized_canny)
            cv2.imshow("ROI", resized_roi)

        cv2.imshow("Curved Lane Detection", resized_combo)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Chạy hàm xử lý video
process_video(r"C:\Users\08688\Documents\video chinh\1.mp4", show_intermediate=False)
