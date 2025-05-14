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
        (int(0.4 * width), int(0.6 * height)),
        (int(0.6 * width), int(0.6 * height)),
        (int(0.9 * width), height)
    ]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

def display_lanes_curved(image, left_fit_points, right_fit_points, line_color=(0, 255, 0), line_thickness=6):
    line_image = np.zeros_like(image)
    if left_fit_points is not None and len(left_fit_points) > 0:
        left_pts = np.array([left_fit_points], dtype=np.int32)
        cv2.polylines(line_image, left_pts, isClosed=False, color=line_color, thickness=line_thickness)
    if right_fit_points is not None and len(right_fit_points) > 0:
        right_pts = np.array([right_fit_points], dtype=np.int32)
        cv2.polylines(line_image, right_pts, isClosed=False, color=line_color, thickness=line_thickness)
    return line_image

def fit_polynomial_lanes(image_shape, lines):
    left_lane_points = []
    right_lane_points = []

    if lines is None:
        return None, None

    height, width = image_shape[:2]

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)

            # Lọc dựa trên độ dốc và vị trí chặt chẽ hơn
            if -0.8 < slope < -0.3 and x1 < width / 2 and x2 < width / 2:
                left_lane_points.extend([(x1, y1), (x2, y2)])
            elif 0.3 < slope < 0.8 and x1 > width / 2 and x2 > width / 2:
                right_lane_points.extend([(x1, y1), (x2, y2)])

    left_fit_curve = None
    right_fit_curve = None
    plot_y = np.linspace(int(height * 0.6), height - 1, 15).astype(int)

    if len(left_lane_points) > 5: # Cần nhiều điểm hơn để fit ổn định
        left_x = [p[0] for p in left_lane_points]
        left_y = [p[1] for p in left_lane_points]
        try:
            left_coeffs = np.polyfit(left_y, left_x, 2)
            left_fit_x = left_coeffs[0] * plot_y**2 + left_coeffs[1] * plot_y + left_coeffs[2]
            left_fit_curve = list(zip(left_fit_x.astype(int), plot_y))
        except (np.RankWarning, np.linalg.LinAlgError):
            print("Warning: Could not fit left lane polynomial.")
            left_fit_curve = None

    if len(right_lane_points) > 5: # Cần nhiều điểm hơn để fit ổn định
        right_x = [p[0] for p in right_lane_points]
        right_y = [p[1] for p in right_lane_points]
        try:
            right_coeffs = np.polyfit(right_y, right_x, 2)
            right_fit_x = right_coeffs[0] * plot_y**2 + right_coeffs[1] * plot_y + right_coeffs[2]
            right_fit_curve = list(zip(right_fit_x.astype(int), plot_y))
        except (np.RankWarning, np.linalg.LinAlgError):
            print("Warning: Could not fit right lane polynomial.")
            right_fit_curve = None

    return left_fit_curve, right_fit_curve


def process_video(video_path="", show_intermediate=False):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        canny_image = canny(frame)
        # Thử áp dụng morphological operations để giảm nhiễu
        kernel = np.ones((5, 5), np.uint8)
        dilated_canny = cv2.dilate(canny_image, kernel, iterations=1)
        eroded_canny = cv2.erode(dilated_canny, kernel, iterations=1) # Thử nghiệm với thứ tự

        roi_image = region_of_interest(eroded_canny) # Sử dụng ảnh đã qua xử lý hình thái

        lines = cv2.HoughLinesP(roi_image, rho=1, theta=np.pi / 180, threshold=40, minLineLength=30, maxLineGap=80) # Điều chỉnh tham số Hough

        left_lane_pts, right_lane_pts = fit_polynomial_lanes(frame.shape, lines)

        line_image = display_lanes_curved(frame, left_lane_pts, right_lane_pts)

        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

        scale_percent = 75
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

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Chạy hàm xử lý video
# Make sure the video path is correct and accessible
# Test with a video that has curved lanes
process_video("C:\\Users\\08688\\Documents\\Zalo Received Files\\nD_20.mp4", show_intermediate=False)
# Example with another video if you have one:
# process_video("path/to/your/curved_lane_video.mp4", show_intermediate=True)