import cv2
import numpy as np
import time

video_path = "test2.mp4"
video_path = "solidYellowLeft.mp4"
video_path = "solidWhiteRight.mp4"
 

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if video_path == 'test2.mp4':
        canny = cv2.Canny(blur, 50, 150)
    elif video_path == "solidYellowLeft.mp4":
        canny = cv2.Canny(blur, 50, 100)
    else:
        canny = cv2.Canny(blur, 50, 100)
    return canny

def region_of_interest(image):
    height = image.shape[0]

    if video_path == "test2.mp4":
        polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
        ])
    elif video_path == "solidYellowLeft.mp4":
        polygons = np.array([
        [(50, height), (1100, height), (470, 290)]
        ])
    else:
        polygons = np.array([
        [(10, height), (1100, height), (470, 280)]
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    try:
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return line_image
    except: 
        pass

def average_slope_intercept(image, lines):
    try:
        left_fit = []
        right_fit = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
            
        left_fit_avg = np.average(left_fit, axis=0)
        right_fit_avg = np.average(right_fit, axis=0)
        left_line = make_coordinates(image, left_fit_avg)
        right_line = make_coordinates(image, right_fit_avg)
        return np.array([left_line, right_line])
    except:
        pass

def make_coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * (3 / 5))

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return np.array([x1, y1, x2, y2])
    except:
        pass


capture = cv2.VideoCapture(video_path)

while( capture.isOpened() ):
    _, frame = capture.read()
    canny_img = canny(frame)
    cropped_image = region_of_interest(canny_img)
    lines = cv2.HoughLinesP(cropped_image, 3, np.pi / 180, 100, np.array([]), minLineLength = 10, maxLineGap = 7)
    averaged_lines = average_slope_intercept(frame, lines)

    line_image = display_lines(frame, averaged_lines)

    combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    if video_path != 'test2.mp4':
        time.sleep(0.04)
    cv2.imshow('result', combined_image)
    if cv2.waitKey(5) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()