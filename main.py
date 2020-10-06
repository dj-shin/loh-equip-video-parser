from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
from matplotlib import pyplot as plt


main_loc = [0.45, 0.06, 0.46, 0.10]
main_value_loc = [0.45, 0.15, 0.55, 0.12]
sub_locs = [
    [0.03, 0.345, 0.33, 0.08],
    [0.03, 0.418, 0.33, 0.08],
    [0.03, 0.488, 0.33, 0.08],
    [0.03, 0.568, 0.33, 0.08],
]
sub_value_locs = [
    [0.75, 0.345, 0.23, 0.08],
    [0.75, 0.418, 0.23, 0.08],
    [0.75, 0.488, 0.23, 0.08],
    [0.75, 0.568, 0.23, 0.08],
]
valid_names = [
    '공격력',
    '방어력',
    '체력',
    '치명타피해',
    '치명타확률',
    '효과적중',
    '효과저항',
    '속도',
]


def is_valid_value(value_string):
    if '%' in value_string:
        value = int(value_string.strip('%'))
        return value <= 90
    elif ',' in value_string:
        value = int(value_string.strip(','))
        return value <= 3600
    else:
        value = int(value_string)
        return value <= 3600


def rect_similar(r1, r2):
    return abs(r1[0] - r2[0]) < 10 and abs(r1[1] - r2[1]) < 10 and abs(r1[2] - r2[2]) < 10
    
    
def find_equip_area(video_path):
    cap = cv2.VideoCapture(video_path)
    rec = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if counter % (total_frames // 10) == 0:
                img = frame
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = (gray > 250).astype(np.uint8) * 255
                ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

                largest = None
                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)
                    if h < gray.shape[0] and w < gray.shape[1] and (largest is None or largest[2] * largest[3] < w * h):
                        largest = x, y, w, h
                if largest is not None:
                    rec.append(largest)
        else:
            break
        counter += 1

    arr = np.array(rec)
    crop_area = arr[np.argmax(arr[:, 3]), :]
    return crop_area


def crop(image, loc, w):
    return image[int(loc[1] * w):int((loc[1] + loc[3]) * w), int(loc[0] * w):int((loc[0] + loc[2]) * w)]


def is_different(img1, img2, w):
    gray1, gray2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score_prod = 1.0
    locs = [main_loc, main_value_loc] + sub_locs + sub_value_locs
    for loc in locs:
        (score, diff) = ssim(crop(gray1, loc, w), crop(gray2, loc, w), full=True)
        score_prod *= score
    return score_prod < 0.9
    
    
def separate_frames(video_path, crop_area):
    width = crop_area[2]
    prev_frame = None
    separated = []

    cap = cv2.VideoCapture(video_path)
    separated = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            crop = frame[crop_area[1]:crop_area[1]+crop_area[3], crop_area[0]:crop_area[0]+crop_area[2], :]
            if np.mean((cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) > 250).astype(np.float64)) > 0.5:
                if prev_frame is None or is_different(prev_frame, crop, width):
                    separated.append(crop)
                    prev_frame = crop
        else:
            break
    return separated


def parse_text(img, loc):
    w = img.shape[1]
    crop_area = img[
        int(loc[1] * w):int((loc[1] + loc[3]) * w),
        int(loc[0] * w):int((loc[0] + loc[2]) * w),
    :]
    text_img = np.pad(crop_area, ((int(0.01*w), int(0.01*w)), (0, 0), (0, 0)), 'constant', constant_values=(255))
    config = '-c tessedit_char_whitelist=공격력체방어치명타확률피해효과저항적중속도 --psm 13'
    return pytesseract.image_to_string(text_img, lang='kor', config=config).strip()


def parse_value(img, loc):
    w = img.shape[1]
    crop_area = img[
        int(loc[1] * w):int((loc[1] + loc[3]) * w),
        int(loc[0] * w):int((loc[0] + loc[2]) * w),
    :]
    text_img = np.pad(crop_area, ((int(0.01*w), int(0.01*w)), (0, 0), (0, 0)), 'constant', constant_values=(255))
    config = '-c tessedit_char_whitelist=0123456789,% --psm 13'
    return pytesseract.image_to_string(text_img, lang='kor', config=config).strip()


if __name__ == '__main__':
    video_path = '../equip/ipad.mp4'
    crop_area = find_equip_area(video_path)
    print(crop_area)
    separated = separate_frames(video_path, crop_area)
    print(len(separated))
    for image in separated:
        main = parse_text(image, main_loc), parse_value(image, main_value_loc)
        subs = []
        for name_loc, value_loc in zip(sub_locs, sub_value_locs):
            sub = parse_text(image, name_loc), parse_value(image, value_loc)
            if sub[0] not in valid_names:
                break
            subs.append(sub)
        print(main, subs)
