import os
import cv2
import dlib
from imutils import face_utils


def crop_boundary(top, bottom, left, right, faces):
    if faces:
        top = max(0, top - 200)
        left = max(0, left - 100)
        right += 100
        bottom += 100
    else:
        top = max(0, top - 50)
        left = max(0, left - 50)
        right += 50
        bottom += 50
    return (top, bottom, left, right)


def crop_face(imgpath, dirName, extName):
    frame = cv2.imread(imgpath)
    basename = os.path.basename(imgpath)
    basename_without_ext = os.path.splitext(basename)[0]
    if frame is None:
        return print(f"Invalid file path: [{imgpath}]")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detect = dlib.get_frontal_face_detector()
    rects = face_detect(gray, 1)
    if not len(rects):
        return print(f"Sorry. HOG could not detect any faces from your image.\n[{imgpath}]")
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)

        top, bottom, left, right = crop_boundary(y, y + h, x, x + w, len(rects) <= 2)
        crop_img_path = os.path.join(dirName, f"{basename_without_ext}_crop_{i}{extName}")
        crop_img = frame[top: bottom, left: right]
        # cv2.imwrite(crop_img_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

        # Lấy phần ảnh đã được loại bỏ gương mặt
        removed_face_img = frame.copy()
        removed_face_img[top: bottom, left: right] = 0

        # Ghi ảnh đã loại bỏ gương mặt
        removed_face_path = os.path.join(dirName, f"{basename_without_ext}_removed_face_{i}{extName}")
        cv2.imwrite(removed_face_path, cv2.cvtColor(removed_face_img, cv2.COLOR_RGB2BGR))

    return print(f"SUCCESS: [{basename}]")


def main():
    extName = ".png"
    dirName = "result02"
    os.makedirs(dirName, exist_ok=True)

    # live_dir = "./"
    live_dir = "F:\Img_thang_3\Standard_Image_bonus"  # INPUT IMAGES DIRECTORY HERE
    live_files = os.listdir(live_dir)
    if not live_files:
        return print(f"No files found in directory: [{live_dir}]")
    for file in live_files:
        if file.endswith("tmp.jpg") or file.endswith("org.jpg"):
            imgpath = os.path.join(live_dir, file)
            crop_face(imgpath, dirName, extName)

if __name__ == "__main__":
    main()
