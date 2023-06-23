import numpy as np
import time
import cv2
import os
import glob
import pandas as pd

def evaluate():
    folder_path = 'result02'
    file_list = glob.glob(folder_path + '/*')
    ls = []
    for file_path in file_list:
        # if file_path.endswith(".png"):
        img = cv2.imread(file_path, 0)
        # Downsample the image and resize it to match the original size
        downsampled = cv2.pyrDown(img)
        downsampled = cv2.resize(downsampled, (img.shape[1], img.shape[0]))

        # Compute the MSE
        mse = np.mean((img - downsampled)**2)

        # Calculate the histogram of the image
        hist, _ = np.histogram(img, bins=np.arange(256))
        # Normalize the histogram
        hist = hist / (img.shape[0] * img.shape[1])
        # Calculate the entropy of the image
        entropy = -np.sum(hist * np.log2(hist + 1e-8))

        ls.append((file_path, mse, entropy))
    print(len(ls))

    # Tạo DataFrame từ danh sách dữ liệu
    df = pd.DataFrame(ls, columns=['file_path', 'mse', 'entropy'])
    # Trích xuất tên file từ đường dẫn
    df['image_name'] = df['file_path'].apply(lambda x: x.split('/')[-1])
    # Chọn và sắp xếp lại thứ tự các cột
    df = df[['image_name', 'mse', 'entropy']]
    # Ghi DataFrame vào file CSV
    df.to_csv("MSE_ENTROPY.csv", index=False)

    # if mse < 10 or entropy < 7:
    #     response = {'message': "Sample photo is not detailed enough", 'status_code': 903}
    #     print(response)
    # else:
    #     response = {'message': "Sample image meets the requirements", 'status_code': 200}
    #     print(response)

if __name__ == "__main__":
    evaluate()
