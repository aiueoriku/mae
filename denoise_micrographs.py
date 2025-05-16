# obtained from cryotransformer

import numpy as np
import mrcfile
import cv2
from numpy.fft import fft2, ifft2
# from scipy.signal import gaussian
from scipy.signal.windows import gaussian

def transform(image):
    i_min = image.min()
    i_max = image.max()

    image = ((image - i_min)/(i_max - i_min)) * 255
    return image.astype(np.uint8)

#GaussianBlur
def standard_scaler(image): 
    kernel_size = 9
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    mu = np.mean(image)
    sigma = np.std(image)
    image = (image - mu)/sigma
    image = transform(image).astype(np.uint8)
    return image

#fastNlMeansDenoising
def contrast_enhancement(image):
    enhanced_image = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    return enhanced_image


def gaussian_kernel(kernel_size = 3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h

#wiener_filter
def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s = img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return dummy

def clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))

    # Apply CLAHE to the image
    img_equalized = clahe.apply(transform(image))
    return img_equalized

def guided_filter(input_image, guidance_image, radius=20, epsilon=0.1):
    # Convert images to float32
    input_image = input_image.astype(np.float32) / 255.0
    guidance_image = guidance_image.astype(np.float32) / 255.0

    # Compute mean values of the guidance image and input image
    mean_guidance = cv2.boxFilter(guidance_image, -1, (radius, radius))
    mean_input = cv2.boxFilter(input_image, -1, (radius, radius))

    # Compute correlation and covariance of the guidance and input images
    mean_guidance_input = cv2.boxFilter(guidance_image * input_image, -1, (radius, radius))
    covariance_guidance_input = mean_guidance_input - mean_guidance * mean_input

    # Compute squared mean of the guidance image
    mean_guidance_sq = cv2.boxFilter(guidance_image * guidance_image, -1, (radius, radius))
    variance_guidance = mean_guidance_sq - mean_guidance * mean_guidance

    # Compute weights and mean of the weights
    a = covariance_guidance_input / (variance_guidance + epsilon)
    b = mean_input - a * mean_guidance
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    # Compute the filtered image
    output_image = mean_a * guidance_image + mean_b

    return transform(output_image)

kernel = gaussian_kernel(kernel_size = 9)
def denoise(image_path):
    if image_path.lower().endswith('.mrc'):
        image = mrcfile.read(image_path)
        image = image.T
        image = np.rot90(image)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    normalized_image = standard_scaler(np.array(image))
    contrast_enhanced_image = contrast_enhancement(normalized_image)
    weiner_filtered_image = wiener_filter(contrast_enhanced_image, kernel, K = 30)
    clahe_image = clahe(weiner_filtered_image)
    guided_filter_image = guided_filter(clahe_image, weiner_filtered_image)
    return guided_filter_image
    




import os
from tqdm import tqdm
import concurrent.futures


# IDリスト
data_ids = [
    10005, 10075, 10184, 10387, 10532, 10737, 11051,
    10017, 10077, 10240, 10576, 10760, 11056,
    10028, 10081, 10289, 10406, 10590, 10816, 11057,
    10059, 10093, 10291, 10444, 10669, 10852, 11183,
    10061, 10096, 10345, 10526, 10671, 10947, 10389
]

def process_and_save(jpg_path, save_path):
    denoised_image = denoise(jpg_path)
    cv2.imwrite(save_path, denoised_image)
    # print(f"Denoised image saved to {save_path}")

### お試し用
# # 保存先フォルダ
# output_folder = "denoised_images"
# os.makedirs(output_folder, exist_ok=True)

# for data_id in data_ids:

#   # ベースパス
#   base_path = "/mnt/ssd2/riku/cryoppp"
#   # data_id = 10005
#   micrographs_path = os.path.join(base_path, str(data_id), "micrographs")
#   jpg_files = sorted([f for f in os.listdir(micrographs_path) if f.lower().endswith('.jpg')])
#   jpg_path = os.path.join(micrographs_path, jpg_files[0])

#   denoised_image = denoise(jpg_path)
#   output_path = os.path.join(output_folder, f"denoised_{data_id}.jpg")
#   cv2.imwrite(output_path, denoised_image)
#   print(f"Denoised image saved to {output_path}")


### 全画像をdenoiseする場合
# ベースパス
base_path = "/mnt/ssd2/riku/cryoppp"

# denoised画像の保存先
output_base = "../cryoppp_denoised"
os.makedirs(output_base, exist_ok=True)

for data_id in tqdm(data_ids, desc="data_id"):
    if data_id == 10389:
        sub_dirs = ["10389A", "10389B"]
        for sub in sub_dirs:
            micrographs_path = os.path.join(base_path, str(data_id), sub)
            if not os.path.exists(micrographs_path):
                continue
            jpg_files = sorted([f for f in os.listdir(micrographs_path) if f.lower().endswith('.jpg')])
            save_dir = os.path.join(output_base, f"{data_id}", sub)
            os.makedirs(save_dir, exist_ok=True)
            jpg_paths = [os.path.join(micrographs_path, f) for f in jpg_files]
            save_paths = [os.path.join(save_dir, f) for f in jpg_files]
            with concurrent.futures.ProcessPoolExecutor() as executor:
                list(tqdm(executor.map(process_and_save, jpg_paths, save_paths), total=len(jpg_files), desc=f"jpgs_{data_id}_{sub}", leave=False))
    else:
        micrographs_path = os.path.join(base_path, str(data_id), "micrographs")
        jpg_files = sorted([f for f in os.listdir(micrographs_path) if f.lower().endswith('.jpg')])
        save_dir = os.path.join(output_base, f"{data_id}")
        os.makedirs(save_dir, exist_ok=True)
        jpg_paths = [os.path.join(micrographs_path, f) for f in jpg_files]
        save_paths = [os.path.join(save_dir, f) for f in jpg_files]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(process_and_save, jpg_paths, save_paths), total=len(jpg_files), desc=f"jpgs_{data_id}", leave=False))
