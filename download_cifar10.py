import os
import tarfile
import shutil
import random

import urllib.request

def download_and_extract_cifar10(destination_folder):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = url.split('/')[-1]
    filepath = os.path.join(destination_folder, filename)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    if not os.path.exists(filepath):
        print(f"Downloading CIFAR-10 dataset from {url}...")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete.")

    extract_path = os.path.join(destination_folder, "cifar-10-batches-py")
    if not os.path.exists(extract_path):
        print("Extracting CIFAR-10 dataset...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(path=destination_folder)
        print("Extraction complete.")
    else:
        print("CIFAR-10 dataset already extracted.")

        # Split into train and val folders
        train_folder = os.path.join(destination_folder, "train")
        val_folder = os.path.join(destination_folder, "val")

        if not os.path.exists(train_folder) or not os.path.exists(val_folder):
            print("Splitting CIFAR-10 dataset into train and val folders...")

            random.seed(42)  # For reproducibility

            # Create train and val directories
            os.makedirs(train_folder, exist_ok=True)
            os.makedirs(val_folder, exist_ok=True)

            # Assuming the extracted dataset contains batches of data
            for batch_file in os.listdir(extract_path):
                if batch_file.startswith("data_batch"):  # Training batches
                    shutil.move(os.path.join(extract_path, batch_file), train_folder)
                elif batch_file.startswith("test_batch"):  # Validation batch
                    shutil.move(os.path.join(extract_path, batch_file), val_folder)

            print("Splitting complete.")
        else:
            print("Train and val folders already exist.")

if __name__ == "__main__":
    destination = "./cifar10_data"
    download_and_extract_cifar10(destination)