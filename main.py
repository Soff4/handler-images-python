# File: data_augmentation_bot.py
import os
import pandas as pd
import numpy as np
from PIL import Image
import random
import zipfile
import shutil
import pickle
import glob
from lzw_handler import LZW
from typing import Optional, Union, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    TEMP_DIR = 'temp_extracted'
    RESULT_DIR = 'results'
    NUM_AUGMENTATIONS = 3

class DataAugmentation:
    def __init__(self, archive_dir: str, config: Config):
        self.archive_dir = archive_dir
        self.data = None
        self.config = config
        self.lzw = LZW()
        self.archive_path = self._find_archive()

    def _find_archive(self) -> str:
        for file in os.listdir(self.archive_dir):
            if file.endswith("_images.zip"):
                return os.path.join(self.archive_dir, file)
        raise FileNotFoundError("No archive matching the pattern '*_images.zip' found.")

    def extract_archive(self):
        if not zipfile.is_zipfile(self.archive_path):
            raise ValueError("The provided file is not a ZIP archive.")
        
        with zipfile.ZipFile(self.archive_path, 'r') as zip_ref:
            zip_ref.extractall(self.config.TEMP_DIR)
    
    def load_data(self, file_path: str):
        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.data = Image.open(file_path)
        else:
            self.data = None
    
    def normalize(self):
        if isinstance(self.data, pd.DataFrame):
            self.data = (self.data - self.data.mean()) / self.data.std()
        else:
            raise TypeError("Normalization can only be applied to CSV data.")
    
    def add_noise(self, noise_level: float = 0.01):
        if isinstance(self.data, pd.DataFrame):
            noise = np.random.randn(*self.data.shape) * noise_level
            self.data += noise
        else:
            raise TypeError("Noise addition can only be applied to CSV data.")
    
    def random_drop(self, drop_fraction: float = 0.1):
        if isinstance(self.data, pd.DataFrame):
            rows_to_drop = random.sample(range(len(self.data)), int(len(self.data) * drop_fraction))
            self.data = self.data.drop(rows_to_drop).reset_index(drop=True)
        else:
            raise TypeError("Row dropping can only be applied to CSV data.")
    
    def rotate_image(self, degrees: int = 45) -> Image.Image:
        if isinstance(self.data, Image.Image):
            return self.data.rotate(degrees)
        else:
            raise TypeError("Image rotation can only be applied to image data.")
    
    def resize_image(self, scale_factor: float = 1.2) -> Image.Image:
        if isinstance(self.data, Image.Image):
            new_size = (int(self.data.width * scale_factor), int(self.data.height * scale_factor))
            return self.data.resize(new_size)
        else:
            raise TypeError("Image resizing can only be applied to image data.")
    
    def add_image_noise(self, noise_level: int = 25) -> Image.Image:
        if isinstance(self.data, Image.Image):
            noise = np.random.normal(0, noise_level, (self.data.height, self.data.width, 3))
            noise = (noise - noise.min()) / (noise.max() - noise.min()) * 255
            noise = noise.astype(np.uint8)
            noise_image = Image.fromarray(noise)
            return Image.blend(self.data, noise_image, 0.5)
        else:
            raise TypeError("Image noise addition can only be applied to image data.")
    
    def lzw_compress_image(self, image_path: str, output_dir: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The file {image_path} does not exist.")
        
        image = Image.open(image_path)
        image_data = image.tobytes()
        image_size = image.size

        compressed_data = self.lzw.compress(image_data.decode('latin1'))
        output_file_name = os.path.splitext(os.path.basename(image_path))[0] + "_compressed.lzw"
        output_path = os.path.join(output_dir, output_file_name)
        
        with open(output_path, 'wb') as f:
            pickle.dump((image_size, compressed_data), f)
        logging.info(f"Compressed image saved to {output_path}")
    
    def lzw_decompress_images(self, input_dir: str, output_dir: str):
        compressed_files = glob.glob(os.path.join(input_dir, '*_compressed.lzw'))
        
        for compressed_file_path in compressed_files:
            with open(compressed_file_path, 'rb') as f:
                image_size, compressed_data = pickle.load(f)
            
            decompressed_data = self.lzw.decompress(compressed_data).encode('latin1')
            image = Image.frombytes('RGB', image_size, decompressed_data)
            output_file_name = os.path.splitext(os.path.basename(compressed_file_path))[0].replace('_compressed', '_decompressed') + ".png"
            output_path = os.path.join(output_dir, output_file_name)
            image.save(output_path)
            logging.info(f"Decompressed image saved to {output_path}")
    
    def save_data(self, output_path: str, data_to_save: Union[pd.DataFrame, Image.Image]):
        if isinstance(data_to_save, pd.DataFrame):
            data_to_save.to_csv(output_path, index=False)
        elif isinstance(data_to_save, Image.Image):
            data_to_save.save(output_path)
        else:
            raise ValueError("No data to save or unsupported data format.")
    
    def augment_image(self, output_file_path: str) -> int:
        augment_index = 1
        aug_methods = [self.rotate_image, self.resize_image, self.add_image_noise]
        for _ in range(self.config.NUM_AUGMENTATIONS):
            for method in aug_methods:
                try:
                    aug_image = method()
                    aug_output_path = f"{output_file_path}_aug{augment_index}.png"
                    self.save_data(aug_output_path, aug_image)
                    augment_index += 1
                except Exception as e:
                    logging.error(f"Failed to augment image {output_file_path}: {e}")
        return augment_index
    
    def augment_csv(self, output_file_path: str):
        aug_methods = [self.normalize, self.add_noise, self.random_drop]
        for method in aug_methods:
            try:
                method()
                aug_output_path = f"{output_file_path}_aug.csv"
                self.save_data(aug_output_path, self.data)
            except Exception as e:
                logging.error(f"Failed to augment CSV {output_file_path}: {e}")
    
    def process_files(self):
        for root, _, files in os.walk(self.config.TEMP_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    self.load_data(file_path)
                    output_file_base = os.path.join(self.config.RESULT_DIR, os.path.splitext(file)[0])
                    
                    if isinstance(self.data, pd.DataFrame):
                        self.augment_csv(output_file_base)
                    elif isinstance(self.data, Image.Image):
                        self.augment_image(output_file_base)
                        self.lzw_compress_image(file_path, self.config.RESULT_DIR)
                except Exception as e:
                    logging.error(f"Failed to process {file_path}: {e}")

    def create_result_archive(self):
        with zipfile.ZipFile('results.zip', 'w') as zipf:
            for root, _, files in os.walk(self.config.RESULT_DIR):
                for file in files:
                    zipf.write(os.path.join(root, file),
                               os.path.relpath(os.path.join(root, file), self.config.RESULT_DIR))
    
    def clean_up(self):
        shutil.rmtree(self.config.TEMP_DIR)
        shutil.rmtree(self.config.RESULT_DIR)

    def augment_data_from_archive(self):
        self.extract_archive()
        os.makedirs(self.config.RESULT_DIR, exist_ok=True)
        self.process_files()
        self.create_result_archive()
        self.clean_up()

# Example usage:
config = Config()
augmenter = DataAugmentation(archive_dir='.', config=config)
augmenter.augment_data_from_archive()