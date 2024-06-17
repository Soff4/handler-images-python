import os
import numpy as np
from PIL import Image
import random
import zipfile
import shutil
import pickle
import glob
from lzw_handler import LZW
import logging
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    TEMP_DIR = 'temp_extracted'
    RESULT_DIR = 'results'
    NUM_AUGMENTATIONS = 1

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
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.data = Image.open(file_path)
        else:
            self.data = None
    
    # x' = (x - x_c) * cos(θ) - (y - y_c) * sin(θ) + x_c
    # y' = (x - x_c) * sin(θ) + (y - y_c) * cos(θ) + y_c
    def rotate_image(self, degrees: int = 45) -> np.ndarray:
        if isinstance(self.data, Image.Image):
            image = np.array(self.data)
            height, width, channels = image.shape
            
            theta = np.radians(degrees)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            x_c = width // 2
            y_c = height // 2
            
            rotated = np.zeros_like(image)
            
            for y in range(height):
                for x in range(width):
                    x_rot = (x - x_c) * cos_theta - (y - y_c) * sin_theta + x_c
                    y_rot = (x - x_c) * sin_theta + (y - y_c) * cos_theta + y_c
                    
                    if 0 <= x_rot < width and 0 <= y_rot < height:
                        rotated[y, x] = image[int(y_rot), int(x_rot)]
                        
            return Image.fromarray(rotated.astype(np.uint8))
        else:
            raise TypeError("Image rotation can only be applied to image data.")

    # x' = x * scale_factor
    # y' = y * scale_factor
    def resize_image(self, scale_factor: float = 1.2) -> np.ndarray:
        if isinstance(self.data, Image.Image):
            image = np.array(self.data)
            height, width, channels = image.shape
            
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            
            resized = np.zeros((new_height, new_width, channels), dtype=np.uint8)
            
            for y in range(new_height):
                for x in range(new_width):
                    x_orig = x / scale_factor
                    y_orig = y / scale_factor
                    
                    if 0 <= x_orig < width and 0 <= y_orig < height:
                        resized[y, x] = image[int(y_orig), int(x_orig)]
                        
            return Image.fromarray(resized)
        else:
            raise TypeError("Image resizing can only be applied to image data.")
        
    # I'(x, y) = α * I(x, y) + (1 - α) * N(x, y)
    def add_image_noise(self, noise_level: int = 25) -> np.ndarray:
        if isinstance(self.data, Image.Image):
            image = np.array(self.data, dtype=np.float32)
            height, width, channels = image.shape
            
            noise = np.random.normal(0, noise_level, (height, width, channels))
            noise = (noise - noise.min()) / (noise.max() - noise.min()) * 255
            noise = noise.astype(np.uint8)
            
            alpha = 0.5
            noisy_image = alpha * image + (1 - alpha) * noise
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            
            return Image.fromarray(noisy_image)
        else:
            raise TypeError("Image noise addition can only be applied to image data.")

    # x' = (x * (1 - perspective_factor) + perspective_factor * width) / width
    # y' = (y * (1 - perspective_factor) + perspective_factor * height) / height
    def change_image_perspective(self, perspective_factor: float = 0.1) -> np.ndarray:
        if isinstance(self.data, Image.Image):
            image = np.array(self.data)
            height, width, channels = image.shape
            
            pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            pts2 = np.float32([[0 + perspective_factor * width, 0], 
                               [width - perspective_factor * width, 0],
                               [0, height], 
                               [width, height]])
            
            M = cv2.getPerspectiveTransform(pts1, pts2)
            warped = cv2.warpPerspective(image, M, (width, height))
            
            return Image.fromarray(warped)
        else:
            raise TypeError("Image perspective change can only be applied to image data.")

    # R' = R * r_factor
    # G' = G * g_factor
    # B' = B * b_factor
    def change_image_color(self, color_factor: float = 0.5) -> np.ndarray:
        if isinstance(self.data, Image.Image):
            image = np.array(self.data, dtype=np.float32)
            height, width, channels = image.shape
            
            r_factor = 1 + (random.random() - 0.5) * color_factor
            g_factor = 1 + (random.random() - 0.5) * color_factor
            b_factor = 1 + (random.random() - 0.5) * color_factor
            
            image[:, :, 0] *= r_factor
            image[:, :, 1] *= g_factor
            image[:, :, 2] *= b_factor
            
            image = np.clip(image, 0, 255).astype(np.uint8)
            
            return Image.fromarray(image)
        else:
            raise TypeError("Image color change can only be applied to image data.")
    
    # x' = width - x - 1
    # y' = y
    def flip_image_horizontally(self) -> np.ndarray:
        if isinstance(self.data, Image.Image):
            image = np.array(self.data)
            height, width, channels = image.shape
            
            flipped = np.zeros_like(image)
            
            for y in range(height):
                for x in range(width):
                    x_flipped = width - x - 1
                    flipped[y, x_flipped] = image[y, x]
                    
            return Image.fromarray(flipped)
        else:
            raise TypeError("Horizontal flip can only be applied to image data.")
        
    # I' = I * contrast_factor
    def change_image_contrast(self, contrast_factor: float = 1.5) -> np.ndarray:
        if isinstance(self.data, Image.Image):
            image = np.array(self.data, dtype=np.float32)
            
            image *= contrast_factor
            image = np.clip(image, 0, 255).astype(np.uint8)
            
            return Image.fromarray(image)
        else:
            raise TypeError("Contrast change can only be applied to image data.")

    def lzw_compress_image(self, image_path, output_dir):
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
    
    def lzw_decompress_images(self, input_dir, output_dir):
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
    
    def save_data(self, output_path, data_to_save):
        if isinstance(data_to_save, Image.Image):
            data_to_save.save(output_path)
        else:
            raise ValueError("No data to save or unsupported data format.")
    
    def augment_image(self, output_file_path):
        augment_index = 1
        aug_methods = [self.rotate_image, self.resize_image, self.add_image_noise, self.change_image_perspective, self.change_image_color, self.flip_image_horizontally, self.change_image_contrast]
        num_augmentations = self.config.NUM_AUGMENTATIONS

        for _ in range(num_augmentations):
            for method in aug_methods:
                try:
                    aug_image = method()
                    aug_output_path = f"{output_file_path}_aug{augment_index}.png"
                    self.save_data(aug_output_path, aug_image)
                    augment_index += 1
                except Exception as e:
                    logging.error(f"Failed to augment image {output_file_path}: {e}")
        return augment_index
    
    def process_files(self):
        for root, _, files in os.walk(self.config.TEMP_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    self.load_data(file_path)
                    output_file_base = os.path.join(self.config.RESULT_DIR, os.path.splitext(file)[0])
                    
                    if isinstance(self.data, Image.Image):
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