import unittest
import os
import shutil
import pandas as pd
from PIL import Image
import numpy as np
from main import DataAugmentation, Config
import zipfile
import glob

class TestDataAugmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = 'test_dir'
        os.makedirs(cls.test_dir, exist_ok=True)
        
        cls.test_csv_path = os.path.join(cls.test_dir, 'test_data.csv')
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        df.to_csv(cls.test_csv_path, index=False)
        
        cls.test_image_path = os.path.join(cls.test_dir, 'test_image.png')
        image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        image.save(cls.test_image_path)
        
        cls.test_zip_path = os.path.join(cls.test_dir, 'test_images.zip')
        with zipfile.ZipFile(cls.test_zip_path, 'w') as zipf:
            zipf.write(cls.test_csv_path, os.path.basename(cls.test_csv_path))
            zipf.write(cls.test_image_path, os.path.basename(cls.test_image_path))

        cls.test_pattern_zip_path = os.path.join(cls.test_dir, 'idchat_images.zip')
        shutil.copyfile(cls.test_zip_path, cls.test_pattern_zip_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        self.config = Config()
        self.augmenter = DataAugmentation(archive_dir=self.test_dir, config=self.config)
    
    def test_find_archive(self):
        archive_path = self.augmenter._find_archive()
        self.assertEqual(archive_path, self.test_pattern_zip_path)
    
    def test_extract_archive(self):
        self.augmenter.extract_archive()
        extracted_files = os.listdir(self.config.TEMP_DIR)
        self.assertIn('test_data.csv', extracted_files)
        self.assertIn('test_image.png', extracted_files)
    
    def test_load_data_csv(self):
        self.augmenter.load_data(self.test_csv_path)
        self.assertIsInstance(self.augmenter.data, pd.DataFrame)
    
    def test_load_data_image(self):
        self.augmenter.load_data(self.test_image_path)
        self.assertIsInstance(self.augmenter.data, Image.Image)
    
    def test_normalize_csv(self):
        self.augmenter.load_data(self.test_csv_path)
        self.augmenter.normalize()
        self.assertTrue(np.allclose(self.augmenter.data.mean(), 0))
    
    def test_add_noise_csv(self):
        self.augmenter.load_data(self.test_csv_path)
        self.augmenter.add_noise(noise_level=0.1)
        self.assertNotEqual(self.augmenter.data.values[0, 0], 1)
    
    def test_random_drop_csv(self):
        self.augmenter.load_data(self.test_csv_path)
        initial_length = len(self.augmenter.data)
        self.augmenter.random_drop(drop_fraction=0.5)
        self.assertLess(len(self.augmenter.data), initial_length)
    
    def test_rotate_image(self):
        self.augmenter.load_data(self.test_image_path)
        rotated_image = self.augmenter.rotate_image(degrees=90)
        self.assertEqual(rotated_image.size, self.augmenter.data.size)
    
    def test_resize_image(self):
        self.augmenter.load_data(self.test_image_path)
        resized_image = self.augmenter.resize_image(scale_factor=2)
        self.assertEqual(resized_image.size, (200, 200))
    
    def test_add_image_noise(self):
        self.augmenter.load_data(self.test_image_path)
        noisy_image = self.augmenter.add_image_noise(noise_level=50)
        self.assertEqual(noisy_image.size, self.augmenter.data.size)
    
    def test_lzw_compress_image(self):
        output_dir = 'output_dir'
        os.makedirs(output_dir, exist_ok=True)
        self.augmenter.lzw_compress_image(self.test_image_path, output_dir)
        compressed_files = glob.glob(os.path.join(output_dir, '*_compressed.lzw'))
        self.assertTrue(len(compressed_files) > 0)
        shutil.rmtree(output_dir)
    
    def test_lzw_decompress_images(self):
        output_dir = 'output_dir'
        os.makedirs(output_dir, exist_ok=True)
        self.augmenter.lzw_compress_image(self.test_image_path, output_dir)
        self.augmenter.lzw_decompress_images(output_dir, output_dir)
        decompressed_files = glob.glob(os.path.join(output_dir, '*_decompressed.png'))
        self.assertTrue(len(decompressed_files) > 0)
        shutil.rmtree(output_dir)
    
    def test_augment_data_from_archive(self):
        self.augmenter.augment_data_from_archive()
        result_archive_path = 'results.zip'
        self.assertTrue(os.path.exists(result_archive_path))
        os.remove(result_archive_path)

if __name__ == '__main__':
    unittest.main()
