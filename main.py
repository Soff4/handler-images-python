# Ці рядки імпортують необхідні модулі та бібліотеки, які будуть використовуватися у коді. 
# Наприклад, os для роботи з файловою системою, pandas для обробки табличних даних, numpy для чисельних обчислень, 
# PIL для роботи з зображеннями, random для генерації випадкових чисел, zipfile для роботи з ZIP-архівами, 
# shutil для операцій з файловою системою, pickle для серіалізації об'єктів, glob для пошуку файлів за шаблоном, 
# LZW для стиснення/розпакування даних за допомогою алгоритму LZW, а також logging для ведення логів.
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

# Ця інструкція налаштовує модуль logging для виведення повідомлень з рівнем INFO та вище у певному форматі, 
# який включає час, рівень повідомлення та саме повідомлення.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Тут визначається клас Config, який містить константи, що використовуються в подальшому коді. 
# TEMP_DIR - назва директорії для тимчасового розпакування архіву, 
# RESULT_DIR - назва директорії для збереження результатів, 
# NUM_AUGMENTATIONS - кількість модифікацій для кожного зображення.
class Config:
    TEMP_DIR = 'temp_extracted'
    RESULT_DIR = 'results'
    NUM_AUGMENTATIONS = 3

# Визначається клас DataAugmentation, який є основним класом для обробки даних. 
# У конструкторі ініціалізуються поля 
#     archive_dir (директорія, де розташований архів), 
#     data (тимчасове збереження даних), 
#     config (об'єкт класу Config), lzw (екземпляр класу LZW для стиснення/розпакування даних), 
# а також викликається метод _find_archive() для пошуку архіву в зазначеній директорії.
class DataAugmentation:
    def __init__(self, archive_dir: str, config: Config):
        self.archive_dir = archive_dir
        self.data = None
        self.config = config
        self.lzw = LZW()
        self.archive_path = self._find_archive()

    # Метод _find_archive() шукає в директорії archive_dir файл з розширенням .zip, назва якого закінчується на _images.zip. 
    # Якщо такий файл знайдений, метод повертає повний шлях до нього. Якщо файл не знайдений, генерується виняток FileNotFoundError.
    def _find_archive(self) -> str:
        for file in os.listdir(self.archive_dir):
            if file.endswith("_images.zip"):
                return os.path.join(self.archive_dir, file)
        raise FileNotFoundError("No archive matching the pattern '*_images.zip' found.")

    # Метод extract_archive() перевіряє, чи є знайдений файл дійсним ZIP-архівом. 
    # Якщо так, він розпаковує вміст архіву в тимчасову директорію self.config.TEMP_DIR.
    def extract_archive(self):
        if not zipfile.is_zipfile(self.archive_path):
            raise ValueError("The provided file is not a ZIP archive.")
        
        with zipfile.ZipFile(self.archive_path, 'r') as zip_ref:
            zip_ref.extractall(self.config.TEMP_DIR)
    
    # Метод load_data(file_path) завантажує дані з файлу за вказаним шляхом file_path. 
    # Якщо файл має розширення .csv, він завантажується як DataFrame за допомогою pd.read_csv(). 
    # Якщо файл є зображенням (з розширеннями .png, .jpg або .jpeg), він завантажується як об'єкт Image за допомогою Image.open(). 
    # В іншому випадку, self.data встановлюється в None.
    def load_data(self, file_path: str):
        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.data = Image.open(file_path)
        else:
            self.data = None
    
    # Метод normalize() нормалізує дані, якщо self.data є об'єктом DataFrame. 
    # Нормалізація виконується шляхом віднімання середнього значення від кожного елемента та ділення на стандартне відхилення. 
    # Якщо self.data не є DataFrame, генерується виняток TypeError.
    def normalize(self):
        if isinstance(self.data, pd.DataFrame):
            self.data = (self.data - self.data.mean()) / self.data.std()
        else:
            raise TypeError("Normalization can only be applied to CSV data.")
    
    # Метод add_noise(noise_level) додає випадковий шум до даних, якщо self.data є об'єктом DataFrame. 
    # Шум генерується як випадковий масив чисел з нормального розподілу, помножений на noise_level. 
    # Якщо self.data не є DataFrame, генерується виняток TypeError.
    def add_noise(self, noise_level: float = 0.01):
        if isinstance(self.data, pd.DataFrame):
            noise = np.random.randn(*self.data.shape) * noise_level
            self.data += noise
        else:
            raise TypeError("Noise addition can only be applied to CSV data.")
    
    # Метод random_drop(drop_fraction) випадково видаляє рядки з даних, якщо self.data є об'єктом DataFrame. 
    # Кількість рядків, які потрібно видалити, визначається як int(len(self.data) * drop_fraction). 
    # Після видалення рядків, індекси DataFrame перезаписуються за допомогою reset_index(drop=True). 
    # Якщо self.data не є DataFrame, генерується виняток TypeError.
    def random_drop(self, drop_fraction: float = 0.1):
        if isinstance(self.data, pd.DataFrame):
            rows_to_drop = random.sample(range(len(self.data)), int(len(self.data) * drop_fraction))
            self.data = self.data.drop(rows_to_drop).reset_index(drop=True)
        else:
            raise TypeError("Row dropping can only be applied to CSV data.")
    
    # Метод rotate_image(degrees) обертає зображення на вказаний кут degrees, якщо self.data є об'єктом Image.Image. 
    # Якщо self.data не є зображенням, генерується виняток TypeError.
    def rotate_image(self, degrees: int = 45) -> Image.Image:
        if isinstance(self.data, Image.Image):
            return self.data.rotate(degrees)
        else:
            raise TypeError("Image rotation can only be applied to image data.")
    
    # Метод resize_image(scale_factor) змінює розмір зображення, множачи його ширину та висоту на scale_factor, якщо self.data є об'єктом Image.Image. 
    # Якщо self.data не є зображенням, генерується виняток TypeError.
    def resize_image(self, scale_factor: float = 1.2) -> Image.Image:
        if isinstance(self.data, Image.Image):
            new_size = (int(self.data.width * scale_factor), int(self.data.height * scale_factor))
            return self.data.resize(new_size)
        else:
            raise TypeError("Image resizing can only be applied to image data.")
    
    # Метод add_image_noise(noise_level) додає шум до зображення, якщо self.data є об'єктом Image.Image. 
    # Шум генерується як масив випадкових чисел з нормального розподілу з середнім 0 та стандартним відхиленням noise_level. 
    # Потім він нормалізується до діапазону [0, 255] і перетворюється на цілочисельний масив. 
    # Далі, шум перетворюється на об'єкт Image.Image, і змішується з оригінальним зображенням за допомогою методу Image.blend() 
    # з коефіцієнтом змішування 0.5. Якщо self.data не є зображенням, генерується виняток TypeError.
    def add_image_noise(self, noise_level: int = 25) -> Image.Image:
        if isinstance(self.data, Image.Image):
            noise = np.random.normal(0, noise_level, (self.data.height, self.data.width, 3))
            noise = (noise - noise.min()) / (noise.max() - noise.min()) * 255
            noise = noise.astype(np.uint8)
            noise_image = Image.fromarray(noise)
            return Image.blend(self.data, noise_image, 0.5)
        else:
            raise TypeError("Image noise addition can only be applied to image data.")
    
    # Метод lzw_compress_image(image_path, output_dir) стискає зображення за вказаним шляхом image_path за допомогою алгоритму LZW 
    # і зберігає стиснуті дані в директорію output_dir. Спочатку перевіряється існування файлу зображення. 
    # Потім зображення відкривається, та перетворюється на байтову послідовність за допомогою image.tobytes(). 
    # Розмір зображення зберігається в image_size. Байтова послідовність декодується з latin1 та стискається методом self.lzw.compress(). 
    # Ім'я файлу для збереження стиснутих даних генерується шляхом додавання суфікса _compressed.lzw до імені вихідного файлу. 
    # Стиснуті дані та розмір зображення серіалізуються за допомогою pickle.dump() і зберігаються у вказаному вихідному шляху. 
    # Повідомлення про успішне збереження стиснутого зображення записується у логи.
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
    
    # Метод lzw_decompress_images(input_dir, output_dir) розпаковує стиснуті зображення з директорії input_dir та зберігає їх у директорію output_dir. 
    # Спочатку знаходяться всі файли у input_dir з суфіксом _compressed.lzw за допомогою glob.glob(). 
    # Для кожного знайденого стиснутого файлу, відкривається та десеріалізуються розмір зображення та стиснуті дані за допомогою pickle.load(). 
    # Стиснуті дані розпаковуються методом self.lzw.decompress() та кодуються назад у байтову послідовність за допомогою encode('latin1'). 
    # Потім створюється новий об'єкт Image.Image з розпакованих даних та розміру зображення. 
    # Ім'я вихідного файлу генерується шляхом заміни _compressed на _decompressed в імені вихідного файлу та додавання розширення .png. 
    # Розпаковане зображення зберігається за вказаним вихідним шляхом за допомогою image.save(). 
    # Повідомлення про успішне збереження розпакованого зображення записується у логи.
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
    
    # Метод save_data(output_path, data_to_save) зберігає дані data_to_save у вказаному вихідному шляху output_path. 
    # Якщо data_to_save є об'єктом pd.DataFrame, він зберігається у форматі CSV за допомогою методу to_csv() без збереження індексів рядків. 
    # Якщо data_to_save є об'єктом Image.Image, він зберігається як зображення за допомогою методу save(). 
    # Якщо data_to_save не є ні DataFrame, ні Image, генерується виняток ValueError.
    def save_data(self, output_path: str, data_to_save: Union[pd.DataFrame, Image.Image]):
        if isinstance(data_to_save, pd.DataFrame):
            data_to_save.to_csv(output_path, index=False)
        elif isinstance(data_to_save, Image.Image):
            data_to_save.save(output_path)
        else:
            raise ValueError("No data to save or unsupported data format.")
    
    # Метод augment_image(output_file_path) виконує серію модифікацій зображення self.data та зберігає результати у вихідних файлах. 
    # Список методів модифікацій aug_methods містить rotate_image, resize_image та add_image_noise. 
    # Цикл виконується self.config.NUM_AUGMENTATIONS разів (за замовчуванням, 3 рази), і для кожної ітерації застосовуються всі методи з aug_methods. 
    # Для кожного успішно виконаного методу модифіковане зображення зберігається з іменем {output_file_path}_augN.png, 
    # де N - порядковий номер модифікації. Якщо під час виконання методу трапляється виняток, він записується у логи за допомогою logging.error(). 
    # Метод повертає кількість успішно збережених модифікованих зображень.
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
    
    # Метод augment_csv(output_file_path) виконує серію модифікацій DataFrame self.data та зберігає результати у вихідних файлах. 
    # Список методів модифікацій aug_methods містить normalize, add_noise та random_drop. 
    # Для кожного методу з aug_methods викликається відповідна функція модифікації, і якщо вона виконується успішно, 
    # модифіковані дані зберігаються у файл {output_file_path}_aug.csv. Якщо під час виконання методу трапляється виняток, 
    # він записується у логи за допомогою logging.error().
    def augment_csv(self, output_file_path: str):
        aug_methods = [self.normalize, self.add_noise, self.random_drop]
        for method in aug_methods:
            try:
                method()
                aug_output_path = f"{output_file_path}_aug.csv"
                self.save_data(aug_output_path, self.data)
            except Exception as e:
                logging.error(f"Failed to augment CSV {output_file_path}: {e}")
    
    # Метод process_files() ітерується по всіх файлах у тимчасовій директорії self.config.TEMP_DIR (куди було розпаковано архів) за допомогою os.walk(). 
    # Для кожного файлу викликається метод load_data() для завантаження його вмісту. 
    # Якщо дані були завантажені успішно, генерується базовий шлях для збереження модифікованих даних у директорії self.config.RESULT_DIR. 
    # Залежно від типу завантажених даних (DataFrame або Image) викликається відповідний метод для модифікації та збереження даних: augment_csv() 
    # ля CSV-файлів або augment_image() для зображень. Додатково, якщо завантажені дані є зображенням, 
    # виконується стиснення зображення методом lzw_compress_image(). 
    # Якщо під час обробки файлу трапляється виняток, він записується у логи за допомогою logging.error().
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

    # Метод create_result_archive() створює ZIP-архів results.zip, який містить всі файли з директорії self.config.RESULT_DIR та її підкаталогів. 
    # Цей архів буде містити результати модифікації вихідних файлів.
    def create_result_archive(self):
        with zipfile.ZipFile('results.zip', 'w') as zipf:
            for root, _, files in os.walk(self.config.RESULT_DIR):
                for file in files:
                    zipf.write(os.path.join(root, file),
                               os.path.relpath(os.path.join(root, file), self.config.RESULT_DIR))
    
    # Метод clean_up() видаляє тимчасову директорію self.config.TEMP_DIR та директорію результатів self.config.RESULT_DIR разом з 
    # їхнім вмістом за допомогою shutil.rmtree().
    def clean_up(self):
        shutil.rmtree(self.config.TEMP_DIR)
        shutil.rmtree(self.config.RESULT_DIR)

    # Метод augment_data_from_archive() є основним методом, який керує процесом модифікації даних з архіву.
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