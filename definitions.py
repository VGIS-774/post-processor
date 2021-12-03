import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Project Root

BACKGROUND_PATH = os.path.join(ROOT_DIR, "background\\")  # Textures path

RGB_PATH = os.path.join(ROOT_DIR, "RGB\\")  # Output images path

SEGMENTATION_PATH = os.path.join(ROOT_DIR, "segmentation\\")  # Output images path

RESULTS_PATH = os.path.join(ROOT_DIR, "result\\")  # Output images path

FILEBROWSER_PATH = os.path.join(os.getenv('WINDIR'), 'explorer.exe')