import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Project Root

BACKGROUND_PATH = os.path.join(ROOT_DIR, "background\\")  # Textures path

RGB_PATH = os.path.join(ROOT_DIR, "RGB\\")  # Input rgb images path

SEGMENTATION_PATH = os.path.join(ROOT_DIR, "segmentation\\")  # Input segmentation masks path

RESULTS_PATH = os.path.join(ROOT_DIR, "result\\")  # Output images path

ORIGIN_RESULT_PATH = os.path.join(RESULTS_PATH, "color\\")

SEGMENTATION_RESULT_PATH = os.path.join(RESULTS_PATH, "segmentation\\")

SEGMENTATION_TRAIN_PATH = os.path.join(SEGMENTATION_RESULT_PATH, "train\\")

SEGMENTATION_VALIDATION_PATH = os.path.join(SEGMENTATION_RESULT_PATH, "validation\\")

ORIGIN_TRAIN_PATH = os.path.join(ORIGIN_RESULT_PATH, "train\\")

ORIGIN_VALIDATION_PATH = os.path.join(ORIGIN_RESULT_PATH, "validation\\")

FILEBROWSER_PATH = os.path.join(os.getenv('WINDIR'), 'explorer.exe')  # Path to windows explorer
