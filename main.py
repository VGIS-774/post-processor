import cv2
import numpy as np
import random
from definitions import *


def get_mask(img):
    height, width = img.shape
    mask = np.zeros((height, width), np.uint8)

    for y in range(height):
        for x in range(width):
            if img[y, x] > 0:
                mask[y, x] = 255

    return mask


def screen_composite(background, foreground):
    height, width = background.shape
    composite = np.zeros((height, width), np.uint8)

    for y in range(height):
        for x in range(width):
            composite[y, x] = 255 - ((255 - foreground[y, x]) * (255 - background[y, x]) / 255)

    return composite


def random_noise(img, mask, alpha):
    height, width = img.shape
    noisy = np.copy(img)

    for y in range(height):
        for x in range(width):
            if mask[y, x] > 0:
                val = alpha * random.uniform(0, 255) + (1.0 - alpha) * noisy[y, x]
                noisy[y, x] = val

    return noisy


def combined_with_mask(background, foreground, mask):
    height, width = background.shape
    combined = background.copy()

    for y in range(height):
        for x in range(width):
            if mask[y, x] == 255:
                combined[y, x] = foreground[y, x]

    return combined


def main():
    rgb = os.listdir(RGB_PATH)
    segmentation = os.listdir(SEGMENTATION_PATH)

    rgb_images = [cv2.imread(os.path.join(RGB_PATH, img)) for img in rgb]
    segmentation_images = [cv2.imread(os.path.join(SEGMENTATION_PATH, img)) for img in segmentation]

    background = cv2.imread(os.path.join(BACKGROUND_PATH, "background.jpg"))
    background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)

    for x, img in enumerate(rgb_images):
        origin = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(segmentation_images[x], cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        foreground = cv2.bitwise_and(origin, origin, mask=binary)
        blurred = cv2.GaussianBlur(foreground, (3, 3), 0)
        noisy = random_noise(blurred, binary, 0.1)

        combined = screen_composite(background, noisy)

        cv2.imwrite(os.path.join(RESULTS_PATH, "Output" + str(x) + ".jpg"), combined,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # np.set_printoptions(suppress=True)
    #
    # image = cv2.imread("normal.jpg")
    #
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #
    # mask = get_mask(gray)
    #
    # blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    #
    # noisy = random_noise(blurred, mask, 0.4)
    #
    # cv2.imwrite("Ouput.jpg", noisy, [int(cv2.IMWRITE_JPEG_QUALITY), 2])
    #
    # cv2.imshow("Original", noisy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
