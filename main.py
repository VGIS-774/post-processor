import cv2
import numpy as np
import random
import subprocess
from tqdm import tqdm
import argparse
from definitions import *


# Opens windows explorer's path
def explore(path):
    # explorer would choke on forward slashes
    path = os.path.normpath(path)

    if os.path.isdir(path):
        subprocess.run([FILEBROWSER_PATH, path])
    elif os.path.isfile(path):
        subprocess.run([FILEBROWSER_PATH, '/select,', os.path.normpath(path)])


# Screen compositing
def screen_composite(background, foreground):
    height, width = background.shape
    composite = np.zeros((height, width), np.uint8)

    for y in range(height):
        for x in range(width):
            composite[y, x] = 255 - ((255 - foreground[y, x]) * (255 - background[y, x]) / 255)

    return composite


# Random noise
def random_noise(img, mask, alpha):
    height, width = img.shape
    noisy = np.copy(img)

    for y in range(height):
        for x in range(width):
            if mask[y, x] > 0:
                val = alpha * random.uniform(0, 255) + (1.0 - alpha) * noisy[y, x]
                noisy[y, x] = val

    return noisy


# Combine foreground with background using a mask
def combined_with_mask(background, foreground, mask):
    height, width = background.shape
    combined = background.copy()

    for y in range(height):
        for x in range(width):
            if mask[y, x] == 255:
                combined[y, x] = foreground[y, x]

    return combined


def write_video(path, resolution, label, arr, color):
    # Create the video writer object
    out = cv2.VideoWriter(os.path.join(path, "Output" + label + ".mp4"),
                          cv2.VideoWriter_fourcc(*'mp4v'), 1, (resolution["width"], resolution["height"]),
                          color)

    # Write all of the images from an array as a video
    for image in arr:
        out.write(image)

    out.release()


def main():
    """ This part is for running the script in the console if you can somehow find directive to all the modules such
    as cv2 """

    # parser = argparse.ArgumentParser(description="Creates a dataset based on entered training set percentage")
    #
    # parser.add_argument('integer', metavar='T', type=int, nargs='+',
    #                     help='an integer for specifying training set percentage')
    #
    # args = parser.parse_args()
    #
    # if not 0 < args.integer[0] < 100:
    #     training = 80
    # else:
    #     training = args.integer[0]

    training = 80
    split = (training, 100 - training)  # Determine the split for validation and training data

    rgb = os.listdir(RGB_PATH)  # List of all rgb image directories
    rgb.sort(key=len)  # Sort the image directory list

    segmentation = os.listdir(SEGMENTATION_PATH)  # List of all segmentation mask directories
    segmentation.sort(key=len)  # Sort the image directory list

    background_videos = os.listdir(BACKGROUND_PATH)  # List of all background video directories

    """ Read all the images as grayscale """
    rgb_images = [cv2.cvtColor(cv2.imread(os.path.join(RGB_PATH, img)), cv2.COLOR_RGB2GRAY) for img in rgb]
    segmentation_images = [cv2.imread(os.path.join(SEGMENTATION_PATH, img)) for img in
                           segmentation]

    height, width = rgb_images[0].shape
    resolution = {"width": width, "height": height}

    image_buffer = list()

    """ Pick a random video and pick a random frame """
    video = cv2.VideoCapture(os.path.join(BACKGROUND_PATH, random.choice(background_videos)))
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)  # Number of frames
    video.set(cv2.CAP_PROP_POS_FRAMES, random.randrange(frames - 3))  # Set to a random frame in a video

    for img_count in tqdm(range(len(rgb_images)), desc="Progress"):

        """ Read the current frame of the video """
        ret, background = video.read()
        background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)

        origin = rgb_images[img_count]  # Load the original rgb image
        seg = cv2.cvtColor(segmentation_images[img_count], cv2.COLOR_RGB2GRAY)  # Load the segmentation mask
        _, mask = cv2.threshold(seg, 0, 255, cv2.THRESH_BINARY)  # Create an alpha mask out of segmentation mask

        foreground = cv2.bitwise_and(origin, origin, mask=mask)  # Extract the foreground from the original image
        blurred = cv2.GaussianBlur(foreground, (1, 1), 0)  # Blur the extracted foreground
        noisy = random_noise(blurred, mask, 0.1)  # Apply random noise to the foreground

        _, arr = cv2.imencode(".jpg", noisy, [int(cv2.IMWRITE_JPEG_QUALITY), 20])  # Encode the foreground with the JPEG
        compressed = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)  # Decode the image

        combined = screen_composite(background, compressed)  # Combine the foreground with the background

        image_buffer.append(combined)  # Add the current combined image to the buffer

        """ Write every 3 frames as a video """
        if (img_count + 1) % 3 == 0:

            """ Pick a random video and pick a random frame """
            video = cv2.VideoCapture(os.path.join(BACKGROUND_PATH, random.choice(background_videos)))
            frames = video.get(cv2.CAP_PROP_FRAME_COUNT)  # Number of frames
            video.set(cv2.CAP_PROP_POS_FRAMES, random.randrange(frames))  # Set to a random frame in a video

            if img_count > len(rgb_images) * (split[0] / 100):
                write_video(ORIGIN_VALIDATION_PATH, resolution, str(int(img_count / 3)), image_buffer, False)
                write_video(SEGMENTATION_VALIDATION_PATH, resolution, str(int(img_count / 3)),
                            segmentation_images[img_count - 2: img_count + 1], True)

            else:
                write_video(ORIGIN_TRAIN_PATH, resolution, str(int(img_count / 3)), image_buffer, False)
                write_video(SEGMENTATION_TRAIN_PATH, resolution, str(int(img_count / 3)),
                            segmentation_images[img_count - 2: img_count + 1], True)

            image_buffer = []  # Reset image buffer

    print(f"\nSaved to: {RESULTS_PATH}, with {split[0]} to {split[1]} split")
    explore(RESULTS_PATH)


if __name__ == '__main__':
    main()
