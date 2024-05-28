import cv2
import os
import GlobalConstants as paths

sentencesCompressedOriginal = paths.source + "sentencesCompressedOriginal"
sentencesCompressedPadded = paths.source + "sentencesCompressedPadded"

def resize_and_pad_image(image, target_size):
    # Calculate the aspect ratio of the original image
    aspect_ratio = image.shape[1] / image.shape[0]

    # Calculate the target height based on the target width and original aspect ratio
    target_height = int(target_size[0] / aspect_ratio)
    target_width = target_size[0]

    # If the target height is greater than the target size, adjust it and calculate the target width
    if target_height > target_size[1]:
        target_height = target_size[1]
        target_width = int(target_height * aspect_ratio)

    # Resize the image to the target width and calculated height
    resized_image = cv2.resize(image, (target_width, target_height))

    # Calculate the necessary padding
    pad_height = target_size[1] - resized_image.shape[0]
    pad_width = target_size[0] - resized_image.shape[1]

    # Add padding to the image with white pixels
    padded_image = cv2.copyMakeBorder(resized_image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[255, 255, 255])   

    return padded_image

target_size = (1080,1080)

for filename in os.listdir(sentencesCompressedOriginal):
    if filename.endswith(".jpg"):  # check if the file is an image
        image_path = os.path.join(sentencesCompressedOriginal, filename)
        padded_path = os.path.join(sentencesCompressedPadded, filename)
        image = cv2.imread(image_path)
        padded_image = resize_and_pad_image(image, target_size)
        cv2.imwrite(padded_path, padded_image)  # overwrite the original image with the padded image

print("Finished")