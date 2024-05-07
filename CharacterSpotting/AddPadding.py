import cv2
import os
import GlobalConstants as paths

compressed_image_path = paths.source + "sentencesCompressed40K"

def resize_and_pad_image(image, target_size):
    
    # Redimensiona la imagen manteniendo la proporción original
    resized_image = cv2.resize(image, target_size)
    
    # Calcula el relleno necesario
    pad_width = target_size[0] - resized_image.shape[1]
    pad_height = target_size[1] - resized_image.shape[0]
    
    # Agrega relleno a la imagen con píxeles blancos
    padded_image = cv2.copyMakeBorder(resized_image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=255)
    
    return padded_image

target_size = (1080,1080)
for filename in os.listdir(compressed_image_path):
    if filename.endswith(".jpg"):  # check if the file is an image
        image_path = os.path.join(compressed_image_path, filename)
        image = cv2.imread(image_path)
        padded_image = resize_and_pad_image(image, target_size)
        # You can now use `padded_image` or save it to a file

