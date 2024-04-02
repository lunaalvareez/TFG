
import os
import xml.etree.ElementTree as ET
import cv2
import GlobalConstants as paths


compressed_image_path = paths.source + "wordsCompressed"
xml_file_path = paths.source + "xml"
buffer = 0.1



def extract_word_data(xml_file):
    idl = []
    textl = []
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        words = root.findall('.//word')
        for word in words:
            word_id = word.attrib.get('id')
            word_text = word.attrib.get('text')
            if word_id and word_text:
                #print(f"File: {os.path.basename(xml_file)}, ID: {word_id}, Text: {word_text}")
                idl.append(word_id)
                textl.append(word_text)
            else:
                print(f"Missing data in {os.path.basename(xml_file)}")
    except ET.ParseError as e:
        print(f"Error parsing {xml_file}: {e}")
    return idl,textl





def str_pixel_width_calculator(string_in):
    char_widths = [0] * len(string_in)

    for ii in range(len(string_in)):
        letter_width = 0

        char = string_in[ii]
        #print(char)
        if char in ['.', ',', "'"]:
            letter_width = 8
        elif char in ["0","1","2","3","4","5","6","7","8","9"]:
            letter_width = 15
        elif char in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']:
            letter_width = 23
        elif char in ['I', 'J', '-']:
            letter_width = 15 
        elif char == 'M':
            letter_width = 26 
        elif char == 'W':
            letter_width = 39
        elif char in ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'k', 'n', 'o', 'p', 'q', 's', 'u', 'v', 'x', 'y', 'z']:
            letter_width = 15  
        elif char in ['i', 'l']:
            letter_width = 8 
        elif char in ['r', 't', 'j', 'f']:
            letter_width = 12 
        elif char in ['m', 'w']:
            letter_width = 24 
        elif char in [';', '!', '?', '(', ')', '"', ":", "&", "#", "*", " ", '/', "+"]:
            letter_width = 15

        char_widths[ii] = letter_width

    char_widths = [width / sum(char_widths) for width in char_widths]  # normalize lengths
    return char_widths

def add_width_buffer(char_widths):
    buffer_char_widths = char_widths[:]
    for width in range(len(char_widths)):
        if width == 0 and width == len(char_widths):
            buffer_char_widths[width] = 1 #To fix single letters
        elif width == 0 or width == len(char_widths):
            buffer_char_widths[width] = char_widths[width] * 1.05 
        else:
            buffer_char_widths[width] = char_widths[width] * 1.1
    return buffer_char_widths


yolo_classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '?']



for filename in os.listdir(xml_file_path):
    if filename.endswith('.xml'):
        file_path = os.path.join(xml_file_path, filename)
        idl, textl = extract_word_data(file_path)

        for i, id in enumerate(idl):
            text = textl[i]
            picture_path = os.path.join(compressed_image_path, f"{id}.png")
            char_widths = str_pixel_width_calculator(text)
            image = cv2.imread(picture_path)

            if image is not None:
                char_widths_with_buffer = add_width_buffer(char_widths)
                char_positions = [0] + [sum(char_widths[:j]) for j in range(1, len(char_widths))]
                yolo_annotation_path = picture_path.replace(".png", ".txt")

                with open(yolo_annotation_path, "a") as yolo_annotation_file:
                    for j, char in enumerate(text):
                        char_width = char_widths[j]
                        x_center = char_positions[j] + (char_width / 2) if j < len(text) - 1 else 1 - (char_widths_with_buffer[j] / 2)
                        y_center = 0.5  # Assuming center alignment vertically
                        box_width = char_widths_with_buffer[j]
                        box_height = 1.0  # Assuming the character spans the full height of the image
                        class_label = yolo_classes.index(char)

                        yolo_annotation_file.write(f"{class_label} {x_center} {y_center} {box_width} {box_height}\n")
            else:
                print(f"Could not read image: {picture_path}")



