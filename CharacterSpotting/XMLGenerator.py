import os
import xml.etree.ElementTree as ET
import xml.dom.minidom
import cv2
import GlobalConstants as paths

source_dir = paths.words_source_RIMES #From RIMES dataset
transcription_dir = paths.words_transcription_RIMES #From RIMES dataset
xml_dir = "xml"

for filename in os.listdir(transcription_dir):
    id = filename.split(".")[0]
    image_path = os.path.join(source_dir, id + ".jpg")
    if os.path.exists(image_path):

        # Create the root element <form>
        form = ET.Element("form")
        form.set("id", id)
        img = cv2.imread(image_path)
        form.set("height", str(img.shape[0]))    
        form.set("width",str(img.shape[1]))
        file_path = os.path.join(transcription_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # words = []

        # Create the subelement <machine-printed-part>
        machine_printed_part = ET.SubElement(form, "machine-printed-part")

        for line_text in lines:
            print_line = ET.SubElement(machine_printed_part, "machine-print-line")
            print_line.set("text", line_text)
            # words.extend(line_text.split(" "))

        # Create the subelement <handwritten-part>
        handwritten_part = ET.SubElement(form, "handwritten-part")

        for i, line_text in enumerate(lines, start=0):
            line = ET.SubElement(handwritten_part, "line")
            line.set("id", f"{id}-{i:02}")
            line.set("text", line_text)
            words = line_text.split(" ")
            for j, word_text in enumerate(words, start=0):
                word = ET.SubElement(line, "word")
                word.set("id", f"{id}-{i:02}-{j:02}")
                word.set("text", word_text)

        # Create the XML tree
        tree = ET.ElementTree(form)
        xml_str = ET.tostring(form, 'utf-8')

        dom = xml.dom.minidom.parseString(xml_str)
        pretty_xml_str = dom.toprettyxml(indent="   ")

        # Write the XML tree in a file
        xml_path = os.path.join(xml_dir, id+".xml")
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f.write('<!DOCTYPE form SYSTEM "http://www.iam.unibe.ch/~fki/iamdb/form-metadata.dtd">\n')
            f.write(pretty_xml_str)

print("Success!")