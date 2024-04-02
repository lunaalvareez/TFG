import os
import xml.etree.ElementTree as ET
import cv2
import GlobalConstants as paths

source_dir = paths.words_source_RIMES #From RIMES dataset
transcription_dir = paths.words_transcription_RIMES #From RIMES dataset
xml_dir = "/xml"

for filename in os.listdir(transcription_dir):
    # Create the root element <form>
    id = filename.split(".")[0]
    form = ET.Element("form")
    form.set("id", id)
    img = cv2.imread(source_dir + "/" + id + ".jpg")
    form.set("height", img.height)
    form.set("width", img.width)

    file_path = os.path.join(transcription_dir, filename)
    with open(file_path, 'r') as f:
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

    # TODO: Add the line and words

    for i, line_text in lines:
        line = ET.SubElement(handwritten_part, "line")
        line.set("id", f"{id}-{i:02}")
        line.set("text", line_text)
        words = line_text.split(" ")
        for j, word_text in words:
            word = ET.SubElement(line, "word")
            word.set("id", f"{id}-{i:02}-{j:02}")
            word.set("text", word_text)

    # Create the XML tree
    tree = ET.ElementTree(form)

    # Write the XML tree in a file
    xml_path = os.path.join(xml_dir, id+".xml")
    tree.write(xml_path, encoding="ISO-8859-1", xml_declaration=True, doctype='<!DOCTYPE form SYSTEM "http://www.iam.unibe.ch/~fki/iamdb/form-metadata.dtd">')
