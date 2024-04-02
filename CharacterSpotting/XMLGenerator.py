import os
import xml.etree.ElementTree as ET
import cv2
import GlobalConstants as paths

source_dir = paths.words_source_RIMES #From RIMES dataset
transcription_dir = paths.words_transcription_RIMES #From RIMES dataset

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

    words = []

    # Create the subelement <machine-printed-part>
    machine_printed_part = ET.SubElement(form, "machine-printed-part")

    for line_text in lines:
        line = ET.SubElement(machine_printed_part, "machine-print-line")
        line.set("text", line_text)
        words.extend(line_text.split(" "))

    # Create the subelement <handwritten-part>
    handwritten_part = ET.SubElement(form, "handwritten-part")

    # TODO: Add the line and words

    # Create the XML tree
    tree = ET.ElementTree(form)

    # Write the XML tree in a file
    tree.write(id + ".xml", encoding="ISO-8859-1", xml_declaration=True, doctype='<!DOCTYPE form SYSTEM "http://www.iam.unibe.ch/~fki/iamdb/form-metadata.dtd">')
