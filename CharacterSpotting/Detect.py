import cv2
import numpy as np
import matplotlib.pyplot as plt
import GlobalConstants as paths

imagePaths = paths.words_source_RIMES
# Get the labels from the yolov3.txt file
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '?']

# Load YOLOv3 model
net = cv2.dnn.readNet("yolo.weights", "yolo.cfg")

# Function to get bounding boxes and detected classes
def get_objects(image):
    height, width, _ = image.shape

    # Preprocess the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layers
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x, y = int(center_x - w/2), int(center_y - h/2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids

# Function to draw bounding boxes on the image
# def draw_boxes(image, boxes, confidences, class_ids):
#     colors = np.random.uniform(0, 255, size=(len(labels), 3))

#     for i in range(len(boxes)):
#         x, y, w, h = boxes[i]
#         label = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
#         color = colors[class_ids[i]]
#         cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
#         cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Lists to store predictions for all images
true_labels_all = []
predicted_labels_all = []


for img in imagePaths:
    image = cv2.imread(img)
    trueLabel = labels.copy()

    boxes, confidences, class_ids = get_objects(image)
    
    if len(boxes) > 0:
        max_confidence_index = np.argmax(confidences)
        most_confident_class_id = class_ids[max_confidence_index]
        predicted_label = labels[most_confident_class_id]

        true_labels_all.append(trueLabel[most_confident_class_id])
        predicted_labels_all.append(predicted_label)

print("SUCCESS!")
       
        # if count >= 550:
        #     # Draw bounding box on the image
        #     draw_boxes(image, [boxes[max_confidence_index]], [confidences[max_confidence_index]], [class_ids[max_confidence_index]])
        #     # Show the resulting image with Matplotlib
        #     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #     plt.axis('off')
        #     plt.show()

