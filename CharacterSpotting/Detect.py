import cv2
import numpy as np
import torch
import GlobalConstants as paths
import Yolov8Model as Yolov8Model

weights_file = "yolov8_weights.pt"
model = Yolov8Model.YOLOv8()
model.load_state_dict(torch.load(weights_file))
model.eval()

imagePaths = paths.words_source_RIMES
# Get the labels from the yolov3.txt file
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '?']

# Function to get bounding boxes and detected classes
def get_objects(image):
    height, width, _ = image.shape

    # Convert image to PyTorch tensor
    image_tensor = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).float()

    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
    
    boxes = []
    confidences = []
    class_ids = []

    for detection in outputs:
        for obj in detection:
            scores = obj[5:]
            class_id = torch.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

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

