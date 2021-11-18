from fsdtorch import inference
import cv2

image_path = "example.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
class_id, class_name, confidence = inference.predict_shape(image_rgb)
print(class_id, class_name, confidence)

cv2.imshow("Example Input", image_rgb)