from fsdtorch import inference
import cv2
<<<<<<< HEAD

=======
>>>>>>> 5930537cead9a64263034f4b9227b95cd40443e1
image_path = "example.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
class_id, class_name, confidence = inference.predict_shape(image_rgb)
<<<<<<< HEAD
print(class_id, class_name, confidence)

cv2.imshow("Example Input", image_rgb)
=======
print(class_name, confidence)
>>>>>>> 5930537cead9a64263034f4b9227b95cd40443e1
