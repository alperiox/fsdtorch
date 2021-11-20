from fsdtorch import inference
import cv2
image_path = "examples\example.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
outputs = inference.predict_shape(image_rgb)
print("Found %d faces in the image!"%len(outputs))
for output in outputs:
    bbox = output['bbox']
    x0, y0, x1, y1 = map(int, bbox)
    image_rgb = cv2.rectangle(image_rgb, (x0, y0), (x1,y1), (255, 0, 0))
    print(output['class_name'], ":", output['confidence'])

cv2.imshow("Example Input", image_rgb)
cv2.waitKey(0)
