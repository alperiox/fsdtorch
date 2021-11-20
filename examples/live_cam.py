import cv2
import numpy as np
from fsdtorch import inference


cam = cv2.VideoCapture(0)
while cam.isOpened():
    success, frame = cam.read()

    # frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

    outputs = inference.predict_shape(frame)

    for output in outputs:
        bbox = output['bbox']
        x0, y0, x1, y1 = map(int, bbox)
        image_rgb = cv2.rectangle(frame, (x0, y0), (x1,y1), (255, 0, 0))
        print(output['class_name'], ":", output['confidence'])

    cv2.imshow("Live Source", frame)


    if cv2.waitKey(1) & 0xFF == 27: # exit with ESC
        cv2.destroyAllWindows()
        break