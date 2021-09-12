import cv2
import numpy as np
from fsdtorch.inference import predict_shape


cam = cv2.VideoCapture(0)
while cam.isOpened():
    success, input_frame = cam.read()

    frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

    class_id, class_name, confidence = predict_shape(frame)

    print("Predicted class id: {} | class name: {} | confidence: {}".format(class_id, class_name, confidence))

    cv2.imshow("Live Source", input_frame)


    if cv2.waitKey(1) & 0xFF == 27: # ESC ile çıkış yapılır
        cv2.destroyAllWindows()
        break