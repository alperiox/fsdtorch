# fsdtorch
Simple package for face shape detection.

Example usage:
```py
from fsdtorch import inference
image_path = "example.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
class_id, class_name, confidence = inference(image_rgb)
```

I have fine-tuned a pre-trained `InceptionResnetv1` model from [`pytorch-facenet`](https://github.com/timesler/facenet-pytorch) with cropped version of [face shape dataset](https://www.kaggle.com/niten19/face-shape-dataset) from [kaggle](www.kaggle.com).

Another example script can be found in `examples` for live shape detection.

Inference is made by using exported `onnx` version of the model and `onnxruntime`. 