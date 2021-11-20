import cv2
import numpy as np
import onnx
import onnxruntime

import os
import gdown

from facenet_pytorch import MTCNN, fixed_image_standardization

mtcnn = MTCNN(
    keep_all = True,
    image_size=160, margin=40, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True)

class_to_idx = {'Empty': 0, 'Heart': 1, 'Oblong': 2, 'Oval': 3, 'Round': 4, 'Square': 5}
idx_to_class = {v:k for k,v in class_to_idx.items()}
model_name = "20211118-8.0-pretrained_resnetv1.onnx"
if model_name not in os.listdir(os.getcwd()):
    print("[ERROR] Couldn't find the fine-tuned model!")
    print("[] Downloading the model through google drive...")
    
    url = 'https://drive.google.com/uc?id=1nLJu05fwG_uYeNNoKd6hPYm_mKvP4qWP'
    output = model_name
    gdown.download(url, output, quiet = False)
    print("[] DONE!")
else:
    print("[] Found %s!"%model_name.split('-')[1])

onnx_model = onnx.load(model_name)
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(model_name)


def preprocess(input_frame):
    """ 
    Processed the given input frame,
    Detects faces, resizes them and standardizes suitably for the finetuned model.
    Returns detected and processed faces as a numpy array.
    Input:
        input_frame - np.array: input frame
    Output: 
        frame - np.array: an array contains detected and processed faces
    """
    frame = np.copy(input_frame)
    # detect and crop the faces
    bboxes, _ = mtcnn.detect(frame)
    faces = []
    for box in bboxes:
        x0,y0,x1,y1 = map(int, box)
        face = frame[x0:x1, y0:y1]
        face = np.array(cv2.resize(face, (160, 160)), dtype = np.float32)
        face = face.transpose(2, 0, 1) # convert it to 3x160x160
        faces.append(fixed_image_standardization(face)) # standardize the image and append it to the list
    faces = np.array(faces) # shape: (num_faces)x3x160x160
    return faces, bboxes

def predict_shape(frame):
    """
    Classifies the face shapes of detected faces in the given frame
    
    
    Model: Fine-tuned Inception_Resnetv1
    https://drive.google.com/file/d/1EcWicvZAtQuggPhQn4Z0Y0Z6kJQspBZG/view?usp=sharing

    Input:
        frame - np.array: Tahmin için kaynak frame, RGB formatında olmalı.
    
    Returns:
        outs - list: a list that contains class ids, class names, model confidences and bounding boxes for detected faces 
    """
    faces, bboxes = preprocess(frame)
    inputs = {ort_session.get_inputs()[0].name: faces}
    ort_outs = ort_session.run(None, inputs)[0]
    class_ids = np.argmax(ort_outs, axis = 1)
    outs = []
    for i in range(len(faces)):
        class_id = class_ids[i]
        class_name = idx_to_class[class_id]
        confidence = ort_outs[i][class_id]
        out = {"class_id":class_ids[i],"class_name": class_name,"confidence": confidence,"bbox": bboxes[i]}
        outs.append(out)
    
    return outs
