import cv2
import numpy as np
import onnx
import onnxruntime

import os
import gdown

idx_to_class = {'Empty': 0, 'Heart': 1, 'Oblong': 2, 'Oval': 3, 'Round': 4, 'Square': 5}
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

def to_numpy(tensor):
    """ istersek torch tensor girdi de verebiliriz """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def preprocess(input_frame):
    """ 
    Verilen girdi frame ini düzenler, bu fonksiyon proje geneli için değil modelin çalışması için tasarlandı.
    Dolayısıyla tahmin yaparken buraya gelen frame i önceden gerektiği gibi düzenleyip, sonra burada model için düzenlenmesine izin vermek gerekiyor.
    Girdi:
        input_frame - np.array: girdi framei
    Çıktı: 
        frame - np.array: düzenlenip işlenen frame
    """
    frame = np.copy(input_frame)
    frame = frame / 255.
    frame = cv2.resize(frame, (160, 160))
    frame = np.expand_dims(frame.transpose(2, 0, 1), 0) # convert it to 1x3x160x160
    frame = np.array(frame, dtype = np.float32)
    return frame

def predict_shape(frame):
    """
    Verilen frame i kullanarak frame de bulunan yüzü sınıflandırır ve şekli döndürür.
    Bu fonksiyonu override ederek batchler halinde girdi göndermek mümkün
    
    
    Kullanılan model: Fine-tunelanmış Inception_Resnetv1
    Model google colabde fine-tune edildi:
    https://drive.google.com/file/d/1EcWicvZAtQuggPhQn4Z0Y0Z6kJQspBZG/view?usp=sharing

    Input:
        frame - np.array: Tahmin için kaynak frame, RGB formatında olmalı.
    
    Returns:
        class_id - int: tahmin edilen sınıfın IDsi
        class_name - string: class_id kullanılarak elde edilen sınıf ismi, idxten sınıf ismine geçmek için idx_to_class kullanılıyor
        confidence - float: modelin tahmin confidence ı
    """
    processed_frame = preprocess(frame)
    inputs = {ort_session.get_inputs()[0].name: processed_frame}
    ort_outs = ort_session.run(None, inputs)
    out = ort_outs[0]
    class_id = np.argmax(out)
    class_name = idx_to_class[class_id]
    confidence = out[0][class_id]
    
    return class_id, class_name, confidence
