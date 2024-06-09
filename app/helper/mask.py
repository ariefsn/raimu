import base64
from os import getcwd, path
from urllib.request import urlopen
from uuid import uuid4

import cv2
import numpy as np
from cv2.typing import MatLike
from keras.api.applications.mobilenet_v2 import preprocess_input
from keras.api.models import load_model
from keras.api.preprocessing.image import img_to_array

from app.helper.utils import isBase64Image, isUrl


def url_to_cv2_image(url)->MatLike:
  cap = cv2.VideoCapture(url)
  ret, image = cap.read()
  return image

def base64_to_cv2_image(b64)->MatLike:
  split = b64.split(',')
  b64Type = split[0]
  b64Value = split[1]
  decoded = base64.b64decode(b64Value)
  # tmp_name = uuid4().hex + '.jpeg'
  # with open(tmp_name, 'wb') as f:
  #   f.write(decoded)
  # image = cv2.imread(tmp_name)
  np_data = np.fromstring(decoded, np.uint8)
  image = cv2.imdecode(np_data, cv2.IMREAD_ANYCOLOR)
  return image

def detect_mask(img)->bool:
  face_path = path.join(getcwd(), 'app', 'assets', 'haarcascade_frontalface_default.xml')
  faceCascade = cv2.CascadeClassifier(face_path)
  model_path = path.join(getcwd(), 'app', 'assets', 'mask_recog.h5')
  model = load_model(model_path)

  frame:MatLike = None
  if isUrl(img):
    frame = url_to_cv2_image(img)
  elif isBase64Image(img):
    frame = base64_to_cv2_image(img)
  else:
    frame = cv2.imread(img)

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)

  for (x, y, w, h) in faces:
      face_frame = frame[y:y+h, x:x+w]
      face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
      face_frame = cv2.resize(face_frame, (224, 224))
      face_frame = img_to_array(face_frame)
      face_frame = np.expand_dims(face_frame, axis=0)
      face_frame = preprocess_input(face_frame)
      
      preds = model.predict(face_frame)
      (mask, withoutMask) = preds[0]
      is_mask = mask > withoutMask

      if is_mask:
         return True

  return False