import base64

import cv2
import numpy as np
from cv2.typing import MatLike
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from app.helper.utils import asset, is_base64_image, is_url

face_path = asset(['haarcascade_frontalface_default.xml'])
faceCascade = cv2.CascadeClassifier(face_path)
model_path = asset(['mask_recog.h5'])
model = load_model(model_path)

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

def detect_mask(img) -> bool:
  try:
    frame: MatLike = None

    # Convert image input (URL, base64, or path) to OpenCV image
    if is_url(img):
      frame = url_to_cv2_image(img)
    elif is_base64_image(img):
      frame = base64_to_cv2_image(img)
    else:
      frame = cv2.imread(img)

    # Validate if image loaded properly
    if frame is None:
      raise ValueError("Failed to load image from input.")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load face cascade globally if not yet loaded
    global faceCascade
    if faceCascade is None or faceCascade.empty():
      faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
      if faceCascade.empty():
        raise RuntimeError("Failed to load Haar cascade classifier.")

    faces = faceCascade.detectMultiScale(
      gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
    )

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

    return False  # No face with mask detected

  except Exception as e:
    print(f"[ERROR] detect_mask failed: {e}")
    return False  # Fail gracefully, assume no mask
