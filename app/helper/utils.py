import base64
from io import BytesIO
from os import getcwd, path
from typing import List

import cv2
import numpy as np
import requests
from PIL import Image


def cast_type(container, from_types, to_types):
  if isinstance(container, dict):
    # cast all contents of dictionary 
    return {cast_type(k, from_types, to_types): cast_type(v, from_types, to_types) for k, v in container.items()}
  elif isinstance(container, list):
    # cast all contents of list 
    return [cast_type(item, from_types, to_types) for item in container]
  else:
    for f, t in zip(from_types, to_types):
      # if item is of a type mentioned in from_types,
      # cast it to the corresponding to_types class
      if isinstance(container, f):
        return t(container)
    # None of the above, return without casting 
    return container


def is_base64_image(data: str)->bool:
  split = data.split(';')
  if split.__len__() < 2:
    return False
  
  return split[0].__contains__('image') or split[1].__contains__('base64')

def is_url(data: str)->bool:
  d = data.lower()
  return d.startswith('http://') or d.startswith('https://')

def asset(paths: List[str] = []):
  return path.join(getcwd(), 'app', 'assets', *paths)

def read_image_from_url(url):
  resp = requests.get(url)
  resp.raise_for_status()  # biar langsung error kalau 404 / 500
  img_array = np.frombuffer(resp.content, np.uint8)
  img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
  if img is None:
      raise ValueError("Failed to decode image from URL")
  return img

def read_image_from_base64(base64_str):
    header, encoded = base64_str.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(BytesIO(img_bytes))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)