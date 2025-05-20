import time
from typing import List

from deepface import DeepFace
from fastapi import FastAPI, Request
from pydantic import ValidationError
from requests import HTTPError

from app.entities.face_recognition import (FaceRecognitionDto,
                                           FaceRecognitionResponse,
                                           FaceRecognitionResponseWrapper)
from app.entities.response import HttpResponse
from app.helper.mapper import Mapper
from app.helper.mask import detect_mask
from app.helper.response import json_error, json_ok
from app.helper.utils import (is_base64_image, is_url, read_image_from_base64,
                              read_image_from_url)

app = FastAPI()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    body = await request.body()
    print(f"[LOG] 📥 Request: {request.method} {request.url.path}")

    response = await call_next(request)

    duration = time.time() - start_time
    print(f"[LOG] 📤 Response: {response.status_code} | Duration: {duration:.4f}s\n")

    return response

@app.get('/', response_model=HttpResponse, responses={})
def ping():
  return json_ok('Welcome to Raimu', 200)

@app.post('/face-verify', name="Post Face Verify", responses={200: { "model": FaceRecognitionResponseWrapper }, 422: { "model": HttpResponse }})
def postFaceVerify(payload: FaceRecognitionDto):
  is_img_source_valid = is_base64_image(payload.img_source) or is_url(payload.img_source)
  is_img_target_valid = is_base64_image(payload.img_target) or is_url(payload.img_target)
  is_anti_spoofing_enabled = payload.anti_spoofing is True
  is_mask_detection_enabled = payload.mask_detection is True
  is_mask_detected_allowed = [True, None].__contains__(payload.mask_detected_allowed)
  model = payload.model
  if not model:
    model = "Facenet512"

  if is_img_source_valid == False or is_img_target_valid == False:
    return json_error('Image should be a url or base64 encoded', 422)
  
  masked_faces:List[str] = []
  if is_mask_detection_enabled:
    # Do the mask check
    mask_source = detect_mask(payload.img_source)
    if mask_source:
      masked_faces.append('imgSource')
      if not is_mask_detected_allowed:
        return json_error('Image source is masked', 422)
    mask_target = detect_mask(payload.img_target)
    if mask_target:
      masked_faces.append('imgTarget')
      if not is_mask_detected_allowed:
        return json_error('Image target is masked', 422)

  try:
    img1 = payload.img_source
    img2 = payload.img_target
    # Image 1
    if (is_url(img1)):
      print("Fetching image source from URL")
      img1 = read_image_from_url(payload.img_source)
      print("Image source URL fetched")
    elif (is_base64_image(img1)):
      print("Read image source from Base64")
      img1 = read_image_from_base64(payload.img_source)

    # Image 2
    if (is_url(img2)):
      print("Fetching image target from URL")
      img2 = read_image_from_url(payload.img_target)
      print("Image target URL fetched")
    elif (is_base64_image(img2)):
      print("Read image target from Base64")
      img2 = read_image_from_base64(payload.img_target)

    result = DeepFace.verify(
      img1_path=img1,
      img2_path=img2,
      anti_spoofing=is_anti_spoofing_enabled,
      model_name=model
    )
    result["facial_areas"] = {
      "img_source": result["facial_areas"]["img1"],
      "img_target": result["facial_areas"]["img2"],
    }
    result["masked_faces"] = masked_faces
    toCamel = Mapper.underscore_to_camelcase(result)
    resultObj = FaceRecognitionResponse.model_validate(toCamel, strict=False)
    return json_ok(resultObj)
  except HTTPError as err:
    if err.args.__len__() > 0:
      return json_error(err.args[0], err.response.status_code)
    return json_error('Unknown error occured. Please contact administrator.')
  except ValidationError as err:
    return json_error("Data is invalid. Please contact administrator.", 404)
  except ValueError as err:
    return json_error("Face not found.", 404)
