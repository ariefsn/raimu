from typing import List

from deepface import DeepFace
from fastapi import FastAPI
from pydantic import ValidationError
from requests import HTTPError

from app.entities.face_recognition import (FaceRecognitionDto,
                                           FaceRecognitionResponse,
                                           FaceRecognitionResponseWrapper)
from app.entities.response import HttpResponse
from app.helper.mapper import Mapper
from app.helper.mask import detect_mask
from app.helper.response import json_error, json_ok
from app.helper.utils import isBase64Image, isUrl

app = FastAPI()

@app.get('/', response_model=HttpResponse, responses={})
def ping():
  return json_ok('Welcome to Raimu', 200)

@app.post('/face-verify', name="Post Face Verify", responses={200: { "model": FaceRecognitionResponseWrapper }, 422: { "model": HttpResponse }})
def postFaceVerify(payload: FaceRecognitionDto):
  is_img_source_valid = isBase64Image(payload.img_source) or isUrl(payload.img_source)
  is_img_target_valid = isBase64Image(payload.img_target) or isUrl(payload.img_target)
  is_anti_spoofing_enabled = payload.anti_spoofing is True
  is_mask_detection_enabled = payload.mask_detection is True
  is_mask_detected_allowed = [True, None].__contains__(payload.mask_detected_allowed)

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
    result = DeepFace.verify(img1_path=payload.img_source, img2_path=payload.img_target, anti_spoofing=is_anti_spoofing_enabled)
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
