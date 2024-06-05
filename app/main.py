# from typing import Union
from json import dumps, loads

from deepface import DeepFace
from fastapi import FastAPI
from pydantic import ValidationError
from requests import HTTPError

from app.entities.face_recognition import (FaceRecognitionDto,
                                           FaceRecognitionResponse,
                                           FaceRecognitionResponseWrapper)
from app.entities.response import HttpResponse
from app.helper.mapper import Mapper
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

  if is_img_source_valid == False or is_img_target_valid == False:
    return json_error('Image should be a url or base64 encoded', 422)

  try:
    result = DeepFace.verify(img1_path=payload.img_source, img2_path=payload.img_target)
    result["facial_areas"] = {
      "img_source": result["facial_areas"]["img1"],
      "img_target": result["facial_areas"]["img2"],
    }
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
