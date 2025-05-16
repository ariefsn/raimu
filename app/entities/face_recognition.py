from typing import List

from app.entities.base import CamelCaseModel
from app.entities.response import HttpResponse


class FaceRecognitionDto(CamelCaseModel):
  img_source: str
  img_target: str
  anti_spoofing: bool | None = None
  mask_detection: bool | None = None
  mask_detected_allowed: bool | None = None
  model: str | None = "Facenet512"

class FaceRecognitionArea(CamelCaseModel):
  x: float
  y: float
  w: float
  h: float
  left_eye: List[float] | None = None
  right_eye: List[float] | None = None

class FaceRecognitionFacialArea(CamelCaseModel):
  img_source: FaceRecognitionArea
  img_target: FaceRecognitionArea

class FaceRecognitionResponse(CamelCaseModel):
  verified: bool
  facial_areas: FaceRecognitionFacialArea
  distance: float
  threshold: float
  model: str
  detector_backend: str
  similarity_metric: str
  time: float
  masked_faces: List[str]

class FaceRecognitionResponseWrapper(HttpResponse):
  data: FaceRecognitionResponse
