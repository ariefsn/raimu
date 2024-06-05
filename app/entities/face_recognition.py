from typing import List

from app.entities.base import CamelCaseModel
from app.entities.response import HttpResponse


class FaceRecognitionDto(CamelCaseModel):
  img_source: str
  img_target: str

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

class FaceRecognitionResponseWrapper(HttpResponse):
  data: FaceRecognitionResponse
