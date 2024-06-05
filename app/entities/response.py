from typing import Any

from app.entities.base import CamelCaseModel


class HttpResponse(CamelCaseModel):
  status: bool
  data: Any
  message: str