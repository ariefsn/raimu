from typing import Dict, List

from fastapi.responses import JSONResponse

from app.entities.base import CamelCaseModel


def json_ok(data: CamelCaseModel | List[CamelCaseModel] | str, code: int = 200)-> JSONResponse:
  d = data
  if (isinstance(data, CamelCaseModel)):
    d = data.model_dump(by_alias=True)
  elif (isinstance(data, str)):
    d = data
  else:
    def map_data(each: CamelCaseModel):
      return each.model_dump(by_alias=True)
    d = {
      "items": map(map_data, data)
    }

  return JSONResponse(
    status_code=code,
    content={
      "status": True,
      "data": d,
      "message": ""
    },
  )

def json_error(message: str, code: int = 500)->Dict[str, any]:
  return JSONResponse(
    status_code=code,
    content={
      "status": False,
      "data": None,
      "message": message
    }
  )

