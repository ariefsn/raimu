from os import getcwd, path
from typing import List


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