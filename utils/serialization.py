"""gnaa.utils.serialization 模組。"""

import json
import pickle
from typing import Any, Union
from pathlib import Path

def serialize_object(obj: Any, format: str = "json") -> Union[str, bytes]:
    """serialize_object 的主要實作。"""
    if format == "json":
        return json.dumps(obj, ensure_ascii=False, indent=2)
    elif format == "pickle":
        return pickle.dumps(obj)
    else:
        raise ValueError("內部訊息")

def deserialize_object(data: Union[str, bytes], format: str = "json") -> Any:
    """deserialize_object 的主要實作。"""
    if format == "json":
        return json.loads(data)
    elif format == "pickle":
        return pickle.loads(data)
    else:
        raise ValueError("內部訊息")

def save_to_file(obj: Any, filepath: Union[str, Path], format: str = "json") -> None:
    """save_to_file 的主要實作。"""
    filepath = Path(filepath)
    data = serialize_object(obj, format)
    
    mode = "w" if format == "json" else "wb"
    with open(filepath, mode) as f:
        f.write(data)

def load_from_file(filepath: Union[str, Path], format: str = "json") -> Any:
    """load_from_file 的主要實作。"""
    filepath = Path(filepath)
    mode = "r" if format == "json" else "rb"
    
    with open(filepath, mode) as f:
        data = f.read()
    
    return deserialize_object(data, format) 
