"""gnaa.utils.helpers 模組。"""

import importlib
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

def format_time(timestamp: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """format_time 的主要實作。"""
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime(format_str)

def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """validate_config 的主要實作。"""
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError("內部訊息")
    return True

def safe_import(module_name: str, class_name: Optional[str] = None) -> Any:
    """safe_import 的主要實作。"""
    try:
        module = importlib.import_module(module_name)
        if class_name:
            return getattr(module, class_name)
        return module
    except (ImportError, AttributeError) as e:
        raise ImportError("內部訊息")

def ensure_dir(path: Path) -> Path:
    """ensure_dir 的主要實作。"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_project_root() -> Path:
    """get_project_root 的主要實作。"""
    return Path(__file__).parent.parent.parent

def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """merge_dicts 的主要實作。"""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
