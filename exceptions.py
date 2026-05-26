"""gnaa.network.exceptions 模組。"""

class AgentsException(Exception):
    """AgentsException 類別，封裝此模組的資料結構與服務邏輯。"""
    pass

class LLMException(AgentsException):
    """LLMException 類別，封裝此模組的資料結構與服務邏輯。"""
    pass

class ConfigException(AgentsException):
    """ConfigException 類別，封裝此模組的資料結構與服務邏輯。"""
    pass

class ToolException(AgentsException):
    """ToolException 類別，封裝此模組的資料結構與服務邏輯。"""
    pass

