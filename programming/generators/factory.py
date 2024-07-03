from .py_generate import PyGenerator
from .model import CodeLlama, ModelBase, GPT4, GPT35, StarCoder

def model_factory(model_name: str, port: str = "", key: str = "") -> ModelBase:
    if "gpt-4" in model_name:
        return GPT4(model_name, key)
    elif "gpt-3.5" in model_name:
        return GPT35(model_name, key)
    elif model_name == "starcoder":
        return StarCoder(port)
    elif model_name == "codellama":
        return CodeLlama(port)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
