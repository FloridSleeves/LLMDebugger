from .py_executor import PyExecutor
from .executor_types import Executor

def executor_factory(lang: str, is_leet: bool = False) -> Executor:
    if lang == "py" or lang == "python":
        return PyExecutor()
    else:
        raise ValueError(f"Invalid language for executor: {lang}")
