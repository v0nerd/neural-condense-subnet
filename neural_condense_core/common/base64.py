import numpy as np
import base64
import io


def ndarray_to_base64(array: np.ndarray) -> str:
    """Convert a NumPy array to a base64-encoded string."""
    try:
        buffer = io.BytesIO()
        np.save(buffer, array)
        buffer.seek(0)
        base64_str = base64.b64encode(buffer.read()).decode("utf-8")
    except Exception:
        base64_str = ""
    return base64_str


def base64_to_ndarray(base64_str: str) -> np.ndarray:
    """Convert a base64-encoded string back to a NumPy array."""
    try:
        buffer = io.BytesIO(base64.b64decode(base64_str))
        buffer.seek(0)
        array = np.load(buffer)
    except Exception:
        array = np.array([])
    return array
