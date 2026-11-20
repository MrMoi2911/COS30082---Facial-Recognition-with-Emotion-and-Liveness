from dataclasses import dataclass
import importlib
import math


@dataclass
class LivenessResult:
    status: str
    score: float | None = None
    detail: str = ""


class MissingLiveness:
    def predict(self, face_rgb):
        return LivenessResult("unavailable", None, "Teammate's trained liveness model is not connected")


def load_liveness(spec=None, checkpoint=None):
    if not spec:
        return MissingLiveness()
    module_name, separator, factory_name = spec.partition(":")
    if not separator:
        raise ValueError("Use --liveness package.module:create_liveness")
    factory = getattr(importlib.import_module(module_name), factory_name)
    return factory(checkpoint)


def checked_prediction(adapter, face_rgb):
    try:
        result = adapter.predict(face_rgb)
        if not isinstance(result, LivenessResult):
            raise TypeError("Adapter must return longface.liveness.LivenessResult")
        if result.status not in {"live", "spoof", "uncertain", "unavailable"}:
            raise ValueError("Invalid liveness status")
        if result.status == "live" and (result.score is None or not math.isfinite(result.score)
                                         or not 0 <= result.score <= 1):
            raise ValueError("Live result requires a finite probability in [0,1]")
        return result
    except Exception as error:
        return LivenessResult("unavailable", None, f"Liveness error: {error}")


def allow_attendance(result, demo=False):
    return result.status == "live" or (demo and result.status == "unavailable")
