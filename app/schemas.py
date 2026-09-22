from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class Question(BaseModel):
    type: Literal["noul", "choice", "score"]
    instructions: str | dict | list
    criteria: dict[str, str] | list[str] | None = None

    @model_validator(mode="after")
    def check_criteria(self):
        c = self.criteria
        if self.type == "noul":
            if c is not None and (not isinstance(c, dict) or set(c) - {"true", "false"}):
                raise ValueError("noul criteria must be an object with optional 'true' and 'false' keys")
        elif self.type == "choice":
            if not isinstance(c, dict) or len(c) < 2:
                raise ValueError("choice criteria must be an object with at least 2 options")
        elif not isinstance(c, list) or not 2 <= len(c) <= 10:
            raise ValueError("score criteria must be a list of 2-10 levels, lowest first")
        return self


class SystemOneRequest(BaseModel):
    model: str = "r4b"
    state: str | dict | list
    questions: dict[str, Question] = Field(min_length=1)
    calibrate: bool = True  # apply a stored calibration if one exists for this model/mode/primitive/template
    trace: bool | None = None  # None: follow SYSTEMONE_TRACE=1


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]]
    usage: dict[str, int]
    timing: dict
    metadata: dict
