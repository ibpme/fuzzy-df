from typing import Any, Callable, Optional, TypedDict

from rapidfuzz import fuzz, utils


class RapidFuzzConfig(TypedDict):
    """RapidFuzz process configuration paramet:rs.
    Follows:
        https://rapidfuzz.github.io/RapidFuzz/Usage/process.html
    """

    scorer: Callable
    processor: Optional[Callable]
    score_cutoff: Optional[Any]
    score_hint: Optional[Any]
    score_multiplier: Optional[Any]
    dtype: Optional[Any]
    workers: int


class RapidFuzzScorerConfig(TypedDict):
    """RapidFuzz scorer configuration parameters.
    Follows:
        https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html
    """

    processor: Optional[Callable]
    score_cutoff: Optional[Any]


class RapidFuzzConfigBuilder:
    """Builder for RapidFuzzConfig."""

    def __init__(self):
        self._config = {
            "scorer": fuzz.WRatio,
            "processor": utils.default_process,
            "score_cutoff": None,
            "score_hint": None,
            "score_multiplier": 1,
            "dtype": None,
            "workers": 1,
        }

    def set_scorer(self, scorer: Callable) -> "RapidFuzzConfigBuilder":
        self._config["scorer"] = scorer
        return self

    def set_processor(self, processor: Optional[Callable]) -> "RapidFuzzConfigBuilder":
        self._config["processor"] = processor
        return self

    def set_score_cutoff(self, score_cutoff: Any) -> "RapidFuzzConfigBuilder":
        self._config["score_cutoff"] = score_cutoff
        return self

    def set_score_hint(self, score_hint: Any) -> "RapidFuzzConfigBuilder":
        self._config["score_hint"] = score_hint
        return self

    def set_score_multiplier(self, score_multiplier: Any) -> "RapidFuzzConfigBuilder":
        self._config["score_multiplier"] = score_multiplier
        return self

    def set_dtype(self, dtype: Any) -> "RapidFuzzConfigBuilder":
        self._config["dtype"] = dtype
        return self

    def set_workers(self, workers: int) -> "RapidFuzzConfigBuilder":
        self._config["workers"] = workers
        return self

    def build(self) -> RapidFuzzConfig:
        if self._config["score_hint"] is None:
            self._config.pop("score_hint")
        if self._config["score_multiplier"] is None:
            self._config.pop("score_multiplier")
        if self._config["dtype"] is None:
            self._config.pop("dtype")
        return RapidFuzzConfig(**self._config)

    def build_scorer(self) -> RapidFuzzScorerConfig:
        return RapidFuzzScorerConfig(
            processor=self._config["processor"],
            score_cutoff=self._config["score_cutoff"],
        )
