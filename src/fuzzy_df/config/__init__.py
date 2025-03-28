from fuzzy_df.config.builder import (
    RapidFuzzConfig,
    RapidFuzzConfigBuilder,
    RapidFuzzScorerConfig,
)

_default_config = RapidFuzzConfigBuilder().build()
_default_scorer_config = RapidFuzzConfigBuilder().build_scorer()

__all__ = [
    "RapidFuzzConfig",
    "RapidFuzzScorerConfig",
    "_default_config",
    "_default_scorer_config",
]
