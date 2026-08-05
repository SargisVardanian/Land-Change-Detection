from .schema import Siglip2TemporalConfig


def default_config() -> Siglip2TemporalConfig:
    return Siglip2TemporalConfig().validate()
