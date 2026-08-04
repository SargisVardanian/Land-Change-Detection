from .schema import QCPRConfig


def default_config() -> QCPRConfig:
    config = QCPRConfig()
    config.validate()
    return config
