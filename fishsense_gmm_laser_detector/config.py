'''Config
'''
import datetime as dt
import logging
import logging.handlers
import os
import time
from importlib.metadata import version
from pathlib import Path
from typing import Dict

from dynaconf import Dynaconf, Validator

IS_DOCKER = os.environ.get('E4EFS_DOCKER', False)


def get_log_path() -> Path:
    """Get log path

    Returns:
        Path: Path to log directory
    """
    if IS_DOCKER:
        return Path('/e4efs/logs')
    log_path = Path('./logs')
    log_path.mkdir(parents=True, exist_ok=True)
    return log_path


def get_data_path() -> Path:
    """Get data path

    Returns:
        Path: Path to data directory
    """
    if IS_DOCKER:
        return Path('/e4efs/data')
    data_path = Path('./data')
    data_path.mkdir(parents=True, exist_ok=True)
    return data_path


def get_config_path() -> Path:
    """Get config path

    Returns:
        Path: Path to config directory
    """
    if IS_DOCKER:
        return Path('/e4efs/config')
    config_path = Path('.')
    return config_path


def get_cache_path() -> Path:
    """Get cache path

    Returns:
        Path: Path to cache directory
    """
    if IS_DOCKER:
        return Path('/e4efs/cache')
    cache_path = Path('./cache')
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path

validators = [
]

settings = Dynaconf(
    envvar_prefix='E4EFS',
    environments=False,
    settings_files=[
        (get_config_path() / 'settings.toml').as_posix(),
        (get_config_path() / '.secrets.toml').as_posix()],
    merge_enabled=True,
    validators=validators
)



def configure_log_handler(handler: logging.Handler):
    handler.setLevel(logging.DEBUG)
    msg_fmt = '%(asctime)s.%(msecs)03dZ - %(name)s - %(levelname)s - %(message)s'
    root_formatter = logging.Formatter(msg_fmt, datefmt='%Y-%m-%dT%H:%M:%S')
    handler.setFormatter(root_formatter)


def configure_logging():
    """Configures logging
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    log_dest = get_log_path().joinpath('e4efs_service.log')
    print(f'Logging to "{log_dest.as_posix()}"')

    log_file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_dest,
        when='midnight',
        backupCount=5
    )
    configure_log_handler(log_file_handler)
    root_logger.addHandler(log_file_handler)

    console_handler = logging.StreamHandler()
    configure_log_handler(console_handler)
    root_logger.addHandler(console_handler)
    logging.Formatter.converter = time.gmtime

    logging_levels: Dict[str, str] = {
        'PIL.TiffImagePlugin': 'INFO',
        'httpcore.http11': 'INFO',
    }
    for logger_name, level in logging_levels.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.getLevelNamesMapping()[level])

    logging.info('Log path: %s', get_log_path())
    logging.info('Data path: %s', get_data_path())
    logging.info('Config path: %s', get_config_path())
    logging.info('Executing fishsense_gmm_laser_detector:%s',
                 version('fishsense_gmm_laser_detector'))
