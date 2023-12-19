import os
import shutil
import yaml
import time
import math
import socket
import numpy as np

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))


def _convert(value):
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (float, int)):
        if math.isnan(value):
            return None
        return value
    elif np.isscalar(value):
        if isinstance(value, (np.float32, np.float64)):
            return float(value)
        if isinstance(value, (np.int32, np.int64)):
            return int(value)
        raise ValueError("{}: {} is not supported.".format(value, type(value)))
    elif isinstance(value, np.ndarray):
        value_list: list = value.tolist()
        for i, value in enumerate(value_list):
            value_list[i] = _convert(value)
        return value_list
    elif isinstance(value, (list, tuple)):
        value_list = list(value)
        for i, value in enumerate(value_list):
            value_list[i] = _convert(value)
        return value_list
    else:
        return None, type(value)


def _convert_config(config):
    config_copy = {}
    for key, value in config.items():
        config_copy[key] = _convert(value)

    return config_copy


def create_log_dir(args):
    # env_name = args.env.replace("/", "_")
    run_id = "{}-{}-{}".format(
        args.agent, args.seed, time.strftime("%Y-%m-%d-%H-%M-%S")
    )
    log_dir = os.path.join(args.logdir, args.agent, run_id)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def save_code(save_dir):
    project_dir = PROJECT_DIR
    shutil.copytree(
        project_dir,
        save_dir + "/code",
        ignore=shutil.ignore_patterns(
            "venv",
            "log*",
            "result*",
            "refs",
            "dataset*",
            "model*",
            ".git",
            "*.pyc",
            ".idea",
            ".DS_Store",
        ),
    )


def timeit(func):
    def wrap(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        print("%s use %.3f sec" % (func.__name__, te - ts))
        return result

    return wrap


def save_config(config, save_dir):
    with open(os.path.join(save_dir, "config.yaml"), "w") as file:
        yaml.safe_dump(_convert_config(config), file, default_flow_style=False)
    try:
        yaml.safe_load(open(os.path.join(save_dir, "config.yaml"), "r"))
    except Exception as e:
        print(e)
        raise ValueError("config cannot be yaml dump.\n{}".format(config))


if __name__ == "__main__":
    print(PROJECT_DIR)
