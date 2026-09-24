"""Đọc cấu hình YAML (hỗ trợ kế thừa qua khoá `_base_`) và các tiện ích chung."""
import os
import random

import numpy as np
import torch
import yaml


def _deep_update(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _parse_value(text):
    """Parse giá trị dòng lệnh như YAML; '1e-4' cũng được hiểu là số."""
    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def load_config(path, overrides=None):
    """Đọc file YAML. `overrides` là list chuỗi "KEY=VALUE" (VALUE được parse như YAML)."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base_name = cfg.pop("_base_", None)
    if base_name:
        base_path = os.path.join(os.path.dirname(os.path.abspath(path)), base_name)
        cfg = _deep_update(load_config(base_path), cfg)

    for item in overrides or []:
        key, value = item.split("=", 1)
        cfg[key.strip()] = _parse_value(value)
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def print_environment():
    n_gpus = torch.cuda.device_count()
    print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | Số GPU: {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} - {props.total_memory / 1024**3:.1f} GB")
