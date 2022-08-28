import argparse
import pathlib
import os
import contextlib
import joblib
import numpy as np
import torch
import yaml
from addict import Dict


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def dict_replace_nan(d, new):
    x = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = dict_replace_nan(v, new)
        elif np.isnan(v) and isinstance(v, float):
            v = new
        x[k] = v
    return x


def none_or_nan(thing):
    if thing is None:
        return True
    elif isinstance(thing, float) and np.isnan(thing):
        return True
    else:
        return False


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def load_config(template_fp, custom_fp):

    assert os.path.isfile(template_fp), template_fp
    if custom_fp:
        assert os.path.isfile(custom_fp), custom_fp

    with open(template_fp, "r") as template_file:
        config_d = yaml.load(template_file, Loader=yaml.FullLoader)

    # overwrite parts of the config
    if custom_fp:

        with open(custom_fp, "r") as custom_file:
            custom_d = yaml.load(custom_file, Loader=yaml.FullLoader)

        config_d["project_name"] = custom_d["project_name"]
        if custom_d["run_name"] is None:
            config_d["run_name"] = os.path.splitext(os.path.basename(custom_fp))[0]
        else:
            config_d["run_name"] = custom_d["run_name"]
        for k, v in custom_d.items():
            if k not in ["project_name", "run_name"]:
                if k not in config_d:
                    config_d[k] = {}
                for k2, v2 in v.items():
                    config_d[k][k2] = v2

    project_name = config_d["project_name"]
    run_name = config_d["run_name"]
    data_d = config_d["data"]
    model_d = config_d["model"]
    run_d = config_d["run"]

    return project_name, run_name, data_d, model_d, run_d


def expand_path(string):
    return pathlib.Path(os.path.expandvars(string))


def generate_inputs(batch, nb):
    if "labels" in batch:
        labels = batch.pop("labels")
        labels = labels.cuda(non_blocking=nb)
    else:
        labels = None

    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cuda(non_blocking=nb)

    return batch, labels


def generate_pretraining_inputs(batch, nb):
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cuda(non_blocking=nb)

    return batch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()


def cycle(iterable):
    while True:
        for x in iterable:
            yield x



