from __future__ import division
import os, sys, time
import pickle
from collections import OrderedDict
import yaml
import numpy as np


# ==================== Quaternion tools ====================
# Quaternions are represented as [w, x, y, z].

EPS = np.finfo(float).eps * 4.0


def quaternion_inverse(quaternion):
    q = np.array(quaternion, copy=True)
    np.negative(q[1:], q[1:])
    return q / np.dot(q, q)


def quaternion_matrix(quaternion):
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < EPS:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]])


def quaternion_multiply(quaternion0, quaternion1):
    # quat0 * quat1 corresponds to R0 * R1
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


# ==================== Hyperparameters ====================


class HyperParamHandler(object):

    def __init__(self, params):
        self.params = params
        self.sizes = list(map(len, params))
        self.max_id = 1
        for size in self.sizes:
            assert(size > 0)
            self.max_id *= size

    def get_params_from_id(self, proc_id):
        assert(0 <= proc_id < self.max_id)
        params = []
        for i, size in enumerate(self.sizes):
            params.append(self.params[i][proc_id % size])
            proc_id //= size
        return params


def instantiate_hyperparams(config, id_, header_delim="HEADER"):
    # Instantiate hyperparameters based on the config file and the id, and return the header.
    config["hyperparams"] = config["hyperparams"] if config["hyperparams"] is not None else []

    hyperparams = [config[key] for key in config["hyperparams"]]
    for i in hyperparams:
        assert isinstance(i, list), "Make sure hyperparameters are given as lists!"
    h = HyperParamHandler(hyperparams)
    new_params = h.get_params_from_id(id_ % h.max_id)

    header = [header_delim]
    for key, val in zip(config["hyperparams"], new_params):
        config[key] = val
        header.extend([key, "=", str(val)])
    return ' '.join(header)


# ==================== YAML ordered load/dump ====================

def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def ordered_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):
    class OrderedDumper(Dumper):
        pass
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


# ==================== Parameter scheduling ====================


class ConstSchedule(object):

    def __init__(self, val):
        self.val = val

    def value(self, t):
        return self.val


class LinearSchedule(object):

    def __init__(self, n_steps, initial_val, final_val, update_every):
        self.n_steps = n_steps
        self.final_val = final_val
        self.initial_val = initial_val
        self.update_every = update_every

    def value(self, t):
        step = int(self.n_steps // self.update_every) * self.update_every
        fraction = min(float(t) / self.n_steps, 1.0)
        return self.initial_val + fraction * (self.final_val - self.initial_val)

# ==================== Loading pregrasps ====================


def load_pregrasps(filename, min_phys_score=-1, max_phys_score=1):
    with open(filename, 'rb') as file:
        if sys.version_info >= (3, 0):
            data = pickle.load(file, encoding='bytes', fix_imports=True)
        else:
            data = pickle.load(file)
    # Used to avoid collision due to a potential error in the transform.
    delta = np.array([0, 0, 0.015])
    pregrasps = [np.hstack((el[b'obj_rel_pos'] + delta, el[b'obj_rel_orient']))
                 for el in data if el[b'score'] >= min_phys_score]
    scores = [el[b'score'] for el in data if el[b'score'] >=
              min_phys_score and el[b'score'] <= max_phys_score]

    return pregrasps, scores
