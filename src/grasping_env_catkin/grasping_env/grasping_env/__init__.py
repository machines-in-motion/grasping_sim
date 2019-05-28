from .core import GraspingEnv
from .grasping_env_pregrasps import GraspingEnvPregrasps
from .grasping_env_reacher import GraspingEnvReacher
from .grasping_env_cylinder import GraspingEnvCylinder
from .utils import ordered_load

# ==================== Factory ====================


def load_config(config_path):
    with open(config_path, 'r') as config_file:
        try:
            return ordered_load(config_file)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)


def make_from_config(config):
    assert "world" in config, "Need to specify the world in the configuration file!"
    if config["world"] == "empty":
        return GraspingEnv.from_config(config)
    elif config["world"] == "pregrasps":
        return GraspingEnvPregrasps.from_config(config)
    elif config["world"] == "reacher":
        return GraspingEnvReacher.from_config(config)
    elif config["world"] == "cylinder":
        return GraspingEnvCylinder.from_config(config)
    else:
        raise RuntimeError("Unknown world configuration!")
