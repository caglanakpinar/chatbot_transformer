from pathlib import Path
from typing import Any, Dict
import yaml

from utils import Paths

import tensorflow as tf


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU {}".format(tpu.cluster_spec().as_dict()["worker"]))
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()


class Params(Paths):
    # Maximum sentence length
    MAX_LENGTH = 40

    # Maximum number of samples to preprocess
    MAX_SAMPLES = 50000

    # For tf.data.Dataset
    BATCH_SIZE = 64
    BUFFER_SIZE = 20000

    # For Transformer
    NUM_LAYERS = 2
    D_MODEL = 256
    NUM_HEADS = 8
    UNITS = 512
    DROPOUT = 0.1
    EPOCHS = 40
    LR = 0.001
    VOCAB_SIZE = None
    START_TOKEN = None
    END_TOKEN = None
    tokenizer = None


    def __init__(
        self,
        trainer_config_path: Path | str = None,
        trainer_arguments: dict = None,
        **kwargs,
    ):
        self.parameter_keys = []
        self.trainer_config_path = trainer_config_path
        self.read_from_config(trainer_config_path, trainer_arguments, **kwargs)

    def get(self, p):
        assert (
            getattr(self, p, None) is not None
        ), f"<{p}> - is not available at train parameters .yaml file"
        return getattr(self, p)

    def read_from_config(
        self, trainer_config_path, trainer_arguments: dict = None, **kwargs
    ):
        if trainer_arguments is None:
            trainer_arguments = self.read_yaml(self.parent_dir / trainer_config_path)
        setattr(self, "parameter_keys", [*trainer_arguments.keys()])
        for p, value in trainer_arguments.items():
            setattr(self, p, value)
        if kwargs is not None:
            for p, value in kwargs.items():
                setattr(self, p, value)

    def store_params(self, params: dict):
        updated_params = {}
        for p, value in self.read_yaml(
            self.parent_dir / self.trainer_config_path
        ).items():
            updated_params[p] = params.get(p, value)
        self.write_yaml(self.parent_dir / self.trainer_config_path, updated_params)

    @staticmethod
    def read_yaml(folder):
        """
        :param folder: file path ending with .yaml format
        :return: dictionary
        """
        with open(
            f"{str(folder)}.yaml"
            if str(folder).split(".")[-1] not in ["yaml", "yml"]
            else folder
        ) as file:
            docs = yaml.full_load(file)
        return docs

    @staticmethod
    def write_yaml(folder, params: dict):
        """
        :param folder: file path ending with .yaml format
        :param params: dict to .yaml format
        """
        with open(
            (
                f"{str(folder)}.yaml"
                if str(folder).split(".")[-1] not in ["yaml", "yml"]
                else folder
            ),
            "w",
        ) as file:
            yaml.dump(params, file)