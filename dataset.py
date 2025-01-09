from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
import pandas as pd
import json
import keras

from configs import Params
from utils import Paths
from pathlib import Path

tf.keras.utils.set_random_seed(1234)


def json_read(path):
    file = open(path, "r", encoding='utf-8')
    return  json.loads(file.read())

def read_text(path):
    file = open(path, "r", encoding='utf-8')
    lines = file.read()
    return [l.replace("\n", "") for l in lines if l != r'\n' and l.replace("\n", "") != ""]


class Data:
    path_to_lines = Paths.parent_dir / "lines.txt"
    path_to_conversations = Paths.parent_dir / "text1.txt"
    text = json_read("text.txt")
    telecom_corpus = pd.read_parquet(
        "https://huggingface.co/api/datasets/talkmap/telecom-conversation-corpus/parquet/default/train/0.parquet"
    )
    chatbot_text = []

    def read_additional_data(self, additional_file_path=None):
        self.chatbot_text = []
        data_path = Path(additional_file_path) if additional_file_path is not None else  Paths.data_path
        if data_path.exists():
            files = [f for f in data_path.iterdir() if f.is_file()]
            for f in files:
                if f.is_file():
                    _f = read_text(f)
                    self.chatbot_text += read_text(f)


    def create_dataset(self, params: Params, questions, answers):
        # decoder inputs use the previous target as input
        # remove START_TOKEN from targets
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                {"inputs": questions, "dec_inputs": answers[:, :-1]},
                {"outputs": answers[:, 1:]},
            )
        )

        dataset = dataset.cache()
        dataset = dataset.shuffle(params.BUFFER_SIZE)
        dataset = dataset.batch(params.BATCH_SIZE)
        return dataset.prefetch(tf.data.AUTOTUNE)
