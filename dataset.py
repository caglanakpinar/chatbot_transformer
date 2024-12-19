from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
import pandas as pd
import json

from configs import Params

tf.keras.utils.set_random_seed(1234)


def json_read(path):
    file = open(path, "r")
    return  json.loads(file.read())


class Data:
    # open source data that are available on public at huggingface
    path_to_zip = tf.keras.utils.get_file(
        "cornell_movie_dialogs.zip",
        origin="http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",
        extract=True,
    )

    path_to_dataset = os.path.join(
        os.path.dirname(path_to_zip), "cornell movie-dialogs corpus"
    )

    path_to_movie_lines = os.path.join(path_to_dataset, "movie_lines.txt")
    path_to_movie_conversations = os.path.join(path_to_dataset, "movie_conversations.txt")

    # commonsense_qa -> huggingface
    commonsense = pd.concat([
        pd.read_parquet(
            "hf://datasets/tau/commonsense_qa/data/train-00000-of-00001.parquet"
        ),
        pd.read_parquet(
            "hf://datasets/tau/commonsense_qa/data/validation-00000-of-00001.parquet"
        )
    ])
    text = json_read("text.txt")
    telecom_corpus = pd.read_parquet(
        "https://huggingface.co/api/datasets/talkmap/telecom-conversation-corpus/parquet/default/train/0.parquet"
    )


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