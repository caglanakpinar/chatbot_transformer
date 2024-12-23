from __future__ import absolute_import, division, print_function, unicode_literals

import re
import tensorflow as tf
import tensorflow_datasets as tfds

from configs import Params
from dataset import Data

tf.keras.utils.set_random_seed(1234)

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

print(f"REPLICAS: {strategy.num_replicas_in_sync}")


def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # removing contractions
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence


def load_conversations(
        params: Params,
        data: Data
):
    # dictionary of line id to text
    id2line = {}
    with open(data.path_to_movie_lines, errors="ignore") as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace("\n", "").split(" +++$+++ ")
        id2line[parts[0]] = parts[4]

    inputs, outputs = [], []
    with open(data.path_to_movie_conversations, "r") as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace("\n", "").split(" +++$+++ ")
        # get conversation in a list of line ID
        conversation = [line[1:-1] for line in parts[3][1:-1].split(", ")]
        for i in range(len(conversation) - 1):
            inputs.append(preprocess_sentence(id2line[conversation[i]]))
            outputs.append(preprocess_sentence(id2line[conversation[i + 1]]))
            if len(inputs) >= params.MAX_SAMPLES:
                return inputs, outputs, inputs, outputs

    core_inputs, core_outputs = inputs, outputs

    for i in data.text['rows']:
        for l in i['row']['messages']:
            if l['role'] == 'user':
                inputs.append(l['content'])
            else:
                outputs.append(l['content'])
    if len(data.chatbot_text) != 0:
        for i in data.chatbot_text:
            _user, _content = i.split(":")
            if _user != 'assistant':
                inputs.append(_content)
            else:
                outputs.append(_content)
    return inputs, outputs, core_inputs, core_outputs


def get_tokenizer(questions, answers):
    try:
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file("tokens")
    except:
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            questions + answers, target_vocab_size=2 ** 13
        )
        tokenizer.save_to_file("tokens")

    # Define start and end token to indicate the start and end of a sentence
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    # Vocabulary size plus start and end token
    VOCAB_SIZE = tokenizer.vocab_size + 2
    return tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE


# Tokenize, filter and pad sentences
def tokenize_and_filter(
    params: Params,
    inputs,
    outputs,
    tokenizer,
):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        # tokenize sentence
        sentence1 = params.START_TOKEN + tokenizer.encode(sentence1) + params.END_TOKEN
        sentence2 = params.START_TOKEN + tokenizer.encode(sentence2) + params.END_TOKEN
        # check tokenized sentence max length
        if len(sentence1) <= params.MAX_LENGTH and len(sentence2) <= params.MAX_LENGTH:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

    # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=params.MAX_LENGTH, padding="post"
    )
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=params.MAX_LENGTH, padding="post"
    )

    return tokenized_inputs, tokenized_outputs
