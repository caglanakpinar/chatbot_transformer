from configs import Params
from dataset import Data
from predict import Predict
from preprocess import load_conversations, get_tokenizer, tokenize_and_filter
from train import Train
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-F','--additional_file_path', required=False, default=None)
args = parser.parse_args()

params = Params(trainer_config_path="hyper_params.yaml")


data = Data()
data.read_additional_data(additional_file_path=args.additional_file_path)
q, a, core_a, core_q = load_conversations(
    params,
    data
)
(
    tokenizer,
    params.START_TOKEN,
    params.END_TOKEN,
    params.VOCAB_SIZE
) = get_tokenizer(
    core_a, core_q
)

q_tokenized, a_tokenized = tokenize_and_filter(
    params,
    q,
    a,
    tokenizer,
)
dataset = data.create_dataset(
    params,
    q_tokenized,
    a_tokenized
)
trainer = Train(params, dataset)
trainer.build()
trainer.train()
trainer.save()

train = Train.load(params)
p = Predict(
    params,
    tokenizer,
    train.model
)

print(p.predict("where have you been?"))
print(p.predict("how are you?"))