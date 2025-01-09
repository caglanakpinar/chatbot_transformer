import os
import threading

from flask import Flask, request
import json
import argparse
from pathlib import Path

from configs import Params
from dataset import Data
from predict import Predict
from preprocess import load_conversations, get_tokenizer
from train import Train
from utils import Paths


parser = argparse.ArgumentParser()
parser.add_argument('-F','--additional_file_path', required=False, default=None)
args = parser.parse_args()

params = Params(trainer_config_path="hyper_params.yaml")


class BaseServe:
    def __init__(self, params: Params):
        self.params = params
        self.prompt = {'prompt': None}
        self.predict: Predict | None = None
        self.port = params.get("port")
        self.host = params.get("host")
        self.load_predict_function()

    def load_predict_function(self):
        params = Params(trainer_config_path="hyper_params.yaml")

        data = Data()
        data.read_additional_data(additional_file_path=args.additional_file_path)
        q, a, core_q, core_a = load_conversations(
            params,
            data
        )
        tokenizer, params.START_TOKEN, params.END_TOKEN, params.VOCAB_SIZE = get_tokenizer(
            core_q, core_a
        )
        train = Train.load(params)
        self.predict = Predict(
            params,
            tokenizer,
            train.model
        )

        print(self.predict.predict("where have you been?"))

    def model_train(self):
        print("training process will start ...")
        os.system("poetry run python3 main.py")

    def read_write(self, files, date, additional_file_path=None):
        lines = []
        for f in files:
            with open(f, "r")as _f:
                _data = _f.readlines()
            lines += _data
        _f.close()
        path = (
            Path(additional_file_path)
            if additional_file_path is not None
            else Paths.data_path
        )
        with open(path / f"{date}.txt", "w") as file:
            for l in lines:
                file.write('%s\n' % l)
        file.close()

    def init_api(self):
        app = Flask(__name__)
        params = self.prompt
        model_train = self.model_train
        read_write = self.read_write

        @app.route("/")
        def render_script():
            data = json.loads(request.data)
            for p in params:
                if p in data.keys():
                    params[p] = data[p]
            return {"output": self.predict.predict(params['prompt'])}

        @app.route("/train")
        def train():
            thr = threading.Thread(target=model_train)
            thr.daemon = True
            thr.start()
            return {"output": "model train done"}

        @app.route("/files")
        def get_files():
            params = json.loads(request.data)
            thr = threading.Thread(target=read_write, daemon=True, kwargs=params)
            thr.daemon = True
            thr.start()
            return {"output": "files created"}

        return app.run(threaded=False, debug=True, port=self.port, host=self.host)

    def serve(self, inputs):
        return {"output": self.predict.predict(inputs['prompt'])}

params = Params(trainer_config_path="hyper_params.yaml")
api = BaseServe(params)
api.init_api()
