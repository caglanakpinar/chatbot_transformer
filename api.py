from flask import Flask, request
import json


from configs import Params
from dataset import Data
from predict import Predict
from preprocess import load_conversations, get_tokenizer
from train import Train


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


    def init_api(self):
        app = Flask(__name__)
        params = self.prompt

        @app.route("/")
        def render_script():
            data = json.loads(request.data)
            for p in params:
                if p in data.keys():
                    params[p] = data[p]
            return {"output": self.predict.predict(params['prompt'])}

        return app.run(threaded=False, debug=False, port=self.port, host=self.host)

    def serve(self, inputs):
        return {"output": self.predict.predict(inputs['prompt'])}

params = Params(trainer_config_path="hyper_params.yaml")
api = BaseServe(params)
api.init_api()
