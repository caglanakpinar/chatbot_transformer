import tensorflow as tf

from configs import Params
from preprocess import preprocess_sentence


class Predict:
    def __init__(self, params: Params, tokenizer: object, model: object) -> object:
        self.params = params
        self.tokenizer = tokenizer
        self.model = model

    def evaluate(self, sentence):
        sentence = preprocess_sentence(sentence)

        sentence = tf.expand_dims(
            self.params.START_TOKEN + self.tokenizer.encode(sentence) + self.params.END_TOKEN, axis=0
        )

        output = tf.expand_dims(self.params.START_TOKEN, 0)

        for i in range(self.params.MAX_LENGTH):
            predictions = self.model(
                inputs=[sentence, output], training=False
            )

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, self.params.END_TOKEN[0]):
                break

            # concatenated the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0)


    def predict(self, sentence):
        prediction = self.evaluate(sentence)
        predicted_sentence = self.tokenizer.decode(
            [i for i in prediction if i < self.tokenizer.vocab_size]
        )
        return predicted_sentence
