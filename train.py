import tensorflow as tf

from configs import Params
from transformer import transformer
from utils import PositionalEncoding, MultiHeadAttentionLayer, Paths

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



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.constant(d_model, dtype=tf.float32)
        self.warmup_steps = warmup_steps

    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}

    def __call__(self, step):
        arg1 = tf.math.rsqrt(tf.cast(step, tf.float32))
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.multiply(
            tf.math.rsqrt(self.d_model), tf.math.minimum(arg1, arg2)
        )


class Train:
    def __init__(self, params: Params, dataset=None):
        self.params = params
        self.model = None
        self.dataset = dataset
        self.filename = "model.h5"
        self.model_checkpoint_callback = self.get_callbacks()

    def loss_function(self, y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, self.params.MAX_LENGTH - 1))

        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )(y_true, y_pred)

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)

    def accuracy(self, y_true, y_pred):
        # ensure labels have shape (batch_size, MAX_LENGTH - 1)
        y_true = tf.reshape(y_true, shape=(-1, self.params.MAX_LENGTH - 1))
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    def get_callbacks(self):
        return  tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Paths.parent_dir / "model.keras"),
            save_freq='epoch',
        )

    def build(self):
        # clear backend
        tf.keras.backend.clear_session()
        learning_rate = CustomSchedule(self.params.D_MODEL)
        optimizer = tf.keras.optimizers.Adam(
            self.params.LR, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )
        # initialize and compile model within strategy scope
        with strategy.scope():
            self.model = transformer(
                vocab_size=self.params.VOCAB_SIZE,
                num_layers=self.params.NUM_LAYERS,
                units=self.params.UNITS,
                d_model=self.params.D_MODEL,
                num_heads=self.params.NUM_HEADS,
                dropout=self.params.DROPOUT,
            )
            self.model.compile(
                optimizer=optimizer, loss=self.loss_function, metrics=[self.accuracy]
            )
        self.model.load_weights(
            "model.keras"
        )

        self.model.summary()

    def train(self):
        self.model.fit(
            self.dataset,
            epochs=self.params.EPOCHS,
            callbacks=[
                self.model_checkpoint_callback
            ]
        )

    def save(self):
        tf.keras.models.save_model(self.model, filepath=self.filename, include_optimizer=False)

    @classmethod
    def load(cls, params: Params):
        _cls = Train(params)
        _cls.model = tf.keras.models.load_model(
            _cls.filename,
            custom_objects={
                "PositionalEncoding": PositionalEncoding,
                "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
            },
            compile=False,
        )
        return _cls