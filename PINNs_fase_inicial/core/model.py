import tensorflow as tf
from tensorflow.keras import layers, models

def build_model():
    model = models.Sequential([
        layers.Input((1,)),
        layers.Dense(64, activation="tanh"),
        layers.Dense(64, activation="tanh"),
        layers.Dense(1, dtype="float32", use_bias=False)
    ])
    return model
