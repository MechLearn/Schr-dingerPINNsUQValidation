from utils.gpu import setup_gpu
setup_gpu()

import tensorflow as tf
from tensorflow.keras import mixed_precision
from model import build_model
from physics import pde_residual

mixed_precision.set_global_policy("mixed_float16")

model = build_model()
optimizer = tf.keras.optimizers.Adam(1e-3)

@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        res = pde_residual(model, x)
        loss = tf.reduce_mean(tf.square(res))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Entrenamiento
for epoch in range(2000):
    x = tf.random.uniform((1024,1), 0, 1)
    loss = train_step(x)
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss = {loss.numpy():.5e}")
