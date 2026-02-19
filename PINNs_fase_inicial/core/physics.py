import tensorflow as tf

def pde_residual(model, x):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        with tf.GradientTape() as tape1:
            tape1.watch(x)
            u = model(x)
        du_dx = tape1.gradient(u, x)
    d2u_dx2 = tape2.gradient(du_dx, x)
    return d2u_dx2
