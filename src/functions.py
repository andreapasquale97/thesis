import tensorflow as tf
import numpy as np


@tf.function
def gauss_vf(xarr, n_dim=None, **kwargs):
    """symgauss test function"""
    if n_dim is None:
        n_dim = xarr.shape[-1]
    a = tf.constant(0.1, dtype=DTYPE)
    n100 = tf.cast(100 * n_dim, dtype=DTYPE)
    pref = tf.pow(1.0 / a / np.sqrt(np.pi), n_dim)
    coef = tf.reduce_sum(tf.range(n100 + 1))
    coef += tf.reduce_sum(tf.square((xarr - 1.0 / 2.0) / a), axis=1)
    coef -= (n100 + 1) * n100 / 2.0
    return pref * tf.exp(-coef)


def gauss_v(x,dim=None):
    if dim is None:
        dim = x.shape[-1]
    dx2 = 0
    a = 0.1
    coef = (1.0/a/np.sqrt(np.pi))**dim
    for d in range(dim):
        dx2 += (x[d] - 0.5) ** 2
    return np.exp(-dx2 * 100.) * coef