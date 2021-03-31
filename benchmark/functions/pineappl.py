import numpy as np
import tensorflow as tf
from vegasflow.utils import generate_condition_function
from vegasflow.configflow import DTYPE, MAX_EVENTS_LIMIT, float_me,DTYPEINT

hbarc2 = tf.constant(389379372.1, dtype=DTYPE)
alpha0 = tf.constant(1.0 / 137.03599911, dtype=DTYPE)
cuts = generate_condition_function(6, condition='and')

dim = 3

@tf.function
def int_photo(s, t, u):
    return alpha0 * alpha0 / 2.0 / s * (t / u + u / t)


@tf.function
def hadronic_pspgen(xarr, mmin, mmax):
    smin = float_me(mmin * mmin)
    smax = float_me(mmax * mmax)

    r1 = xarr[:, 0]
    r2 = xarr[:, 1]
    r3 = xarr[:, 2]

    tau0 = smin / smax
    tau = tf.pow(tau0, r1)
    y = tf.pow(tau, 1.0 - r2)
    x1 = y
    x2 = tau / y
    s = tau * smax

    jacobian = tf.math.log(tau0) * tf.math.log(tau0) * tau * r1

    # theta integration (in the CMS)
    cos_theta = 2.0 * r3 - 1.0
    jacobian *= 2.0

    t = -0.5 * s * (1.0 - cos_theta)
    u = -0.5 * s * (1.0 + cos_theta)

    # phi integration
    jacobian *= 2.0 * np.math.acos(-1.0)

    return s, t, u, x1, x2, jacobian


def fill(grid, x1, x2, q2, yll, weight):
    zeros = np.zeros(len(weight), dtype=np.uintp)
    grid.fill_array(x1, x2, q2, zeros, yll, zeros, weight)


@tf.function
def pineappl(xarr, n_dim=None,weight=None, **kwargs):
    s, t, u, x1, x2, jacobian = hadronic_pspgen(xarr, 10.0, 7000.0)

    ptl = tf.sqrt((t * u / s))
    mll = tf.sqrt(s)
    yll = 0.5 * tf.math.log(x1 / x2)
    ylp = tf.abs(yll + tf.math.acosh(0.5 * mll / ptl))
    ylm = tf.abs(yll - tf.math.acosh(0.5 * mll / ptl))

    jacobian *= hbarc2

    # apply cuts
    t_1 = ptl >= 14.0
    t_2 = tf.abs(yll) <= 2.4
    t_3 = ylp <= 2.4
    t_4 = ylm <= 2.4
    t_5 = mll >= 60.0
    t_6 = mll <= 120.0
    full_mask, indices = cuts(t_1, t_2, t_3, t_4, t_5, t_6)

    weight = tf.boolean_mask(jacobian * int_photo(s, u, t), full_mask, axis=0)
    #x1 = tf.boolean_mask(x1, full_mask, axis=0)
    #x2 = tf.boolean_mask(x2, full_mask, axis=0)
    #yll = tf.boolean_mask(yll, full_mask, axis=0)
    #vweight = weight * tf.boolean_mask(kwargs.get('weight'), full_mask, axis=0)
    return tf.scatter_nd(indices, weight, xarr.shape[0:1])

@tf.function(input_signature=[
                   tf.TensorSpec(shape=[None,dim], dtype=DTYPE),
                   tf.TensorSpec(shape=[], dtype=DTYPEINT),
                  tf.TensorSpec(shape=[None], dtype=DTYPE)
               ]
           )
def pineappl1(xarr, n_dim=None,weight=None, **kwargs):
    s, t, u, x1, x2, jacobian = hadronic_pspgen(xarr, 10.0, 7000.0)

    ptl = tf.sqrt((t * u / s))
    mll = tf.sqrt(s)
    yll = 0.5 * tf.math.log(x1 / x2)
    ylp = tf.abs(yll + tf.math.acosh(0.5 * mll / ptl))
    ylm = tf.abs(yll - tf.math.acosh(0.5 * mll / ptl))

    jacobian *= hbarc2

    # apply cuts
    t_1 = ptl >= 14.0
    t_2 = tf.abs(yll) <= 2.4
    t_3 = ylp <= 2.4
    t_4 = ylm <= 2.4
    t_5 = mll >= 60.0
    t_6 = mll <= 120.0
    full_mask, indices = cuts(t_1, t_2, t_3, t_4, t_5, t_6)

    weight = tf.boolean_mask(jacobian * int_photo(s, u, t), full_mask, axis=0)
    #x1 = tf.boolean_mask(x1, full_mask, axis=0)
    #x2 = tf.boolean_mask(x2, full_mask, axis=0)
    #yll = tf.boolean_mask(yll, full_mask, axis=0)
    #vweight = weight * tf.boolean_mask(kwargs.get('weight'), full_mask, axis=0)
    #tf_scatter_nd_input_signature(weight,indices)
    return tf.scatter_nd(indices, weight, tf.shape(xarr)[0:1])