"""
Arquitetura 1D-CAE e função de perda ponderada em picos de radar.
"""
import tensorflow as tf
from tensorflow.keras import layers, models

def perda_radar(y_true, y_pred):
    erro_quadratico = tf.square(y_true - y_pred)
    peso_picos = 1.0 + 9.0 * tf.abs(y_true)
    mse_ponderado = tf.reduce_mean(erro_quadratico * peso_picos)
    y_true_flat = tf.reshape(y_true, [-1, 1024])
    y_pred_flat = tf.reshape(y_pred, [-1, 1024])
    cosseno = tf.keras.losses.cosine_similarity(y_true_flat, y_pred_flat)
    return mse_ponderado + 0.1 * tf.reduce_mean(cosseno)

def construir_autoencoder(input_shape=(1024, 1)):
    inputs = layers.Input(shape=input_shape)
    c1 = layers.Conv1D(64, 7, strides=2, padding="same", activation="relu")(inputs)
    c2 = layers.Conv1D(128, 5, strides=2, padding="same", activation="relu")(c1)
    c3 = layers.Conv1D(256, 3, strides=2, padding="same", activation="relu")(c2)
    latent = layers.Conv1D(128, 3, padding="same", activation="relu")(c3)
    d1 = layers.Conv1DTranspose(128, 3, strides=2, padding="same", activation="relu")(latent)
    d2 = layers.Conv1DTranspose(64, 5, strides=2, padding="same", activation="relu")(d1)
    d3 = layers.Conv1DTranspose(32, 7, strides=2, padding="same", activation="relu")(d2)
    outputs = layers.Conv1D(1, 3, padding="same", activation="linear")(d3)
    return models.Model(inputs=inputs, outputs=outputs, name="1D_CAE_Radar")
