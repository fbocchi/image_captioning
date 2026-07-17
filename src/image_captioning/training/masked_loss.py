import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
def masked_loss(y_true, y_pred):
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true,
        y_pred,
        from_logits=False,
    )

    mask = tf.not_equal(
        y_true,
        0,
    )

    loss = tf.where(
        mask,
        loss,
        0.0,
    )

    return tf.math.divide_no_nan(
        tf.reduce_sum(loss),
        tf.reduce_sum(tf.cast(mask, loss.dtype))
    )