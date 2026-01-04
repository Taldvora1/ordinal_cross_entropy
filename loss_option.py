import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow import map_fn, argmax, gather
from distributions import get_beta_probabilities, get_poisson_probabilities, get_binominal_probabilities, \
	get_exponential_probabilities
tf.random.set_seed(42)
tf.config.experimental.enable_op_determinism()

def cross_entropy_loss(y_true, y_pred):
    class_hot = y_true
    y_true = tf.argmax(class_hot, axis=-1)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)

    # Observed class loss
    observed_class = -tf.math.log(y_pred_softmax + 1e-7) * class_hot

    # Unobserved class loss

    # Total loss
    loss = tf.reduce_sum(observed_class ) / tf.cast(tf.shape(y_true)[0], dtype=tf.float32)

    return loss

def weights_cross_entropy_loss(y_true, y_pred):
    from config import init_w_old as init_w
    class_hot = y_true
    y_true = tf.argmax(class_hot, axis=-1)

    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)

    # Observed class loss
    observed_class = -tf.math.log(y_pred_softmax + 1e-7) * class_hot *init_w
    # Total loss
    loss = tf.reduce_sum(observed_class ) / tf.cast(tf.shape(y_true)[0], dtype=tf.float32)

    return loss


def ordinal_loss(y_true, y_pred):
    from config import init_weights_ordinal_loss

    class_hot = y_true
    y_true = tf.argmax(class_hot, axis=-1)

    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)

    init_weights_ordinal_loss = tf.convert_to_tensor(init_weights_ordinal_loss, dtype=tf.float32)

    def select_columns_by_indices(penalty_matrix, y_true_indices):
        y_true_indices = tf.cast(y_true_indices, dtype=tf.int32)
        selected_weights = tf.gather(penalty_matrix, y_true_indices, axis=1)
        return selected_weights

    weights_ = select_columns_by_indices(init_weights_ordinal_loss, y_true)
    weights_ = tf.transpose(weights_)

    true_indices = tf.argmax(class_hot, axis=-1)
    true_class_mask = tf.one_hot(true_indices, depth=tf.shape(y_pred_softmax)[1])

    q = (1.0 - y_pred_softmax) * true_class_mask + y_pred_softmax * (1.0 - true_class_mask)

    weighted_q = weights_ * q

    loss_per_sample = tf.reduce_sum(weighted_q, axis=-1)
    loss = tf.reduce_mean(loss_per_sample)

    return loss


def ordinal_cross_entropy_loss(y_true, y_pred):
    from config import init_weights_
    class_hot = y_true
    y_true = tf.argmax(class_hot, axis=-1)



    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)

    def select_rows_by_indices(init_weights_, y_true):
        init_weights_ = tf.convert_to_tensor(init_weights_, dtype=tf.float32)
        y_true_int = tf.cast(y_true, dtype=tf.int32)
        selected_weights = tf.gather(init_weights_, y_true_int)
        return selected_weights

    weights_ = select_rows_by_indices(init_weights_, y_true)
    weights_ = tf.reshape(weights_, [-1, tf.shape(y_pred_softmax)[1]])

    # Observed class loss
    observed_class = -tf.math.log(y_pred_softmax + 1e-7) * class_hot *weights_

    # Unobserved class loss
    unobserved_class = -tf.math.log(1 - y_pred_softmax + 1e-7) *(1 - class_hot)* weights_

    # Total loss
    loss = tf.reduce_sum(observed_class + unobserved_class) / tf.cast(tf.shape(y_true)[0], dtype=tf.float32)

    return loss

# Compute categorical cross-entropy applying regularization based on beta distribution to targets.
def categorical_ce_beta_regularized(num_classes, eta=1.0):
    # Params [a,b] for beta distribution
    params = {}

    params['4'] = [
        [1,6],
        [6,10],
        [9,6],
        [6,1]
    ]

    params['5'] = [
        [1,9], # [2, 18], # [1, 9],
        [6,14], # [6, 14], # [3, 7],
        [12,12], # [10, 10], # [5, 5],
        [14,6], # [14, 6], # [7, 3],
        [9,1] # [18, 2] # [9, 1]
    ]

    params['6'] = [
        [1,10],
        [7,20],
        [15,20],
        [20,15],
        [20,7],
        [10,1]
    ]

    params['8'] = [
        [1,14],
        [7,31],
        [17,37],
        [27,35],
        [35,27],
        [37,17],
        [31,7],
        [14,1]
    ]

    # Precompute class probabilities for each label
    cls_probs = []
    for i in range(0, num_classes):
        cls_probs.append(get_beta_probabilities(num_classes, params[str(num_classes)][i][0], params[str(num_classes)][i][1]))
    cls_probs = tf.constant(cls_probs, dtype=tf.float32)

    def _compute(y_true, y_pred):
        y_prob = map_fn(lambda y: tf.cast(gather(cls_probs, argmax(y)), tf.float32), y_true)
        y_true = (1 - eta) * y_true + eta * y_prob

        return categorical_crossentropy(y_true, y_pred, from_logits=True)

    return _compute

    # Compute categorical cross-entropy applying regularization based on poisson distribution to targets.
def categorical_ce_poisson_regularized(num_classes, eta=1.0):
	cls_probs = get_poisson_probabilities(num_classes)

	def _compute(y_true, y_pred):
		y_prob = map_fn(lambda y: tf.cast(gather(cls_probs, argmax(y)), tf.float32), y_true)
		y_true = (1 - eta) * y_true + eta * y_prob

		return categorical_crossentropy(y_true, y_pred, from_logits=True)

	return _compute

# Compute categorical cross-entropy applying regularization based on binomial distribution to targets.
def categorical_ce_binomial_regularized(num_classes, eta=1.0):
	cls_probs = get_binominal_probabilities(num_classes)

	def _compute(y_true, y_pred):
		y_prob = map_fn(lambda y: tf.cast(gather(cls_probs, argmax(y)), tf.float32), y_true)
		y_true = (1 - eta) * y_true + eta * y_prob

		return categorical_crossentropy(y_true, y_pred, from_logits=True)

	return _compute


# Compute categorical cross-entropy applying regularization based on exponential distribution to targets.
def categorical_ce_exponential_regularized(num_classes, eta=1.0, tau=1.0):
	cls_probs = get_exponential_probabilities(num_classes, tau)

	def _compute(y_true, y_pred):
		y_prob = map_fn(lambda y: tf.cast(gather(cls_probs, argmax(y)), tf.float32), y_true)
		y_true = (1 - eta) * y_true + eta * y_prob

		return categorical_crossentropy(y_true, y_pred, from_logits=True)

	return _compute

def ordinal_ce_beta_regularized(num_classes, eta=1.0):
    from config import init_weights_ 
    # Params [a,b] for beta distribution
    params = {}

    params['4'] = [
        [1, 6],
        [6, 10],
        [9, 6],
        [6, 1]
    ]

    params['5'] = [
        [1, 9],
        [6, 14],
        [12, 12],
        [14, 6],
        [9, 1]
    ]

    params['6'] = [
        [1, 10],
        [7, 20],
        [15, 20],
        [20, 15],
        [20, 7],
        [10, 1]
    ]

    params['8'] = [
        [1, 14],
        [7, 31],
        [17, 37],
        [27, 35],
        [35, 27],
        [37, 17],
        [31, 7],
        [14, 1]
    ]

    # Precompute class probabilities for each label using your helper
    cls_probs = []
    for i in range(num_classes):
        a, b = params[str(num_classes)][i]
        cls_probs.append(get_beta_probabilities(num_classes, a, b))
    cls_probs = tf.constant(cls_probs, dtype=tf.float32)

    def _compute(y_true, y_pred):
        """
        y_true: one-hot labels, shape (batch_size, num_classes)
        y_pred: logits or probabilities, shape (batch_size, num_classes)
        """
        # ----- 1. Build q'(i) = (1-eta)*one_hot + eta*beta_prior -----
        # indices of the true classes
        true_idx = tf.argmax(y_true, axis=-1)  # (batch_size,)

        # beta prior for each sample, shape (batch_size, num_classes)
        beta_targets = tf.gather(cls_probs, true_idx)

        # q'(i): smoothed / unimodal targets
        q = (1.0 - eta) * tf.cast(y_true, tf.float32) + eta * beta_targets

        # ----- 2. Plug q into your Ordinal Cross-Entropy (OCE) -----
        class_hot = q   # this replaces y_{k,i} in our formula

        # still need an index vector for selecting weights
        y_true_indices = tf.argmax(class_hot, axis=-1)

        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)

        # same helper as in ordinal_cross_entropy_loss
        def select_rows_by_indices(init_w, y_idx):
            init_w = tf.convert_to_tensor(init_w, dtype=tf.float32)
            y_int = tf.cast(y_idx, dtype=tf.int32)
            return tf.gather(init_w, y_int)

        weights_ = select_rows_by_indices(init_weights_, y_true_indices)
        weights_ = tf.reshape(weights_, [-1, tf.shape(y_pred_softmax)[1]])

        # Observed class loss term
        observed_class = -tf.math.log(y_pred_softmax + 1e-7) * class_hot * weights_

        # Unobserved class loss term
        unobserved_class = -tf.math.log(1.0 - y_pred_softmax + 1e-7) * (1.0 - class_hot) * weights_

        # Total loss: average over batch
        loss = tf.reduce_sum(observed_class + unobserved_class) / tf.cast(
            tf.shape(y_true_indices)[0], dtype=tf.float32
        )

        return loss

    return _compute

def oce_exponential(num_classes, eta=1.0, tau=1.0):
	cls_probs = get_exponential_probabilities(num_classes, tau)

	def _compute(y_true, y_pred):
		y_prob = map_fn(lambda y: tf.cast(gather(cls_probs, argmax(y)), tf.float32), y_true)
		y_true = (1 - eta) * y_true + eta * y_prob

		return ordinal_cross_entropy_loss(y_true, y_pred)

	return _compute

def test_loss():
    from config import y_true, y_pred
    import numpy as np
    y_true_np=np.array(y_true)
    y_pred_np=np.array(y_pred)
    # Convert y_true to one-hot encoded format
    num_classes = y_pred_np.shape[1]
    y_true_one_hot = np.eye(num_classes)[y_true_np]

    # Convert to TensorFlow tensors
    y_true = tf.convert_to_tensor(y_true_one_hot, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred_np, dtype=tf.float32)

    print("Custom Cross Entropy Loss:", cross_entropy_loss(y_true, y_pred).numpy())
    print("Weights Cross Entropy Loss:", weights_cross_entropy_loss(y_true, y_pred).numpy())
    print("Ordinal Loss:", ordinal_loss(y_true, y_pred).numpy())
    print("Ordinal Cross Entropy Loss:", ordinal_cross_entropy_loss(y_true, y_pred).numpy())
    print("Categorical CE Beta Regularized Loss:", categorical_ce_beta_regularized(num_classes=5, eta=0.85)(y_true, y_pred).numpy())
    print("Categorical CE Poisson Regularized Loss:", categorical_ce_poisson_regularized(num_classes=5, eta=0.85)(y_true, y_pred).numpy())
    print("Categorical CE Binomial Regularized Loss:", categorical_ce_binomial_regularized(num_classes=5, eta=0.85)(y_true, y_pred).numpy())
    print("Categorical CE Exponential Regularized Loss:", categorical_ce_exponential_regularized(num_classes=5, eta=0.85, tau=1.0)(y_true, y_pred).numpy())
    print("oce_exponential Loss:", oce_exponential(num_classes=5, eta=0.85, tau=1.0)(y_true, y_pred).numpy())


test_loss()