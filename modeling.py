import tensorflow as tf


class DCN(tf.keras.Model):

    def __init__(self, deep_layer_sizes, projection_dim=None, vocabularies=None):
        super().__init__()
        self.embedding_dimension = 32
        str_features = ["movie_id", "user_id", "user_zip_code", "user_occupation_text"]
        int_features = ["user_gender", "bucketized_user_age"]
        self._all_features = str_features + int_features
        self._embeddings = {}
        # Compute embeddings for string features.
        for feature_name in str_features:
            vocabulary = vocabularies[feature_name]
            self._embeddings[feature_name] = tf.keras.Sequential(
                [tf.keras.layers.experimental.preprocessing.StringLookup(
                    vocabulary=vocabulary, mask_token=None),
                    tf.keras.layers.Embedding(len(vocabulary) + 1,
                                              self.embedding_dimension)
                ])
        # Compute embeddings for int features.
        for feature_name in int_features:
            vocabulary = vocabularies[feature_name]
            self._embeddings[feature_name] = tf.keras.Sequential(
                [tf.keras.layers.experimental.preprocessing.IntegerLookup(
                    vocabulary=vocabulary, mask_value=None),
                    tf.keras.layers.Embedding(len(vocabulary) + 1,
                                              self.embedding_dimension)
                ])
        self._cross_layer = Cross(projection_dim=projection_dim, kernel_initializer="glorot_uniform")
        self._deep_layers = [tf.keras.layers.Dense(layer_size, activation="relu")
                             for layer_size in deep_layer_sizes]
        self._logit_layer = tf.keras.layers.Dense(1)

    def train_step(self, features):
        # Set up a gradient tape to record gradients.
        with tf.GradientTape() as tape:
            # Concatenate embeddings
            embeddings = []
            for feature_name in self._all_features:
                embedding_fn = self._embeddings[feature_name]
                embeddings.append(embedding_fn(features[feature_name]))
            x = tf.concat(embeddings, axis=1)
            x = self._cross_layer1(x)
            # Build Deep Network
            for deep_layer in self._deep_layers:
                x = deep_layer(x)
            labels = features.pop("user_rating")
            scores = self._logit_layer(x)
            loss = self.compiled_loss(labels, scores)
            # Handle regularization losses as well.
            regularization_loss = sum(self.losses)
            total_loss = loss + regularization_loss
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(labels, scores)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, features):
        # Loss computation.
        embeddings = []
        for feature_name in self._all_features:
            embedding_fn = self._embeddings[feature_name]
            embeddings.append(embedding_fn(features[feature_name]))
        x = tf.concat(embeddings, axis=1)
        x = self._cross_layer(x)
        # Build Deep Network
        for deep_layer in self._deep_layers:
            x = deep_layer(x)
        labels = features.pop("user_rating")
        scores = self._logit_layer(x)
        loss = self.compiled_loss(labels, scores)
        # Handle regularization losses as well.
        regularization_loss = sum(self.losses)
        total_loss = loss + regularization_loss
        self.compiled_metrics.update_state(labels, scores)
        return {m.name: m.result() for m in self.metrics}



class Cross(tf.keras.layers.Layer):
    """Cross Layer in Deep & Cross Network to learn explicit feature interactions.
    A layer that creates explicit and bounded-degree feature interactions
    efficiently. The `call` method accepts `inputs` as a tuple of size 2
    tensors. The first input `x0` is the base layer that contains the original
    features (usually the embedding layer); the second input `xi` is the output
    of the previous `Cross` layer in the stack, i.e., the i-th `Cross`
    layer. For the first `Cross` layer in the stack, x0 = xi.
    The output is x_{i+1} = x0 .* (W * xi + bias + diag_scale * xi) + xi,
    where .* designates elementwise multiplication, W could be a full-rank
    matrix, or a low-rank matrix U*V to reduce the computational cost, and
    diag_scale increases the diagonal of W to improve training stability (
    especially for the low-rank case).
    Example:

        ```python
        # after embedding layer in a functional model:
        input = tf.keras.Input(shape=(None,), name='index', dtype=tf.int64)
        x0 = tf.keras.layers.Embedding(input_dim=32, output_dim=6)
        x1 = Cross()(x0, x0)
        x2 = Cross()(x0, x1)
        logits = tf.keras.layers.Dense(units=10)(x2)
        model = tf.keras.Model(input, logits)
        ```

    Args:
        projection_dim: project dimension to reduce the computational cost.
          Default is `None` such that a full (`input_dim` by `input_dim`) matrix
          W is used. If enabled, a low-rank matrix W = U*V will be used, where U
          is of size `input_dim` by `projection_dim` and V is of size
          `projection_dim` by `input_dim`. `projection_dim` need to be smaller
          than `input_dim`/2 to improve the model efficiency. In practice, we've
          observed that `projection_dim` = d/4 consistently preserved the
          accuracy of a full-rank version.
        diag_scale: a non-negative float used to increase the diagonal of the
          kernel W by `diag_scale`, that is, W + diag_scale * I, where I is an
          identity matrix.
        use_bias: whether to add a bias term for this layer. If set to False,
          no bias term will be used.
        kernel_initializer: Initializer to use on the kernel matrix.
        bias_initializer: Initializer to use on the bias vector.
        kernel_regularizer: Regularizer to use on the kernel matrix.
        bias_regularizer: Regularizer to use on bias vector.

    Input shape: A tuple of 2 (batch_size, `input_dim`) dimensional inputs.
    Output shape: A single (batch_size, `input_dim`) dimensional output.
  """

    def __init__(self, projection_dim=None, diag_scale=0.0, use_bias=True, kernel_initializer="truncated_normal",
                 bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, **kwargs):
        super(Cross, self).__init__(**kwargs)
        self._projection_dim = projection_dim
        self._diag_scale = diag_scale
        self._use_bias = use_bias
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._input_dim = None
        self._supports_masking = True
        if self._diag_scale < 0:
            raise ValueError(
                "`diag_scale` should be non-negative. Got `diag_scale` = {}".format(
                    self._diag_scale))

    def build(self, input_shape):
        last_dim = input_shape[-1]
        if self._projection_dim is None:
            self._dense = tf.keras.layers.Dense(
                last_dim,
                kernel_initializer=self._kernel_initializer,
                bias_initializer=self._bias_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                use_bias=self._use_bias,
            )
        else:
            if self._projection_dim < 0 or self._projection_dim > last_dim / 2:
                raise ValueError(
                    "`projection_dim` should be smaller than last_dim / 2 to improve "
                    "the model efficiency, and should be positive. Got "
                    "`projection_dim` {}, and last dimension of input {}".format(
                        self._projection_dim, last_dim))
            self._dense_u = tf.keras.layers.Dense(
                self._projection_dim,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                use_bias=False,
            )
            self._dense_v = tf.keras.layers.Dense(
                last_dim,
                kernel_initializer=self._kernel_initializer,
                bias_initializer=self._bias_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                use_bias=self._use_bias,
            )
        self.built = True

    def call(self, x0, x=None):
        """Computes the feature cross.
        Args:
          x0: The input tensor
          x: Optional second input tensor. If provided, the layer will compute
            crosses between x0 and x; if not provided, the layer will compute
            crosses between x0 and itself.
        Returns:
         Tensor of crosses.
        """
        if not self.built:
            self.build(x0.shape)
        if x is None:
            x = x0
        if x0.shape[-1] != x.shape[-1]:
            raise ValueError(
                "`x0` and `x` dimension mismatch! Got `x0` dimension {}, and x "
                "dimension {}. This case is not supported yet.".format(
                    x0.shape[-1], x.shape[-1]))
        if self._projection_dim is None:
            prod_output = self._dense(x)
        else:
            prod_output = self._dense_v(self._dense_u(x))
        if self._diag_scale:
            prod_output = prod_output + self._diag_scale * x
        res = x0 * x + x
        return res

    def get_config(self):
        config = {
            "projection_dim":
                self._projection_dim,
            "diag_scale":
                self._diag_scale,
            "use_bias":
                self._use_bias,
            "kernel_initializer":
                tf.keras.initializers.serialize(self._kernel_initializer),
            "bias_initializer":
                tf.keras.initializers.serialize(self._bias_initializer),
            "kernel_regularizer":
                tf.keras.regularizers.serialize(self._kernel_regularizer),
            "bias_regularizer":
                tf.keras.regularizers.serialize(self._bias_regularizer),
        }
        base_config = super(Cross, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


