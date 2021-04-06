import tensorflow as tf


class tDCN(tf.keras.Model):

    def __init__(self, vocabularies=None):
        super().__init__()
        self.embedding_dimension = 8
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
        self._tcross_layer = tCross(embedding_dimension=self.embedding_dimension,
                                    num_features=len(self._all_features),
                                    kernel_initializer="glorot_uniform")
        self._deep_layer = tf.keras.layers.Dense(192, activation="relu")
        self._wide_layer = tf.keras.layers.Dense(192, activation="relu")
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
            x0 = self._tcross_layer(x)
            x1 = self._wide_layer(x)
            x = tf.concat([x0, x1], axis=1)
            x = self._deep_layer(x)
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
        x0 = self._tcross_layer(x)
        x1 = self._wide_layer(x)
        x = tf.concat([x0, x1], axis=1)
        x = self._deep_layer(x)
        labels = features.pop("user_rating")
        scores = self._logit_layer(x)
        loss = self.compiled_loss(labels, scores)
        # Handle regularization losses as well.
        regularization_loss = sum(self.losses)
        total_loss = loss + regularization_loss
        self.compiled_metrics.update_state(labels, scores)
        return {m.name: m.result() for m in self.metrics}


class tCross(tf.keras.layers.Layer):

    def __init__(self, use_bias=True, num_features=None, embedding_dimension=None,
                 kernel_initializer="truncated_normal", bias_initializer="zeros", kernel_regularizer=None,
                 bias_regularizer=None, **kwargs):
        super(tCross, self).__init__(**kwargs)
        self._use_bias = use_bias
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._input_dim = None
        self._supports_masking = True
        self.num_features = num_features
        self.embedding_dimension = embedding_dimension

    def build(self, input_shape):
        last_dim = input_shape[-1]
        self._key = tf.keras.layers.Dense(
            last_dim,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            use_bias=self._use_bias,
            activation='gelu'
        )
        self._query = tf.keras.layers.Dense(
            last_dim,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            use_bias=self._use_bias,
            activation='gelu'
        )
        self._value = tf.keras.layers.Dense(
            last_dim,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            use_bias=self._use_bias,
            activation='gelu'
        )
        self._dense = tf.keras.layers.Dense(
            last_dim,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            use_bias=self._use_bias,
        )
        self._act_fn = tf.keras.layers.Softmax()
        self._dropout = tf.keras.layers.Dropout(0.15)
        self._layernorm = tf.keras.layers.LayerNormalization(axis=1, epsilon=1e-5)
        self.built = True

    def call(self, x):
        if not self.built:
            self.build(x.shape)
        key = self._key(x)
        query = self._query(x)
        value = self._value(x)
        key = tf.reshape(key, [-1, self.num_features, self.embedding_dimension])
        query = tf.reshape(query, [-1, self.num_features, self.embedding_dimension])
        value = tf.reshape(value, [-1, self.num_features, self.embedding_dimension])
        # x = x + tf.transpose(x)
        y = tf.matmul(tf.transpose(key, perm=[0, 2, 1]), query)
        # y = self._act_fn(y)
        # y = self._dropout(y)
        res = tf.matmul(value, y)
        res = tf.reshape(res, [-1, x.shape[-1]])
        # res = self._dense(res + x)
        # res = self._layernorm(res)
        return res + x

    def get_config(self):
        config = {
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
        base_config = super(tCross, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
