import tensorflow as tf

def fully_connected(input, output_shape, initializer, scope="fc", is_last=False, is_decoder=False):
    with tf.variable_scope(scope):
        input_shape = input.get_shape()[-1].value
        W = tf.get_variable("weight", [input_shape, output_shape], initializer=initializer)
        b = tf.get_variable("bias", [output_shape], initializer=initializer)
        fc = tf.add(tf.matmul(input, W), b)
        fc = normalize(fc)

        if not is_last:
            if not is_decoder:
                output = tf.nn.relu(fc)
            else:
                output = lrelu(fc)
        else:
            output = fc

    return output

def normalize(inputs,
              type="bn",
              decay=.99,
              is_training=True,
              activation_fn=None,
              scope="normalize"):
    '''Applies {batch|layer} normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`. If type is `bn`, the normalization is over all but
        the last dimension. Or if type is `ln`, the normalization is over
        the last dimension. Note that this is different from the native
        `tf.contrib.layers.batch_norm`. For this I recommend you change
        a line in ``tensorflow/contrib/layers/python/layers/layer.py`
        as follows.
        Before: mean, variance = nn.moments(inputs, axis, keep_dims=True)
        After: mean, variance = nn.moments(inputs, [-1], keep_dims=True)
      type: A string. Either "bn" or "ln".
      decay: Decay for the moving average. Reasonable values for `decay` are close
        to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
        Lower `decay` value (recommend trying `decay`=0.9) if model experiences
        reasonably good training performance but poor validation and/or test
        performance.
      is_training: Whether or not the layer is in training mode. W
      activation_fn: Activation function.
      scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    if type == "bn":
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims

        # use fused batch norm if inputs_rank in [2, 3, 4] as it is much faster.
        # pay attention to the fact that fused_batch_norm requires shape to be rank 4 of NHWC.
        if inputs_rank in [2, 3, 4]:
            if inputs_rank == 2:
                inputs = tf.expand_dims(inputs, axis=1)
                inputs = tf.expand_dims(inputs, axis=2)
            elif inputs_rank == 3:
                inputs = tf.expand_dims(inputs, axis=1)

            outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                                   decay=decay,
                                                   center=True,
                                                   scale=True,
                                                   activation_fn=activation_fn,
                                                   updates_collections=None,
                                                   is_training=is_training,
                                                   scope=scope,
                                                   zero_debias_moving_mean=True,
                                                   fused=True)
            # restore original shape
            if inputs_rank == 2:
                outputs = tf.squeeze(outputs, axis=[1, 2])
            elif inputs_rank == 3:
                outputs = tf.squeeze(outputs, axis=1)
        else:  # fallback to naive batch norm
            outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                                   decay=decay,
                                                   center=True,
                                                   scale=True,
                                                   activation_fn=activation_fn,
                                                   updates_collections=None,
                                                   is_training=is_training,
                                                   scope=scope,
                                                   fused=False)
    elif type == "ln":
        outputs = tf.contrib.layers.layer_norm(inputs=inputs,
                                               center=True,
                                               scale=True,
                                               activation_fn=activation_fn,
                                               scope=scope)
    elif type == "in":
        with tf.variable_scope(scope):
            batch, steps, channels = inputs.get_shape().as_list()
            var_shape = [channels]
            mu, sigma_sq = tf.nn.moments(inputs, [1], keep_dims=True)
            shift = tf.Variable(tf.zeros(var_shape))
            scale = tf.Variable(tf.ones(var_shape))
            epsilon = 1e-8
            normalized = (inputs - mu) / (sigma_sq + epsilon) ** (.5)
            outputs = scale * normalized + shift
            if activation_fn:
                outputs = activation_fn(outputs)
    else:
        raise ValueError("Currently we support `bn` or `ln` only.")

    return outputs



def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)