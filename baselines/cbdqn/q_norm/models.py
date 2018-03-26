import tensorflow as tf
import tensorflow.contrib.layers as layers


def _mlp(hiddens, nbins, inpt, num_actions, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
            if layer_norm:
                out = layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.relu(out)
        # Q values
        q_out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        # Q bins conv version
        #q_out = tf.expand_dims(q_out, -1)
        #print("pre_conv: q_out.get_shape(): ", q_out.get_shape())
        #q_out = layers.convolution2d(q_out,
        #                             num_outputs=nbins,
        #                             kernel_size=1,
        #                             stride=1)
        ##q_out = tf.squeeze(q_out)
        #print("post_conv: q_out.get_shape(): ", q_out.get_shape())
        #return q_out 
        
        # Q bins
        q_out_bin = layers.fully_connected(q_out, num_outputs=num_actions*nbins, activation_fn=None)# discretization
        # 2D reshape of Q bins
        q_out_2D = tf.reshape(q_out_bin,[tf.shape(q_out_bin)[0],num_actions,nbins]) # in order to use (act network) local max for each bin for all actions
        return q_out, q_out_2D


def mlp(nbins, hiddens=[], layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _mlp(hiddens, nbins, layer_norm=layer_norm, *args, **kwargs)

def softmax(target, axis, name=None):
  with tf.name_scope(name, 'softmax', values=[target]):
    max_axis = tf.reduce_max(target, axis, keep_dims=True)
    target_exp = tf.exp(target-max_axis)
    normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
    softmax = target_exp / normalize
    return softmax

def _cnn_to_mlp(convs, hiddens,nbins, dueling, inpt, num_actions, scope, reuse=False, layer_norm=False):

    # 9
    #print("_cnn_to_mlp::num_actions: ", num_actions)

    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
        conv_out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = conv_out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                if layer_norm:
                    action_out = layers.layer_norm(action_out, center=True, scale=True)
                action_out = tf.nn.relu(action_out)

            # num_units = num_actions
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)# regression action values
            
            # num_units = num_actions * nbins
            # shape: [batch_size, num_actions * nbins]
            action_scores_bin = layers.fully_connected(action_out, num_outputs=num_actions*nbins, activation_fn=None)# discretization
            # [-1, 900]
            #print("action_scores_bin.get_shape(): ",
            #        action_scores_bin.get_shape())
            
            # [batch_size, num_actions, n_bin]
            action_scores_bin_rs = tf.reshape(action_scores_bin,[tf.shape(action_scores_bin)[0],num_actions,nbins]) # in order to use (act network) local max for each bin for all actions

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = conv_out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        state_out = layers.layer_norm(state_out, center=True, scale=True)
                    state_out = tf.nn.relu(state_out)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores

        return action_scores_bin, action_scores_bin_rs


def cnn_to_mlp(convs, hiddens,nbins =200, dueling=False, layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """

    return lambda *args, **kwargs: _cnn_to_mlp(convs, hiddens,nbins, dueling, layer_norm=layer_norm, *args, **kwargs)

