"""Deep Q learning graph

The functions in this file can are used to create the following functions:

======= act ========

    Function to chose an action given an observation

    Parameters
    ----------
    observation: object
        Observation that can be feed into the output of make_obs_ph
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps_ph: float
        update epsilon a new value, if negative not update happens
        (default: no update)

    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.


======= act (in case of parameter noise) ========

    Function to chose an action given an observation

    Parameters
    ----------
    observation: object
        Observation that can be feed into the output of make_obs_ph
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps_ph: float
        update epsilon a new value, if negative not update happens
        (default: no update)
    reset_ph: bool
        reset the perturbed policy by sampling a new perturbation
    update_param_noise_threshold_ph: float
        the desired threshold for the difference between non-perturbed and perturbed policy
    update_param_noise_scale_ph: bool
        whether or not to update the scale of the noise for the next time it is re-perturbed

    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.


======= train =======

    Function that takes a transition (s,a,r,s') and optimizes Bellman equation's error:

        td_error = Q(s,a) - (r + gamma * max_a' Q(s', a'))
        loss = huber_loss[td_error]

    Parameters
    ----------
    obs_t: object
        a batch of observations
    action: np.array
        actions that were selected upon seeing obs_t.
        dtype must be int32 and shape must be (batch_size,)
    reward: np.array
        immediate reward attained after executing those actions
        dtype must be float32 and shape must be (batch_size,)
    obs_tp1: object
        observations that followed obs_t
    done: np.array
        1 if obs_t was the last observation in the episode and 0 otherwise
        obs_tp1 gets ignored, but must be of the valid shape.
        dtype must be float32 and shape must be (batch_size,)
    weight: np.array
        imporance weights for every element of the batch (gradient is multiplied
        by the importance weight) dtype must be float32 and shape must be (batch_size,)

    Returns
    -------
    td_error: np.array
        a list of differences between Q(s,a) and the target in Bellman's equation.
        dtype is float32 and shape is (batch_size,)

======= update_target ========

    copy the parameters from optimized Q function to the target Q function.
    In Q learning we actually optimize the following error:

        Q(s,a) - (r + gamma * max_a' Q'(s', a'))

    Where Q' is lagging behind Q to stablize the learning. For example for Atari

    Q' is set to Q once every 10000 updates training steps.

"""
import tensorflow as tf
import baselines.common.tf_util as U
import numpy as np

def scope_vars(scope, trainable_only=False):
    """
    Get variables inside a scope
    The scope can be specified as a string
    Parameters
    ----------
    scope: str or VariableScope
        scope in which the variables reside.
    trainable_only: bool
        whether or not to return only the variables that were marked as trainable.
    Returns
    -------
    vars: [tf.Variable]
        list of variables in `scope`.
    """
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )


def scope_name():
    """Returns the name of current scope as a string, e.g. deepq/q_func"""
    return tf.get_variable_scope().name


def absolute_scope_name(relative_scope_name):
    """Appends parent scope name to `relative_scope_name`"""
    return scope_name() + "/" + relative_scope_name


def default_param_noise_filter(var):
    if var not in tf.trainable_variables():
        # We never perturb non-trainable vars.
        return False
    if "fully_connected" in var.name:
        # We perturb fully-connected layers.
        return True

    # The remaining layers are likely conv or layer norm layers, which we do not wish to
    # perturb (in the former case because they only extract features, in the latter case because
    # we use them for normalization purposes). If you change your network, you will likely want
    # to re-consider which layers to perturb and which to keep untouched.
    return False


def build_act(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = make_obs_ph("observation")
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

        _ , q_val= q_func(observations_ph.get(), num_actions, scope="q_func") #TODO in here we do not use q values in 1 D like num*bins but use 2d actions,values
        #TODO so we could choose max bin for each action and choose max action
        
        #[-1, 9, 100]
        print("build_act::q_val.get_shape(): ", q_val.get_shape())
        #print("q_val.shape: ", tf.shape(q_val))
        deterministic_actions = tf.argmax(tf.argmax(q_val, axis = 2), axis=1)

        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
        
        _act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                         outputs=[output_actions,deterministic_actions],
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])
        def act(ob, stochastic=True, update_eps=-1):
            return _act(ob, stochastic, update_eps)
        return act


def build_act_with_param_noise(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None, param_noise_filter_func=None):
    """Creates the act function with support for parameter space noise exploration (https://arxiv.org/abs/1706.01905):

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.
    param_noise_filter_func: tf.Variable -> bool
        function that decides whether or not a variable should be perturbed. Only applicable
        if param_noise is True. If set to None, default_param_noise_filter is used by default.

    Returns
    -------
    act: (tf.Variable, bool, float, bool, float, bool) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    if param_noise_filter_func is None:
        param_noise_filter_func = default_param_noise_filter

    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = make_obs_ph("observation")
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")
        update_param_noise_threshold_ph = tf.placeholder(tf.float32, (), name="update_param_noise_threshold")
        update_param_noise_scale_ph = tf.placeholder(tf.bool, (), name="update_param_noise_scale")
        reset_ph = tf.placeholder(tf.bool, (), name="reset")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))
        param_noise_scale = tf.get_variable("param_noise_scale", (), initializer=tf.constant_initializer(0.01), trainable=False)
        param_noise_threshold = tf.get_variable("param_noise_threshold", (), initializer=tf.constant_initializer(0.05), trainable=False)

        # Unmodified Q.
        q_values = q_func(observations_ph.get(), num_actions, scope="q_func")

        # Perturbable Q used for the actual rollout.
        q_values_perturbed = q_func(observations_ph.get(), num_actions, scope="perturbed_q_func")
        # We have to wrap this code into a function due to the way tf.cond() works. See
        # https://stackoverflow.com/questions/37063952/confused-by-the-behavior-of-tf-cond for
        # a more detailed discussion.
        def perturb_vars(original_scope, perturbed_scope):
            all_vars = scope_vars(absolute_scope_name(original_scope))
            all_perturbed_vars = scope_vars(absolute_scope_name(perturbed_scope))
            assert len(all_vars) == len(all_perturbed_vars)
            perturb_ops = []
            for var, perturbed_var in zip(all_vars, all_perturbed_vars):
                if param_noise_filter_func(perturbed_var):
                    # Perturb this variable.
                    op = tf.assign(perturbed_var, var + tf.random_normal(shape=tf.shape(var), mean=0., stddev=param_noise_scale))
                else:
                    # Do not perturb, just assign.
                    op = tf.assign(perturbed_var, var)
                perturb_ops.append(op)
            assert len(perturb_ops) == len(all_vars)
            return tf.group(*perturb_ops)

        # Set up functionality to re-compute `param_noise_scale`. This perturbs yet another copy
        # of the network and measures the effect of that perturbation in action space. If the perturbation
        # is too big, reduce scale of perturbation, otherwise increase.
        q_values_adaptive = q_func(observations_ph.get(), num_actions, scope="adaptive_q_func")
        perturb_for_adaption = perturb_vars(original_scope="q_func", perturbed_scope="adaptive_q_func")
        kl = tf.reduce_sum(tf.nn.softmax(q_values) * (tf.log(tf.nn.softmax(q_values)) - tf.log(tf.nn.softmax(q_values_adaptive))), axis=-1)
        mean_kl = tf.reduce_mean(kl)
        def update_scale():
            with tf.control_dependencies([perturb_for_adaption]):
                update_scale_expr = tf.cond(mean_kl < param_noise_threshold,
                    lambda: param_noise_scale.assign(param_noise_scale * 1.01),
                    lambda: param_noise_scale.assign(param_noise_scale / 1.01),
                )
            return update_scale_expr

        # Functionality to update the threshold for parameter space noise.
        update_param_noise_threshold_expr = param_noise_threshold.assign(tf.cond(update_param_noise_threshold_ph >= 0,
            lambda: update_param_noise_threshold_ph, lambda: param_noise_threshold))

        # Put everything together.
        deterministic_actions = tf.argmax(q_values_perturbed, axis=1)
        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
        updates = [
            update_eps_expr,
            tf.cond(reset_ph, lambda: perturb_vars(original_scope="q_func", perturbed_scope="perturbed_q_func"), lambda: tf.group(*[])),
            tf.cond(update_param_noise_scale_ph, lambda: update_scale(), lambda: tf.Variable(0., trainable=False)),
            update_param_noise_threshold_expr,
        ]
        _act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph, reset_ph, update_param_noise_threshold_ph, update_param_noise_scale_ph],
                         outputs=output_actions,
                         givens={update_eps_ph: -1.0, stochastic_ph: True, reset_ph: False, update_param_noise_threshold_ph: False, update_param_noise_scale_ph: False},
                         updates=updates)
        def act(ob, reset, update_param_noise_threshold, update_param_noise_scale, stochastic=True, update_eps=-1):
            return _act(ob, stochastic, update_eps, reset, update_param_noise_threshold, update_param_noise_scale)
        return act


def build_train(make_obs_ph, 
        q_func, 
        num_actions, 
        optimizer, 
        grad_norm_clipping=None, 
        gamma=1.0, 
        old_qmin = -100,
        old_qmax = 100,
        nbins = 200,
        new_qmin = -100,
        new_qmax = 100,
        double_q=False, 
        scope="deepq", 
        reuse=None, 
        param_noise=False, 
        param_noise_filter_func=None):
    """Creates the train function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that takes a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions
    reuse: bool
        whether or not to reuse the graph variables
    optimizer: tf.train.Optimizer
        optimizer to use for the Q-learning objective.
    grad_norm_clipping: float or None
        clip gradient norms to this value. If None no clipping is performed.
    gamma: float
        discount rate.
    double_q: bool
        if true will use Double Q Learning (https://arxiv.org/abs/1509.06461).
        In general it is a good idea to keep it enabled.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    param_noise_filter_func: tf.Variable -> bool
        function that decides whether or not a variable should be perturbed. Only applicable
        if param_noise is True. If set to None, default_param_noise_filter is used by default.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    train: (object, np.array, np.array, object, np.array, np.array) -> np.array
        optimize the error in Bellman's equation.
`       See the top of the file for details.
    update_target: () -> ()
        copy the parameters from optimized Q function to the target Q function.
`       See the top of the file for details.
    debug: {str: function}
        a bunch of functions to print debug data like q_values.
    """
    if param_noise:
        act_f = build_act_with_param_noise(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse,
            param_noise_filter_func=param_noise_filter_func)
    else:
        act_f = build_act(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse)

    print("build_train::num_actions: ", num_actions) #OK


    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = make_obs_ph("obs_t")
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = make_obs_ph("obs_tp1")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # q network evaluation
        q_t,q_t2D = q_func(obs_t_input.get(), num_actions, scope="q_func", reuse=True)  # (1D num_actions* bins,2D num_actions,values)
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/q_func")

        # target q network evalution
        q_tp1,q_tp12D = q_func(obs_tp1_input.get(), num_actions, scope="target_q_func")
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/target_q_func")
        
        # DEBUG
        #print("tf.shape(act_t_ph): ", tf.shape(act_t_ph))
        #print("q_t.get_shape()): ", q_t.get_shape())
        #print("q_t2D.get_shape()): ", q_t2D.get_shape())
        #print("act_t_ph.get_shape()): ", act_t_ph.get_shape())
        #print("size.get_shape()): ", size.get_shape())
       

        # Compute train logits
        logits_list = []
        for i in range(32):
            logits_list.append( tf.nn.softmax(q_t2D[i,act_t_ph[i],:]) )
        logits_train = tf.stack(logits_list) # v_dist_t_selected
        #print("logits_train.get_shape()): ", logits_train.get_shape())
       

        # For each action, compute average Q over bins
        delta_z = (old_qmax - old_qmin)/(nbins-1)
        start = old_qmin
        end = old_qmax + delta_z
        z = tf.range(start, end, delta_z)
        print("z.get_shape: ", z.get_shape())
        
        # Q_{target}(\phi_{j+1},a,\theta)
        q_as = []
        for action in range(num_actions):
            dist = tf.nn.softmax(q_tp1[:,nbins*action:nbins*(action+1)])
            print("dist.get_shape: ", dist.get_shape())
            q_a = tf.reduce_sum(tf.multiply(dist, z), axis=1, keep_dims=True)
            q_as.append(q_a)

        # max_a Q_{target}(\phi_{j+1},a,\theta)
        q_target_avg = tf.concat(q_as, axis=1)
        q_tp1_best = tf.reduce_max(q_target_avg, 1) # a^*
        q_tp1_best_act = U.argmax(q_tp1_best, axis=1)
        q_tp1_best_act = tf.cast(q_tp1_best_act, tf.int32)
        print("q_tp1_best.get_shape(): ", q_tp1_best.get_shape())


        # compute RHS of bellman equation
        #q_tp1_best_masked = (1.0 - done_mask_ph) * tot_val
        #q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked# target Q value
        #q_t_selected_target_clip = tf.clip_by_value(q_t_selected_target,old_qmin,old_qmax)


        # extract P_(x_{t+1},a*)
        logits_list = []
        for i in range(32):
            logits_list.append( tf.nn.softmax(q_tp12D[i,act_t_ph[i],:]) )
        logits_target = tf.stack(logits_list) # v_dist_tp1_selected
        print("logits.get_shape()): ", logits_target.get_shape())

        # Cross entropy from (DIST_RL)
        z = tf.tile(tf.reshape(tf.range(old_qmin, old_qmax + delta_z, delta_z), [1,
            nbins]), [batch_size, 1])
        r = tf.tile(tf.reshape(rew_t_ph, [batch_size, 1]), [1, nbins])
        done = tf.tile(tf.reshape(done_mask_ph, [batch_size, 1]), [1,
            nbins])

        Tz = r + z * gamma * (1-done)
        T_z = tf.maximum(tf.minimum(T_z, old_qmax), old_qmin)
        b = (Tz - old_qmin)/delta_z # Should be float
        l,u = tf.floor(b), tf.ceil(b)
        l_id = tf.cast(l, tf.int32)
        u_id = tf.cast(u, tf.int32)
        
        v_t_dist_selected = tf.reshape(logits_train,[-1]) # P(x_t, a_t)
        add_index = tf.range(batch_size) * nbins

        err = tf.zeros([batch_size])

        for j in range(nbins):
            l_index = l_id[:, j] + add_index
            u_index = u_id[:, j] + add_index

            p_tl = tf.gather(v_dist_t_selected, l_index)
            p_tu = tf.gather(v_dist_t_selected, u_index)

            log_p_tl = tf.log(p_tl)
            log_p_tu = tf.log(p_tu)
            p_tp1 = logits_target[:,j]

            err = err + p_tp1 * ((u[:,j] - b[:,j]) * log_p_tl + (b[:,j] -
                l[:,j]) * log_p_tu)

        err = tf.negative(err)

        # compute the error (potentially clipped)
        weighted_error = tf.reduce_mean(importance_weights_ph * err)

        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            gradients = optimizer.compute_gradients(weighted_error, var_list=q_func_vars)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
            optimize_expr = optimizer.apply_gradients(gradients)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                importance_weights_ph
            ],
            outputs=[new_errors],

            updates=[optimize_expr]
        )
        val = U.function( # this is added only to monitor if values are calculated correctly
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                importance_weights_ph
            ],
            outputs=[new_errors],
            #,q_t,q_tp1, q_t_selected , q_t_selected_target , q_tp1_best ,tot_val,q_t_val  ]
        )
        update_target = U.function([], [], updates=[update_target_expr])

        q_values = U.function([obs_t_input], q_t2D)

        return act_f, train, update_target, {'q_values': q_values} , val

def q_value_avg(q_dist, nbins, num_actions, old_qmin, old_qmax, delta_z):

    start = old_qmin
    end = old_qmax + delta_z
    delta = delta_z
    z = tf.range(start, end, delta)

    q_as = []

    for action in range(num_actions):
        dist = q_dist[:, nbins*action: nbins*(action+1)]
        print("dist.get_shape(): ", dist.get_shape())
        print("z.get_shape(): ", z.get_shape())
        q_a = tf.reduce_sum(tf.multiply(dist, z), axis = 1, keep_dims = True)
        q_as.append(q_a)

    q_values = tf.concat(q_as, axis=1)

    return q_values

def build_dist_act(make_obs_ph, dist_func, num_actions, nbins, old_qmin, old_qmax, scope="deepq", reuse=None):
    delta_z = (old_qmax - old_qmin) / (nbins - 1)
    
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = make_obs_ph("observation")
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

        q_dist = dist_func(observations_ph.get(), num_actions, scope="dist_func")
        q_values = q_value_avg(q_dist, nbins, num_actions, old_qmin, old_qmax, delta_z)
        deterministic_actions = tf.argmax(q_values, axis=1)

        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))

        _act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                         outputs=[output_actions,deterministic_actions],
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])
        def act(ob, stochastic=True, update_eps=-1):
            return _act(ob, stochastic, update_eps)
        return act

def build_dist_train(make_obs_ph, 
        q_func, 
        num_actions, 
        optimizer, 
        grad_norm_clipping=None, 
        gamma=1.0, 
        old_qmin = -100,
        old_qmax = 100,
        nbins = 200,
        new_qmin = -100,
        new_qmax = 100,
        double_q=False, 
        scope="deepq", 
        reuse=None, 
        param_noise=False, 
        param_noise_filter_func=None):
    
    act_f = build_dist_act(make_obs_ph, q_func, num_actions, nbins, old_qmin, old_qmax, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = make_obs_ph("obs_t")
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = make_obs_ph("obs_tp1")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # value distribution network evaluation
        # p(x_t, a)
        q_t = q_func(obs_t_input.get(), num_actions, scope="dist_func", reuse=True)  # reuse parameters from act
        q_t_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/dist_func")

        # target value distribution network evalution
        # p(x_(t+1), a)
        q_tp1 = q_func(obs_tp1_input.get(), num_actions, scope="target_dist_func")
        q_tp1_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/target_dist_func")

        # (0) Calculate p(x_t, a_t)
        # x_t is given by ob_t_input, and a_t is given by act_t_ph
        batch_size = tf.shape(obs_t_input.get())[0]

        v_index1 = tf.range(batch_size) * tf.shape(q_t)[1]
        v_index1 = tf.tile(tf.reshape(v_index1, [batch_size, 1]), [1, nbins])

        v_index2 = act_t_ph * nbins 
        v_index2 = tf.tile(tf.reshape(v_index2, [batch_size, 1]), [1, nbins])

        v_index2 = v_index2 + tf.range(nbins)
        v_index = v_index1 + v_index2

        v_index = tf.reshape(v_index, [-1])

        v_dist_t_selected = tf.gather(tf.reshape(q_t, [-1]), v_index)
        v_dist_t_selected = tf.reshape(v_dist_t_selected, [batch_size, nbins])
        #  => v_dist_t_selected is p(x_t, a_t)

        # (1) Calculate Q(X_(t+1), a)
        old_qmin = -old_qmax
        delta_z = (old_qmax - old_qmin) / (nbins - 1)
        q_tp1_avg = q_value_avg(q_tp1, nbins, num_actions, old_qmin, old_qmax, delta_z)

        # (2) Get argmax_a Q(X_(t+1), a)
        q_tp1_best = tf.reduce_max(q_tp1_avg, 1)
        q_tp1_best_act = tf.argmax(q_tp1_avg, axis=1)
        q_tp1_best_act = tf.cast(q_tp1_best_act, tf.int32) # a* at t+1 step.

        # (3) Extract P(x_(t+1), a*)
        v_tp_index1 = tf.range(batch_size) * tf.shape(q_tp1)[1]
        v_tp_index1 = tf.tile(tf.reshape(v_tp_index1, [batch_size, 1]), [1, nbins])
        
        v_tp_index2 = q_tp1_best_act * nbins # (3, 5, 7) => (3* 51, 5* 51, 7* 51)
        v_tp_index2 = tf.tile(tf.reshape(v_tp_index2, [batch_size, 1]), [1, nbins])        

        # Check 1 : tf.range broadcasting
        v_tp_index2 = v_tp_index2 + tf.range(nbins)
        v_tp_index = v_tp_index1 + v_tp_index2
        v_tp_index = tf.reshape(v_tp_index, [-1])
        v_dist_tp1_selected = tf.gather(tf.reshape(q_tp1, [-1]), v_tp_index)
        v_dist_tp1_selected = tf.reshape(v_dist_tp1_selected, [batch_size, nbins]) # P(x_(t+1), a*)

        # (4) Make T_z, b_j, l, u in matrix form
        z = tf.tile(tf.reshape(tf.range(-old_qmax, old_qmax + delta_z, delta_z), [1, nbins]), [batch_size, 1])
        r = tf.tile(tf.reshape(rew_t_ph, [batch_size, 1]), [1, nbins])
        done = tf.tile(tf.reshape(done_mask_ph, [batch_size, 1]), [1, nbins])

        T_z = r + z * gamma * (1 - done)
        T_z = tf.maximum(tf.minimum(T_z, old_qmax), old_qmin) # Restrict upper and lower value of T_z to old_qmax and old_qmin
        b = (T_z - old_qmin) / delta_z
        l, u = tf.floor(b), tf.ceil(b)
        l_id = tf.cast(l, tf.int32)
        u_id = tf.cast(u, tf.int32)

        v_dist_t_selected = tf.reshape(v_dist_t_selected, [-1])
        add_index = tf.range(batch_size) * nbins

        err = tf.zeros([batch_size])

        for j in range(nbins):
            l_index = l_id[:, j] + add_index
            u_index = u_id[:, j] + add_index

            p_tl = tf.gather(v_dist_t_selected, l_index)
            p_tu = tf.gather(v_dist_t_selected, u_index)
            log_p_tl = tf.log(p_tl)
            log_p_tu = tf.log(p_tu)
            p_tp1 = v_dist_tp1_selected[:,j]
            err = err + p_tp1 * ((u[:,j] - b[:,j]) * log_p_tl + (b[:,j] - l[:,j]) * log_p_tu)

        err = tf.negative(err)
        weighted_error = tf.reduce_mean(err)

        if grad_norm_clipping is not None:
            gradients = optimizer.compute_gradients(weighted_error, var_list=q_t_vars)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
            optimize_expr = optimizer.apply_gradients(gradients)

            #optimize_expr = U.minimize_and_clip(optimizer,
            #                                    weighted_error,
            #                                    var_list=q_t_vars,
            #                                    clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=q_t_vars)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_t_vars, key=lambda v: v.name),
                                   sorted(q_tp1_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                importance_weights_ph
            ],
            outputs=weighted_error,
            updates=[optimize_expr]
        )

        update_target = U.function([], [], updates=[update_target_expr])

        q_values = U.function([obs_t_input], q_t)

        return act_f, train, update_target, {'q_dist_values': q_values}

