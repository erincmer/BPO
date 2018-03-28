from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype

def _mlpPolicy(hiddens, ob, ob_space, ac_space, scope, gaussian_fixed_var=True, reuse=False):
    assert isinstance(ob_space, gym.spaces.Box)

    with tf.variable_scope(scope, reuse=reuse):
        pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        with tf.variable_scope("obfilter"):
            ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - ob_rms.mean) / ob_rms.std, -5.0, 5.0)
            last_out = obz
            #for i in range(num_hid_layers):
            for (i,hidden) in zip(range(len(hiddens)),hiddens):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hidden, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        with tf.variable_scope('pol'):
            last_out = obz
            #for i in range(num_hid_layers):
            #for hidden in hiddens:
            for (i,hidden) in zip(range(len(hiddens)),hiddens):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hidden, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final', kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final', kernel_initializer=U.normc_initializer(0.01))

        pd = pdtype.pdfromflat(pdparam)

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, pd.sample(), pd.mode())
        _act = U.function([stochastic, ob], [ac, vpred])

        return pd.logits, _act

def mlpPolicy(hiddens=[]):
    return lambda *args, **kwargs: _mlpPolicy(hiddens, *args, **kwargs)


