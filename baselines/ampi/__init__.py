from baselines.ampi import models  # noqa
from baselines.ampi.simple import learn, load  # noqa
from baselines.ampi.simple_cost import learn_cost  # noqa
from baselines.ampi.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=True)
