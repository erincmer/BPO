from baselines.cbdqn.basic import models  # noqa
from baselines.cbdqn.basic.build_graph import build_act, build_train  # noqa
from baselines.cbdqn.basic.simple import learn, load  # noqa
from baselines.cbdqn.basic.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=True)
