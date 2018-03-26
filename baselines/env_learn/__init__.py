from baselines.env_learn import models  # noqa
from baselines.env_learn.build_graph import build_act, build_train  # noqa
from baselines.env_learn.simple import learn, load  # noqa
from baselines.env_learn.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=True)
