from baselines.deepq_dist import models  # noqa
from baselines.deepq_dist.build_graph import build_act, build_train, build_dist_act, build_dist_train  # noqa
from baselines.deepq_dist.simple import learn, load  # noqa
from baselines.deepq_dist.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=True)
