import gym

from baselines import deepq_dist
from baselines import logger


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    logger.configure("./log/xp5")
    env = gym.make("CartPole-v0")
    nbins = 51
    model = deepq_dist.models.mlp_to_dist(nbins,[64])
    act = deepq_dist.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback,
        old_qmin = -100,
        old_qmax = 100,
        nbins=nbins,
        new_qmin=0,
        new_qmax=10
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
