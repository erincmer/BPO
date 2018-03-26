import gym

from baselines import deepq2
from baselines import logger


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    logger.configure("./log/xp2")
    env = gym.make("CartPole-v0")
    nbins = 10
    model = deepq2.models.mlp(nbins,[64])
    act = deepq2.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback,
        qmin = -20,
        qmax = 20,
        nbins=nbins
        )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
