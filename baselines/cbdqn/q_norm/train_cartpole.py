import gym

from baselines.cbdqn import q_norm
from baselines import logger


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    logger.configure("./log/xp2")
    env = gym.make("CartPole-v0")
    nbins = 10
    model = q_norm.models.mlp(nbins,[64])
    act = q_norm.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback,
        old_qmin = -40,
        old_qmax = 200,
        nbins=nbins,
        new_qmin=0,
        new_qmax=10
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
