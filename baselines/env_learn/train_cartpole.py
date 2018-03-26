import gym

from baselines import env_learn


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    env = gym.make("CartPole-v0")
    model = env_learn.models.mlp([64])
    model2 = env_learn.models.mlp_env([64])
    act = env_learn.learn(
        env,
        q_func=model,
        env_func=model2,
        lr=1e-3,
        max_timesteps=50000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
