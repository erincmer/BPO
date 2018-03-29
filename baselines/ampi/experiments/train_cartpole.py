import gym

from baselines import ampi
from baselines.ampi import mlp_policy
from baselines import logger

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

LOG_DIR="/home/gpu_user/assia/ws/tf/BPO/baselines/ampi/log"
XP_NAME = "basic"

def main():

    # policy model
    #def policy_fn(name, ob_space, ac_space):
    #    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
    #        hid_size=64, num_hid_layers=2)

    env = gym.make("CartPole-v0")
    logger.configure(LOG_DIR + XP_NAME)

    # q model
    model = ampi.models.mlp([64])
    policy_fn = ampi.mlp_policy.mlpPolicy([64])
    
    act = ampi.learn(
        env,
        q_func=model,
        policy_fn=policy_fn,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to cartpole_model.pkl")
    act.save(LOG_DIR + "/cartpole_model.pkl")


if __name__ == '__main__':
    main()
