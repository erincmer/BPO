from baselines import deepq2
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger
from baselines.common.atari_wrappers import make_atari

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')# changed to Beamrider since it gives larger rewards easy to see progress
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=0) # made it false code was complaining
    parser.add_argument('--dueling', type=int, default=0)# made it false for code simplicity
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    args = parser.parse_args()
    logger.configure("./log/xp2") # log results under BeamRider
    set_global_seeds(args.seed)
    
    # Environment definition
    env = make_atari(args.env)
    env = bench.Monitor(env, logger.get_dir())
    env = deepq2.wrap_atari_dqn(env)
    
    #
    print("Build model ...")
    model = deepq2.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        nbins=100,
        dueling=bool(args.dueling),
    )
    print("model OK")
    
    print("train ...")
    act = deepq2.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=args.num_timesteps,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized),
        old_qmin = -7,# min value of Q values
        old_qmax = 17,# max value of Q values
        new_qmin = 0,
        new_qmax = 10,
        nbins = 20 # number of bins
    )
    print("train OK ...")
    # act.save("pong_model.pkl") XXX
    env.close()


if __name__ == '__main__':
    main()
