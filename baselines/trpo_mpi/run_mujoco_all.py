from baselines.trpo_mpi import run_mujoco
from baselines import bench, logger
from multiprocessing import Process
import numpy as np
def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def mujoco_arg_parser(env):
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default=env)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(5e6))
    return parser


def single_exp(env2,j,i,args_env,seed):
    logger.configure("./log/" + env2 +"/"+ "Loss_" + str(j) + "_Run_" + str(i))
    log_str = "./log/" + env2 + "/" + "Loss_" + str(j) + "_Run_" + str(i)

    run_mujoco.train(args_env.env, num_timesteps=args_env.num_timesteps, seed=seed,is_Original = j,log_str = log_str)
def main():
    env2 = "HalfCheetah-v2"



    proc = []
    trial = 10
    loss = 1
    for j in range(trial):


        seed = np.random.randint(10000)
        args_env = mujoco_arg_parser(env2).parse_args()
        p = Process(target=single_exp, args=(env2,loss,j,args_env,seed))
        proc.append(p)

    for p in proc:
        p.start()

            # Join after all processes started, so you actually get parallelism
    for p in proc:
        p.join()





if __name__ == '__main__':
    main()