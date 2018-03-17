#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi

def train(env_id, num_timesteps, seed,is_Original=0,log_str="./logs/"):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure(log_str)
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    print(seed)
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=32, num_hid_layers=2)
    env = make_mujoco_env(env_id, workerseed)
    batch_num = 5000
    if is_Original==1:
        max_kl = 0.01

        is_Original = 1
    else:
        max_kl = 0.01


    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=batch_num, max_kl=max_kl, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3,is_Original =is_Original)
    env.close()

def main():
    args = mujoco_arg_parser().parse_args()
    #train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()

