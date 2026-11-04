import numpy as np
import torch
import gym
import argparse
import d4rl
import utils
import APPO_Pro as APPO
import os


#args.policy_names
policy_names = ['APPO']

#设置GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        # print( eval_env.sim.data.qpos[:2],state[:2])两者是等价的
        # print("goal:",eval_env.target_goal)
        while not done:
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100  # 将return转化为score

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
    print("---------------------------------------")
    return d4rl_score


def main(args):
    # file_name定义非常重要
    file_name = f"{args.policyname}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policyname}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")
    if not os.path.exists("./AAAIresults"):
        os.makedirs("./AAAIresults")
    if not os.path.exists("./AAAIinfos"):
        os.makedirs("./AAAIinfos")
    if args.save_model and not os.path.exists("./AAAImodels"):
        os.makedirs("./AAAImodels")
    exist_files = os.listdir('./AAAIresults/')
    if file_name + '.npy' in exist_files:
        print(f'{file_name} exit, skipping')
        return

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if args.policyname == 'APPO':
        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "device": device,

            # TD3 + BC
            "tau": 0.005,
            "policy_noise": 0.2 * max_action,
            "noise_clip": 0.5 * max_action,
            "policy_freq": 2,
            "actor_lr": 3e-4,
            "value_lr": 3e-4,
            "critic_lr": args.critic_lr,
            "discount":  args.discount,

            # APPO
            "alpha":         args.alpha,
            "bc_weight"  :   args.bc_weight,
            "vae_weight" :   args.vae_weight,
            "delta_alpha":   args.delta_alpha,
            "Is_SubOptimal": args.Is_SubOptimal,
            "adv_threshold": 0
        }
        policy = APPO.TD3_BC(**kwargs)


    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)  # 创建对象
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))  # 获取数据集

    if args.normalize and 'antmaze' in args.env:
        replay_buffer.reward = np.where(replay_buffer.reward == 1.0, 0, -1.0)

    if args.normalize and not 'antmaze' in args.env:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1  # 这个不作变换
        print(
            "---------------------------------------No normalize---------------------------------------")  # antmaze会跳到这里来

    evaluations = []
    info_log = []
    for t in range(int(args.max_timesteps)):
        info = policy.train(replay_buffer, args.batch_size)
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            info_log.append(info)
            print(f"Time steps: {t + 1}", info)
            if 'antmaze' in args.env:
                evaluations.append(
                    eval_policy(policy, args.env, args.seed, mean, std, seed_offset=100, eval_episodes=100))
            else:
                evaluations.append(eval_policy(policy, args.env, args.seed, mean, std))
            np.save(f"./AAAIresults/{file_name}", evaluations)
            np.save(f"./AAAIinfos/{file_name}", info_log)
            if args.save_model: policy.save(f"./AAAImodels/{file_name}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policyname", type=str)  # Policy name
    parser.add_argument("--env", type=str)  # OpenAI gym environment name
    parser.add_argument("--seed", default=20, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--normalize", default=True)
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    args = parser.parse_args()
    main(args)