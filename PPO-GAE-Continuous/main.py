import pandas as pd
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from agents import PPO_continuous
from datetime import datetime
import os


def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _ = env.step(action)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


def main(args, env_name, number, seed):
    rate = np.zeros([100000, 11])
    for b in range(10):
        env = gym.make(env_name)
        env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
        # Set random seed
        env.seed(seed)
        env.action_space.seed(seed)
        env_evaluate.seed(seed)
        env_evaluate.action_space.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        args.state_dim = env.observation_space.shape[0]
        args.action_dim = env.action_space.shape[0]
        args.max_action = float(env.action_space.high[0])
        args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
        print("env={}".format(env_name))
        print("state_dim={}".format(args.state_dim))
        print("action_dim={}".format(args.action_dim))
        print("max_action={}".format(args.max_action))
        print("max_episode_steps={}".format(args.max_episode_steps))

        evaluate_num = 0  # Record the number of evaluations
        evaluate_rewards = []  # Record the rewards during the evaluating
        total_steps = 0  # Record the total steps during the training

        replay_buffer = ReplayBuffer(args)
        agent = PPO_continuous(args)

        # Build a tensorboard
        now = datetime.now()
        dt_string = now.strftime("%m-%d-%Y_%H-%M-%S")
        directory = "/Users/winslowfan/Documents/Lectures/Sheffield/Final Dissertation/ReinforcementLearning/DRL-code-pytorch/5.PPO-continuous/pretrained"

        if not os.path.exists(directory):
            os.makedirs(directory)

        # model_saving_dir = '/GAE1_preTrained/'
        # if not os.path.exists(model_saving_dir):
        #     os.makedirs(model_saving_dir)
        # model_saving_dir = model_saving_dir + '/' + env_name[0] + '/' + dt_string + '/'
        # if not os.path.exists(model_saving_dir):
        #     os.makedirs(model_saving_dir)
        directory = directory + '/' + env_name + '/' + dt_string + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_{}_number_{}_seed_{}_{}'.format(env_name, args.policy_dist, number, seed, dt_string))

        state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
        if args.use_reward_norm:  # Trick 3:reward normalization
            reward_norm = Normalization(shape=1)
        elif args.use_reward_scaling:  # Trick 4:reward scaling
            reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
        episode = 0
        suc = np.zeros([100, 1])
        while total_steps < args.max_train_steps:
            if episode % 100 == 0:
                rate[int(episode / 100)][0] = episode
                rate[int(episode / 100)][b+1] = suc.sum() / 100
                print(rate[int(episode / 100)][0], rate[int(episode / 100)][b+1])
                suc = np.zeros([100, 1])
            s = env.reset()
            if args.use_state_norm:
                s = state_norm(s)
            if args.use_reward_scaling:
                reward_scaling.reset()
            episode_steps = 0
            done = False
            episode += 1
            ss = []
            rs = []
            a_s = []
            a_logs = []
            s_s = []
            ds = []
            dws = []
            while not done:
                episode_steps += 1
                ss.append(s)
                a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
                if args.policy_dist == "Beta":
                    action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
                else:
                    action = a
                a_s.append(action)
                a_logs.append(a_logprob)

                s_, r, done, _ = env.step(action)
                s_s.append(s_)
                ds.append(done)
                state = s_

                if args.use_state_norm:
                    s_ = state_norm(s_)
                if args.use_reward_norm:
                    r = reward_norm(r)
                elif args.use_reward_scaling:
                    r = reward_scaling(r)

                # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                # dw means dead or win,there is no next state s';
                # but when reaching the max_episode_steps,there is a next state s' actually.
                if done and abs(state[1]) <= 5 and state[2] >= 160:
                    dw = True
                else:
                    dw = False
                if done and abs(state[2]) <= 10 and state[3] >= 160:
                    suc[int(episode % 100)] = 1
                dws.append(dw)

                # print(suc[int(episode % 100)])
                if done:
                    print(
                        'Episode: {}; Time step: {}, Reward: {}, Final state: {}, suc: {}, dw: {}, rate: {}'.format(
                            episode, total_steps, r, state, suc[int(episode % 100)], dw, suc.sum()/100))
                    # print(suc)



                # Take the 'action'，but store the original 'a'（especially for Beta）
                replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
                s = s_
                total_steps += 1

                # When the number of transitions in buffer reaches batch_size,then update
                if replay_buffer.count == args.batch_size:
                    agent.update(replay_buffer, total_steps)
                    replay_buffer.count = 0

                # Evaluate the policy every 'evaluate_freq' steps
                if total_steps % args.save_model_freq == 0:
                    print(
                        "--------------------------------------------------------------------------------------------")
                    print("saving model at : " + directory)

                    actor_path = directory + "PPO_actor_GAE_{}_{}.pth".format(datetime.now().strftime("%m-%d-%Y_%H-%M-%S"), episode)
                    critic_path = directory + "PPO_critic_GAE_{}_{}.pth".format(datetime.now().strftime("%m-%d-%Y_%H-%M-%S"), episode)
                    # actor_target_path = model_saving_dir + 'TD3_actor_target_{}.pth'.format(total_steps)
                    # critic_target_path = model_saving_dir + 'SAC_critic_target_{}.pth'.format(i_episode)
                    torch.save(agent.actor.state_dict(), actor_path)
                    torch.save(agent.critic.state_dict(), critic_path)
                    # torch.save(agent.actor_target.state_dict(), actor_target_path)
                    # torch.save(agent.critic_target.state_dict(), critic_target_path)
                    print("model saved")
                    print(
                        "--------------------------------------------------------------------------------------------")
        print(datetime.now() - now)
        rate_df = pd.DataFrame(rate)
        rate_df.to_csv('/Users/winslowfan/Documents/Lectures/Sheffield/Final Dissertation/ReinforcementLearning/PPO_test/GAE_test_{}_100_3000_successful_rate.csv'.format(dt_string))
                    # evaluate_num += 1
                    # evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                    # evaluate_rewards.append(evaluate_reward)
                    # # print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                    # writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
                    # # Save the rewards
                    # if evaluate_num % args.save_freq == 0:
                    #     np.save('./data_train/PPO_continuous_{}_env_{}_number_{}_seed_{}.npy'.format(args.policy_dist, env_name, number, seed), np.array(evaluate_rewards))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--save_model_freq", type=float, default=2e5, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Beta", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-3, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.98, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=20, help="PPO parameter")
    parser.add_argument("--action_std_init", type=float, default=0.6, help="Initial action std")
    parser.add_argument("--action_std_decay_rate", type=float, default=5e-2, help="Linearly decay rate of action std")
    parser.add_argument("--min_action_std", type=float, default=1e-2, help="Minimum value of action std")
    parser.add_argument("--action_std_decay_freq", type=int, default=2e5, help="Frequency of action std decay")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=False, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=False, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=False, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--use_action_std", type=bool, default=False, help="Trick 11: use action standard error")

    args = parser.parse_args()

    env_name = ["RobotInterception-v4"]
    env_index = 0
    main(args, env_name=env_name[env_index], number=1, seed=10)
