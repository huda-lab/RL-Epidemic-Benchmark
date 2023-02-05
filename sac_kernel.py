PERIOD = 7

# Env
import gym, json, time
import argparse
from gym import spaces
from epipolicy.core.epidemic import construct_epidemic
from epipolicy.obj.act import construct_act
import numpy as np
import math 

class EpiEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, session, vac_starts):
        super(EpiEnv, self).__init__()
        self.epi = construct_epidemic(session)
        total_population = np.sum(self.epi.static.default_state.obs.current_comp)
        obs_count = self.epi.static.compartment_count * self.epi.static.locale_count * self.epi.static.group_count
        action_count = 0
        action_param_count =  0
        for itv in self.epi.static.interventions:
            if not itv.is_cost:
                action_count += 1
                action_param_count += len(itv.cp_list)
        self.act_domain = np.zeros((action_param_count, 2), dtype=np.float32)
        index = 0
        for itv in self.epi.static.interventions:
            if not itv.is_cost:
                for cp in itv.cp_list:
                    self.act_domain[index, 0] = cp.min_value
                    self.act_domain[index, 1] = cp.max_value
                    index += 1
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=0, high=1, shape=(action_count,), dtype=np.float32)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=total_population, shape=(obs_count,), dtype=np.float32)

        self.time_passed = 0 # To keep track of how many days have passed 
        self.vac_starts = vac_starts # number of days to prepare a vaccination / make it available 

    def step(self, action):
        if self.time_passed < self.vac_starts: 
            action[0] = 0 
        # print("================================================================")
        # print("time elapsed: ", self.time_passed) 
        # print("action: ", action) 
        # print("================================================================")

        self.time_passed += PERIOD 


        expanded_action = np.zeros(len(self.act_domain), dtype=np.float32)
        index = 0
        for i in range(len(self.act_domain)):
            if self.act_domain[i, 0] == self.act_domain[i, 1]:
                expanded_action[i] = self.act_domain[i, 0]
            else:
                expanded_action[i] = action[index]
                index += 1

        epi_action = []
        index = 0
        for itv_id, itv in enumerate(self.epi.static.interventions):
            if not itv.is_cost:
                epi_action.append(construct_act(itv_id, expanded_action[index:index+len(itv.cp_list)]))
                index += len(itv.cp_list)

        total_r = 0
        for i in range(PERIOD):
            state, r, done = self.epi.step(epi_action)
            total_r += r
            if done:
                self.time_passed = 0 
                break
        return state.obs.current_comp.flatten(), total_r, done, dict()

    def reset(self):
        state = self.epi.reset()
        return state.obs.current_comp.flatten()  # reward, done, info can't be included
    def render(self, mode='human'):
        pass
    def close(self):
        pass
    
class RewardScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale
    
    def reward(self, rew):
        # modify rew
        return rew * self.scale
    
######
epi_ids = ["SIR_A", "SIR_B", "SIRV_A", "SIRV_B", 
            "COVID_A", "COVID_B", "COVID_C"] 

def make_env(gym_id, seed, idx, vac_starts):
    def thunk():
        if 'jsons'in gym_id: 
            if gym_id.split('/')[-1] in epi_ids:
                fp = open('{}.json'.format(gym_id), 'r')
                session = json.load(fp)
                env = EpiEnv(session, vac_starts=vac_starts)
        else:
            env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env = RewardScale(env, 100)
        # Our env is deterministic
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def make_primal_env(gym_id, vac_starts):
    def thunk():
        if 'jsons'in gym_id: 
            if gym_id.split('/')[-1] in epi_ids:
                fp = open('{}.json'.format(gym_id), 'r')
                session = json.load(fp)
                env = EpiEnv(session, vac_starts=vac_starts)
        else:
            env = gym.make(gym_id)
        return env
    return thunk
#####
def parse_args(main_args = None):
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="SAC",
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="SIR_A",
        help="the id of the gym environment")
#     parser.add_argument("--learning-rate", type=float, default=3e-4,
#         help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=210000,
        help="total timesteps of the experiments")
    parser.add_argument("--vac-starts", type=int, default=0,
        help="vac_starts")
#     parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
#         help="if toggled, `torch.backends.cudnn.deterministic=False`")
#     parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
#         help="if toggled, cuda will be enabled by default")
#     parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
#         help="if toggled, this experiment will be tracked with Weights and Biases")
#     parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
#         help="the wandb's project name")
#     parser.add_argument("--wandb-entity", type=str, default=None,
#         help="the entity (team) of wandb's project")
#     parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
#         help="weather to capture videos of the agent performances (check out `videos` folder)")

#     parser.add_argument("--policy_plot_interval", type=int, default=1,
#         help="seed of the experiment")

    # Algorithm specific arguments
    parser.add_argument("--learning-starts", type=int, default=1000,
        help="learning starts")
    parser.add_argument("--target-entropy-scale", type=int, default=1,
        help="scale of target entropy with the dimension of action")
    parser.add_argument("--train-freq", type=int, default=5,
        help="train freq")
    parser.add_argument("--gradient-steps", type=int, default=1,
        help="gradient steps")
#     parser.add_argument("--num-envs", type=int, default=1,
#         help="the number of parallel game environments")
#     parser.add_argument("--num-steps", type=int, default=2048,
#         help="the number of steps to run in each environment per policy rollout")
#     parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
#         help="Toggle learning rate annealing for policy and value networks")
#     parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
#         help="Use GAE for advantage computation")
#     parser.add_argument("--gamma", type=float, default=0.99,
#         help="the discount factor gamma")
#     parser.add_argument("--gae-lambda", type=float, default=0.95,
#         help="the lambda for the general advantage estimation")
#     parser.add_argument("--num-minibatches", type=int, default=32,
#         help="the number of mini-batches")
#     parser.add_argument("--update-epochs", type=int, default=10,
#         help="the K epochs to update the policy")
#     parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
#         help="Toggles advantages normalization")
#     parser.add_argument("--clip-coef", type=float, default=0.2,
#         help="the surrogate clipping coefficient")
#     parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
#         help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
#     parser.add_argument("--ent-coef", type=float, default=0.0,
#         help="coefficient of the entropy")
#     parser.add_argument("--vf-coef", type=float, default=0.5,
#         help="coefficient of the value function")
#     parser.add_argument("--max-grad-norm", type=float, default=0.5,
#         help="the maximum norm for the gradient clipping")
#     parser.add_argument("--target-kl", type=float, default=None,
#         help="the target KL divergence threshold")
    if main_args is not None:
        args = parser.parse_args(main_args.split())
    else:
        args = parser.parse_args()
#     args.num_steps //= PERIOD
    args.total_timesteps //= PERIOD
#     args.batch_size = int(args.num_envs * args.num_steps)
#     args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args
#####
if __name__ == "__main__":
    args = parse_args()
    seeds = [0,1,2,3]
    for seed in seeds:
        args.seed = seed
        run_name = f"{args.gym_id.split('/')[-1]}__{args.exp_name}_scale__{args.seed}__{int(time.time())}"
        env = make_env(args.gym_id, args.seed, 0, vac_starts=args.vac_starts)()
        test_env = make_primal_env(args.gym_id, vac_starts=args.vac_starts)()

        import torch
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
        from stable_baselines3.common.logger import TensorBoardOutputFormat

        class SummaryWriterCallback(BaseCallback):
            def __init__(self, verbose: int = 0):
                super().__init__()
                self.best_total_r = -math.inf 
            def _on_training_start(self):
                self._log_freq = 292  # log every 1000 calls

                output_formats = self.logger.output_formats
                # Save reference to tensorboard formatter object
                # note: the failure case (not formatter found) is not handled here, should be done with try/except.
                self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

            def _on_step(self) -> bool:
                # PLOT POLICY
                if self.n_calls % self._log_freq == 0:
                    env_obs = torch.Tensor(self.training_env.reset())
                    test_obs = torch.Tensor(test_env.reset())
                    done = False
                    timestep = 0
                    total_r = 0
                    itv_line = []
                    while not done:
                        with torch.no_grad():
                            action_mean, _ = self.model.predict(env_obs, deterministic=True)
                            action_mean = np.array(action_mean)
                            test_action_mean = np.clip(np.mean(action_mean, axis=0), 0, 1)

                        test_obs, r, done, _ = test_env.step(test_action_mean)
                        test_obs = torch.Tensor(test_obs)
                        itv_index = 0
                        itv_array = []
                        for itv in test_env.epi.static.interventions:
                            if not itv.is_cost:
                                v = float(test_action_mean[itv_index])
                                self.tb_formatter.writer.add_scalar('charts/policy_{}/{}'.format(self.num_timesteps, itv.name), v, timestep)
                                itv_array.append(v)
                                itv_index += 1
                        itv_line.append(itv_array)

                        env_obs, _, _, _ = self.training_env.step(action_mean)
                        env_obs = torch.Tensor(env_obs)

                        total_r += r
                        timestep += PERIOD
                    line = '|'.join([str(self.num_timesteps), str(total_r), str(itv_line)]) + '\n'
                    self.tb_formatter.writer.add_scalar("charts/learning_curve", total_r, self.num_timesteps)
                    print("At global step {}, total_rewards={}, best_total_rewards={}".format(self.num_timesteps, total_r, self.best_total_r))
                    self.tb_formatter.writer.flush()
                    csv_file = open('runs/{}_1/records.csv'.format(run_name), 'a')
                    csv_file.write(line)
                    csv_file.close()
                    line2 = '|'.join([str(self.num_timesteps), str(env.obs_rms.mean), str(env.obs_rms.var), str(env.obs_rms.count)]) + '\n' 
                    csv_file2 = open('runs/{}_1/obs_normalization.csv'.format(run_name), 'a')
                    csv_file2.write(line2)
                    csv_file2.close()

                    """
                    Saving Model Checkpoints 
                    """
                    if total_r > self.best_total_r: 
                        self.best_total_r = total_r 
                        print("Saving Best Checkpoint:") 
                        self.model.save('runs/{}_1/model_checkpoints/'.format(run_name) )

        # learning_starts = [1000, 5000, 10000]
        # target_entropies = [-env.action_space.shape[0], -2*env.action_space.shape[0], -4*env.action_space.shape[0]]
        # choice = 0
        # learning_start = learning_starts[choice // 3]
        # target_entropy = target_entropies[choice % 3]

        print(f"Running with {run_name}")
        model = SAC("MlpPolicy", env, verbose=0, tensorboard_log="runs/", learning_starts=args.learning_starts, target_entropy=-args.target_entropy_scale * env.action_space.shape[0],
                   train_freq=args.train_freq, gradient_steps=args.gradient_steps)
        model.learn(total_timesteps=args.total_timesteps, log_interval=4, callback=SummaryWriterCallback(), tb_log_name=run_name)