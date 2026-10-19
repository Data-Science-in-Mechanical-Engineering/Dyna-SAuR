from safe_rl import ppo_lagrangian
from safe_rl.utils.run_utils import setup_logger_kwargs
import gymnasium
from dynasaur.env.cartpole_continuous import CartPoleEnv
from dynasaur.env.walker_custom import Walker2dEnvCustom
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed')
#cart_pole or walker available
parser.add_argument('--env', type=str, default="cart_pole", help='Environment selection')
args = parser.parse_args()
if args.env == "walker":
	logger_kwargs = setup_logger_kwargs(
		exp_name='walker_ppo_lagrangian',
		seed=args.seed,
		data_dir='PPO_Lagrangian_Data',
		use_wandb=True,  
		wandb_project='safe-rl-walker'
	)

	ppo_lagrangian(
		env_fn = lambda : Walker2dEnvCustom(),
		ac_kwargs = dict(hidden_sizes=(128,128)),
		logger_kwargs = logger_kwargs,
		seed=args.seed
		)
elif args.env == "cart_pole":
	logger_kwargs = setup_logger_kwargs(
		exp_name='cartpole_ppo_lagrangian',
		seed=args.seed,
		data_dir='PPO_Lagrangian_Data',
		use_wandb=True,  
		wandb_project='safe-rl-cartpole'
	)

	ppo_lagrangian(
		env_fn = lambda : CartPoleEnv(),
		ac_kwargs = dict(hidden_sizes=(64,64)),
		logger_kwargs = logger_kwargs,
		seed=args.seed
		)
else:
	raise NotImplementedError("Only cart_pole and walker environments are available")
