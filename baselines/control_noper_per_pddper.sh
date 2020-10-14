python run.py --alg=deepq --env=CartPole-v1 --num_timesteps=1e6 --save_path=~/models/cartpole-v1-20201014-noper --log_path=~/logs/cartpole-v1-20201014-noper --gamma=0.99 --dueling=False --prioritized_replay=False
python run.py --alg=deepq --env=Acrobot-v1 --num_timesteps=1e6 --save_path=~/models/acrobot-v1-20201014-noper --log_path=~/logs/acrobot-v1-20201014-noper --gamma=0.99 --dueling=False --prioritized_replay=False
python run.py --alg=deepq --env=CartPole-v1 --num_timesteps=1e6 --save_path=~/models/cartpole-v1-20201014-per --log_path=~/logs/cartpole-v1-20201014-per --gamma=0.99 --dueling=False --prioritized_replay=True
python run.py --alg=deepq --env=Acrobot-v1 --num_timesteps=1e6 --save_path=~/models/acrobot-v1-20201014-per --log_path=~/logs/acrobot-v1-20201014-per --gamma=0.99 --dueling=False --prioritized_replay=True
python run.py --alg=deepq --env=CartPole-v1 --num_timesteps=1e6 --save_path=~/models/cartpole-v1-20201014-pddper --log_path=~/logs/cartpole-v1-20201014-pddper --gamma=0.99 --dueling=True --prioritized_replay=True
python run.py --alg=deepq --env=Acrobot-v1 --num_timesteps=1e6 --save_path=~/models/acrobot-v1-20201014-pddper --log_path=~/logs/acrobot-v1-20201014-pddper --gamma=0.99 --dueling=True --prioritized_replay=True
