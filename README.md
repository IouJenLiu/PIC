# Permutation Invariant Critic (PIC) #

The repository contains Pytorch implementation of MADDPG with Permutation Invariant Critic (PIC).

##Platform: 
Ubuntu 16.04 

##Known Dependencies: 
Python (3.7), Pytorch (4.1.0), multiagent-particle-envs, openAI gym (https://github.com/openai/gym)

## Install the improved MPE:
To install, `cd` into the root directory and type `pip install -e .`
Please ensure that `multiagent-particle-envs` has been added to your `PYTHONPATH`.

## Training 
    cd maddpg
	python main_vec.py --exp_name coop_navigation_n6 --scenario simple_spread_n15_vs5  --critic_type gcn_max  --cuda 
    




