# Permutation Invariant Critic (PIC) #

The repository contains Pytorch implementation of MADDPG with Permutation Invariant Critic (PIC).

## Platform: 
Ubuntu 16.04 

## Dependencies: 
Python (3.7), Pytorch (4.1.0), openAI gym (https://github.com/openai/gym)

## Install the improved MPE:
    cd multiagent-particle-envs
    pip install -e .
Please ensure that `multiagent-particle-envs` has been added to your `PYTHONPATH`.

## Training 
    cd maddpg
	python main_vec.py --exp_name coop_navigation_n6 --scenario simple_spread_n6  --critic_type gcn_max  --cuda 
    

## Results


## Paper

If you used this code for your experiments or found it helpful, please consider citing the following paper:

<pre>
@inproceedings{pic,
 author    = {Iou-Jen Liu and Raymond A. Yeh and Alexander G. Schwing},
 title     = {PIC: Permutation Invariant Critic for Multi-Agent Deep Reinforcement Learning},
 year      = {2019},
 booktitle = {CoRL}
}
</pre>



