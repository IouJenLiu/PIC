## PIC: Permutation Invariant Critic for Multi-Agent Deep Reinforcement Learning #
### CORL 2019
#### [[Project Website]](http://www.isle.illinois.edu/~yeh17/projects/invariant_critic/index.html) [[PDF]](https://128.84.21.199/abs/1911.00025)

Iou-Jen Liu&ast;, [Raymond A. Yeh](http://www.isle.illinois.edu/~yeh17/index.html)&ast;, [Alexander G. Schwing](http://www.alexander-schwing.de/)<br/>
University of Illinois at Urbana-Champaign<br/>
(* indicates equal contribution)

The repository contains Pytorch implementation of MADDPG with Permutation Invariant Critic (PIC).

If you used this code for your experiments or found it helpful, please consider citing the following paper:

<pre>
@inproceedings{LiuCORL2019,
  author = {I.-J. Liu$^\ast$ and R.~A. Yeh$^\ast$ and A.~G. Schwing},
  title = {PIC: Permutation Invariant Critic for Multi-Agent Deep Reinforcement Learning},
  booktitle = {Proc. CORL},
  year = {2019},
  note = {$^\ast$ equal contribution},
}
</pre>

#### Platform and Dependencies: 
* Ubuntu 16.04 
* Python (3.7)
* Pytorch (1.1.0)
* OpenAI gym (https://github.com/openai/gym)

### Install the improved MPE:
    cd multiagent-particle-envs
    pip install -e .
Please ensure that `multiagent-particle-envs` has been added to your `PYTHONPATH`.

### Training 
    cd maddpg
	python main_vec.py --exp_name coop_navigation_n6 --scenario simple_spread_n6  --critic_type gcn_max  --cuda 

### Acknowledgement
The MADDPG code is based on the DDPG implementation of https://github.com/ikostrikov/pytorch-ddpg-naf

The improved MPE code is based on the MPE implementation of https://github.com/openai/multiagent-particle-envs

The GCN code is based on the implementation of https://github.com/tkipf/gcn

### License
PIC is licensed under the MIT License

