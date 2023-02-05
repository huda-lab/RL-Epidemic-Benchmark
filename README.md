# RL-Epidemic-Benchmark
PyTorch implementation of the paper:

- **Title**: *Planning Multiple Epidemic Interventions with Reinforcement Learning* 
- **ArXiv**: [coming soon] 
<!-- - **Authors**:  -->
<!-- - **Conference**:  -->
<!-- - More details:  -->

### Abstract 

Combating an epidemic entails finding a plan that describes when and how to apply different interventions, such as mask-wearing mandates, vaccinations, school or workplace closures. An optimal plan will curb an epidemic with minimal loss of life, disease burden, and economic cost. Finding an optimal plan is an intractable computational problem in realistic settings. Policy-makers, however, would greatly benefit from tools that can efficiently search for plans that minimize disease and economic costs especially when considering multiple possible interventions over a continuous and complex action space given a continuous and equally complex state space. We formulate this problem as a Markov decision process. Our formulation is unique in its ability to represent multiple continuous interventions over any disease model defined by ordinary differential equations. We illustrate how to effectively apply state-of-the-art actor-critic reinforcement learning algorithms (PPO and SAC) to search for plans that minimize overall costs. We empirically evaluate the learning performance of these algorithms and compare their performance to hand-crafted baselines that mimic plans constructed by policy-makers. Our method outperforms baselines. Our work confirms the viability of a computational approach to support policy-makers. 

![The agent-environment interaction in the Markov decision process formalizing the epidemic planning problem.](EpiPolicyRL.png)

### Selected Requirements 

- EpiPolicy Epidemic Simulator 
- Python 3.8/3.9 
- Pytorch
- stable-baselines3
- gym
- Numba


Note: For installing **all** requirements use --> `pip3 install -r requirements.txt`

### Epipolicy 

EpiPolicy is a population-based epidemic simulator and policy aid that allows users to customize its compartmental model to capture different epidemic scenarios for epidemics like COVID-19. Its user interface makes it easy to define various interventions such as social distancing, school closure, vaccination, and disease-specific interventions. EpiPolicy aids policy makers by simulating and constructing locale-specific intervention schedules that reduce disease burden, and minimize social and economic costs. 

More information on EpiPolicy: https://epipolicy.github.io/ 

### How To Run 

``` 
# clone project
git clone https://github.com/Nikunj-Gupta/Planning-Multiple-Epidemic-Interventions-with-Reinforcement-Learning.git

# Install requirements (preferably in a virtual environment)
pip3 install -r requirements.txt

# Run PPO/SAC on multiple scenarios in EpiPolicy 

python3 <RL_algorithm>.py --gym-id <scenario_name> 

# Example: run PPO on SIR_A 
python3 ppo_kernel.py --gym-id SIR_A

# Example: run SAC on SIRV_B 
python3 sac_kernel.py --gym-id SIRV_B 

# To reproduce the results in the paper --> Refer/Use plots.ipynb 
```

Note: We used the following seeds: 0, 1, 2, 3 

##### Other Options for PPO 

```
python3 ppo_kernel.py --gym-id <SCENARIO> 
[--exp-name] [--learning-rate] [--seed]
[--total-timesteps] [--torch-deterministic] 
[--cuda][--track] [--wandb-project-name] 
[--wandb-entity] [--capture-video] 
[--policy_plot_interval] [--num-envs] 
[--num-steps] [--anneal-lr] [--gae]
[--gamma][--gae-lambda][--num-minibatches]
[--update-epochs][--norm-adv][--clip-coef]
[--clip-vloss][--ent-coef][--vf-coef]
[--max-grad-norm][--target-kl] 
```
##### Other Options for SAC 

```
python3 sac_kernel.py --gym-id <SCENARIO> 
[--exp-name] [--learning-starts] [--seed]
[--total-timesteps] [--target-entropy-scale] 
[--train-freq][--gradient-steps] 
```
##### To reproduce the baselines 
Code for baselines is in `plots.ipynb`

### Citations 

Planning Multiple Epidemic Interventions with Reinforcement Learning
```
Coming soon... 
```

EpiPolicy: a tool for combating epidemics

```
@article{mai2022epipolicy,
  title={EpiPolicy: a tool for combating epidemics},
  author={Mai, Anh Le Xuan and Mannino, Miro and Tariq, Zain and Abouzied, Azza and Shasha, Dennis},
  journal={XRDS: Crossroads, The ACM Magazine for Students},
  volume={28},
  number={2},
  pages={24--29},
  year={2022},
  publisher={ACM New York, NY, USA}
}
```


Planning Epidemic Interventions with EpiPolicy 

```
@inproceedings{tariq2021planning,
  title={Planning Epidemic Interventions with EpiPolicy},
  author={Tariq, Zain and Mannino, Miro and Le Xuan Anh, Mai and Bagge, Whitney and Abouzied, Azza and Shasha, Dennis},
  booktitle={The 34th Annual ACM Symposium on User Interface Software and Technology},
  pages={894--909},
  year={2021}
}
```

### Acknowledgements 
We acknowledge [[https://github.com/vwxyzjn/cleanrl]](https://github.com/vwxyzjn/cleanrl) for a Clean Implementation of RL Algorithms 
