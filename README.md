# reproduceRL

You are welcome to send me feedbacks: pan6 AT ualberta DOT ca

## How to run the code

**Dependency**: The repository should work at least for Tensorflow version 1.14.0/1.13.0, python version 3.6/3.7, gym 0.14.0/0.17.1

**Example to run**: python main.py jsonfiles/fta.json 0
This will run the first configuration in the ``fta.json`` file. Experiment configurations are in the jsonfile files under jsonfiles directory. Please see main.py for a detailed explanation regarding the meaning of the index after jsonfiles/fta.json

## This repository includes code for the following papers

### Fuzzy Tiling Activations: A Simple Approach to Learning Sparse Representations Online, https://openreview.net/forum?id=zElset1Klrp

**Bibtex**:
```
@article{
pan2021fuzzy,
title={Fuzzy Tiling Activations: A Simple Approach to Learning Sparse Representations Online},
author={Yangchen Pan and Kirby Banman and Martha White},
journal={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=zElset1Klrp}
}
```

In our main paper, we use the setting fta_input_max = 20.0 and fta_eta = 2.0 because we use 20 tiles. As a suggestion for practical usage, one may tune fta_input_max a bit. We observe that on those simple discrete domains, setting it as 1.0 can be better (please keep fta_eta = 2*fta_input_max/n_tiles). 

If you do not want to use our repository, you can take out the ftann.py file under agents/network directory. An example is inside ftann.py file to show how to directly use our activation function. 

The implementation of FTA contains more functionalities than those introduced in the paper. Specifically, it allows to use both 1) multiple tilings (each input scalar uses the same set of multiple tiling vectors); 2) individual tilings (each input scalar uses its own tiling vector); 3) different activations before and after applying FTA function. Please see the comments inside ftann.py for explanations. 

**Wang, Han implements a Pytorch version of FTA**. The code is available at https://github.com/hwang-ua/fta_pytorch_implementation

### The category of papers about sampling experiences in model-based RL (specifically, search-control in Dyna)

**Bibtex**:
```
@article{
pan2019hcdyna,
Author = {Pan, Yangchen and Yao, Hengshuai and Farahmand, Amir-massoud and White, Martha},
journal = {International Joint Conference on Artificial Intelligence},
Title = {Hill Climbing on Value Estimates for Search-control in Dyna},
Year = {2019}
}
@article{
pan2020frequencybased,
title={Frequency-based Search-control in Dyna},
author={Yangchen Pan and Jincheng Mei and Amir-massoud Farahmand},  
journal={International Conference on Learning Representations},
year={2020}
}
@article{
pan2022per,
title={Understanding and Mitigating the Limitations of Prioritized Experience Replay},
author={Yangchen Pan and Jincheng Mei and Amir-massoud Farahmand and Martha White and Hengshuai Yao and Mohsen Rohani and Jun Luo},
journal={Conference on Uncertainty in Artificial Intelligence},
year={2022}
}
```
Use jsonfiles/sampledist.json. It may not be easy to extract specific functions to use in your own repository. I suggest to implement your own version; it should be easy to do. The projection operation and dimension-dependent stepsize may not be needed, so you can start from the simplest version of the hill climbing. 
