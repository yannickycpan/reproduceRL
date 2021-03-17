# reproduceRL

## How to run the code

**Dependency**: The repository should work at least for Tensorflow version 1.14.0/1.13.0, python version 3.6/3.7, gym 0.14.0/0.17.1

**Example to run**: python main.py jsonfiles/fta.json 0
This will run the first configuration in the ``fta.json`` file. Experiment configurations are in the jsonfile files under jsonfiles directory. Please see main.py for a detailed explanation regarding the meaning of the index after jsonfiles/fta.json

## This repository includes code for the following paper

### Fuzzy Tiling Activations: A Simple Approach to Learning Sparse Representations Online, https://openreview.net/forum?id=zElset1Klrp

**Bibtex**:
```
@inproceedings{
pan2021fuzzy,
title={Fuzzy Tiling Activations: A Simple Approach to Learning Sparse Representations Online},
author={Yangchen Pan and Kirby Banman and Martha White},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=zElset1Klrp}
}
```

In our main paper, we use the setting fta_input_max = 20.0 and fta_eta = 2.0 because we use 20 tiles. As a suggestion for practical usage, one may tune fta_input_max a bit. We observe that on those simple discrete domains, setting it as 1.0 can be better (please keep fta_eta = 2*fta_input_max/n_tiles). 

If you do not want to use our repository, you can take out the ftann.py file under agents/network directory. An example is inside ftann.py file to show how to directly use our activation function. 

The implementation of FTA contains more powerful functionality than the one introduced in the paper. Specifically, using multiple tilings are also included in the paper. Please see the comments inside ftann.py for explanations. 

You are welcome to send me feedbacks, my email address is inside the paper. 

**Wang, Han implements a Pytorch version of FTA**. The code is available at https://github.com/hwang-ua/fta_pytorch_implementation


