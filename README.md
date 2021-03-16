# reproduceRL

This repository includes code for the following paper: 

1. Fuzzy Tiling Activations: A Simple Approach to Learning Sparse Representations Online, https://openreview.net/forum?id=zElset1Klrp

In our main paper, we use the setting fta_input_max = 20.0 and fta_eta = 2.0 because we use 20 tiles. As a suggestion for practical usage, one may tune fta_input_max a bit. We observe that on those simple discrete domains, setting it as 1.0 can be better (please keep fta_eta = 2*fta_input_max/n_tiles). 

If you do not want to use our repository, you can take out the ftann.py file under agents/network directory. An example is inside ftann.py file to show how to directly use our activation function. 

The implementation contains more powerful functionality than the one introduced in the paper. 

You are welcome to send me feedbacks, my email address is inside the paper. 

**Wang, Han implements a Pytorch version of FTA**. The code is available at https://github.com/hwang-ua/fta_pytorch_implementation
