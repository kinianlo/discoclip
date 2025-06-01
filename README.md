# DisCoClip

This repository contains the code for the paper "DisCoClip" which focuses on replacing the text encoder in CLIP with tensor-network-based models.

# Datasets

We train and test our model on SVO-Probes, ARO and ComVG.

# General pipeline for CCG models

A sentence is first parsed into diagrams using a cached parser. Then the diagrams are converted into tensor networks using the `MPSAnsatz` with a chosen bond dimension. 
Before we do any training, we need to build up a large set of symbols, which will each be assigned an integer ID. 
We do this because we want to follow the same Pytorch convention, which there is a preprocessor that converts a sentence as a string into a list of integers. The only difference is that for each network, we also have to specify the einsum string that describes how the tensors are contracted.

# General pipeline for non-CCG models

For non-CCG models, we will have another preprocessor that converts a sentence into a list of integers, but there is no need to specify the einsum string, as the tensors are contracted in a fixed way.

# Models

We have implemented the following models:
- `Bobcat`: A CCG model that uses tensors of different ranks for different types of words.
- `Tree`: A CCG model that represents each word as a vector and uses rank-3 tensors to compose them. 
- `MPS`: A non-CCG model that uses a matrix for each word and neighbouring words are contracted in a fixed way.
- `Add` and `Mul`: Non-CCG models that use element-wise addition and multiplication, respectively, to compose the words.