# Adaptive Posterior Learning: few-shot learning with a surprise-based memory module
Open source implementation of the Omniglot experiments in Adaptive Posterior Learning
 (ICLR 2019).

 This code has been reimplemented in PyTorch from the original TensorFlow
 implementation. Results may vary slightly from those reported in the paper.

 The authors thank Roman Lyapin for help with implementing parts of this codebase.

## How to run

Run ```train.py``` to train a classification model. Default hyperparameters
are sensible and will result in a good performance. We recommend first training
for 200 classes and then reusing that encoder for the desired final number of
classes.

```test.py``` demonstrates how to test the model either in the online setting,
or in the case of a fixed context size (as is done in most meta-learning papers).

If you use this code, please cite:

Tiago Ramalho, Marta Garnelo\
*Adaptive Posterior Learning: few-shot learning with a surprise-based memory module*\
In the proceedings of the International Conference on Learning Representations (ICLR), 2019.
