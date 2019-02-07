# Finetuning Pretrained Network

The code imagenet_finetune.py does the following:
* Constructs a deep network. This network starts with ResNet 50 up to its average pooling
layer. Then, a small network with 32 hidden nodes then 10 output nodes (dense connections)
is added on top.
* Initializes the weights of the ResNet 50 portion with the parameters from training on ImageNet.
* Performs training on only the new layers using CIFAR10 dataset – all other weights are fixed
to their values learned on ImageNet.

The code modelloadprint.py does the following:
* Tests the trained model on CIFAR 10 dataset
* It then generates HTML to print the test results
