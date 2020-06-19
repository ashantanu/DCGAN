# DCGAN Implementation using PyTorch
DCGAN implementation for learning.

Used below resources:
* [Dataloader tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
* [Transforms](https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transforms)
* [ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html?highlight=imagefolder#torchvision.datasets.ImageFolder)
* [DCGAN Paper](https://arxiv.org/pdf/1511.06434.pdf)
* [GAN Paper](https://arxiv.org/abs/1406.2661)
* [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
* [PyTorch Layers](https://pytorch.org/docs/stable/nn.html)
* Convolutions: [Guide to Convolutions](https://arxiv.org/pdf/1603.07285.pdf) or [This Blog](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)
* [Weight Initialization](https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch) and [Pytorch functions for it](https://pytorch.org/docs/stable/nn.init.html)
* [Weights in BatchNorm](https://github.com/pytorch/pytorch/issues/16149), [affine in batchnorm](https://discuss.pytorch.org/t/affine-parameter-in-batchnorm/6005/3)
* Why to use Detach in this code, and why is it not used in generator step: [1](https://github.com/pytorch/examples/issues/116) and [2](https://stackoverflow.com/questions/46944629/why-detach-needs-to-be-called-on-variable-in-this-example) 

# Notes
* For reproducibility, manually set the random of pytorch and other python libraries. Refer [this](https://pytorch.org/docs/stable/notes/randomness.html) for reproducibility pytorch using CUDA.
* GAN notes [here]()
* Transpose Convolution: Like an opposite of convolution. For ex. Maps 1x1 to 3x3. 
* Upsampling: opposite of pooling. Fills in pixels by copying pixel values, using nearest neighbor or some other method.
* For keeping track of the generator’s learning progression, generate a fixed batch of latent vectors. We can pass it through generator to see visualize how generator improves.

# Other Useful Resources
* [GAN Hacks](https://github.com/soumith/ganhacks)
* [Pytorch Autograd Tutorials](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#gradients)
* Pytorch autograd [works](https://towardsdatascience.com/pytorch-autograd-understanding-the-heart-of-pytorchs-magic-2686cd94ec95#:~:text=The%20leaves%20of%20this%20graph,way%20using%20the%20chain%20rule%20.)

# TODO
* Check what is dilation in conv2d layer
* Check back-propagation in transpose convolution
* Weight initialization should use values from config
* Understand weight initialization in BatchNorm: how does it work?, what is affine used for?, how to initialize it properly
* Is there a choice to be made for sampling latent variable 