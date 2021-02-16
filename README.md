Master thesis of Jean-Martin Vlaeminck
======================================

This code was initially written based on the code from akamaster and from PyTorch's models.
However, it consistently performed 1% lower than akamaster's code, so I decided to give it up and base my code on that of akamaster.
In hindsight, maybe this code was working after all, as the akamaster version has a large variance in its results.
But anyway, it is probably better to iteratively improve the akamaster code than to go from sratch.

Sources for the code:
- https://github.com/akamaster/pytorch_resnet_cifar10/, by Yerlan Idelbayev
- https://github.com/pytorch/examples/blob/master/imagenet/main.py and https://github.com/pytorch/vision/
- minor source: https://github.com/izmailovpavel/torch_swa_examples/blob/master/models/preresnet.py
- minor sources: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
