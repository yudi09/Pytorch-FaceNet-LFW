# Pytorch-FaceNet-LFW

## Abstract
This repository contains the implementation of [FaceNet paper](https://arxiv.org/pdf/1503.03832.pdf). Embeddings can be generated from AlexNet, ResNet18, ResNet152. Note that embeddings are generated in batches, with batch_size of 16, 32 or 64 and embeddings dimension can be 64 or 128. Once batch embeddings are generated, online random tiplets are generated and CNN architecture is learnt through minimizing triplet loss function. [LFW Dataset](http://vis-www.cs.umass.edu/lfw/) is used for experiments.

## Requirements
1. python3.6
2. [pytorch](http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl)
3. pytorch-vision
4. pillow
5. pickle
6. cuda version 9.0/9.1
7. cuDNN >= 7.0

```bash
pip install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl pytorch-vision pillow pickle
```

## Dataset
LFW colored images are being used. Only those identities are selected which have 20 - 30 images. Data is then divided into train, dev and test directories with ~80%, ~10%, ~10% split respectively.

## Results

Architecture  | batch_size | embedding_dim | train_acc, dev_acc vs iterations | train_loss vs iterations
----|----|----|----|----|
ResNet18 | 32 | 128 | ![Screen Shot](results/resnet1.png) | ![Screen Shot](results/resnet2.png) <br>

ResNet18 | 64 | 128 | ![Screen Shot](results/resnet3.png) | ![Screen Shot](results/resnet4.png) <br>

AlexNet | 32 | 128 | ![Screen Shot](results/alexnet1.png) | ![Screen Shot](results/alexnet2.png) <br>

ResNet152 | 32 | 128 | ![Screen Shot](results/resnet5.png) | ![Screen Shot](results/resnet6.png) <br>

ResNet152 | 32 | 64 | ![Screen Shot](results/resnet7.png) | ![Screen Shot](results/resnet8.png) <br>



## References
* [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
* [Coursera: Convolutional Neural Networks week 4](https://www.coursera.org/learn/convolutional-neural-networks/home/welcome)
* [siamese-triplet](https://github.com/adambielski/siamese-triplet)
* [triplet-network-pytorch](https://github.com/andreasveit/triplet-network-pytorch)
* [facenet_pytorch](https://github.com/liorshk/facenet_pytorch)




