# Fall 2019 IE534/CS598:  HW4

By moutaz elias

Build the Residual Network speciﬁed in Figure 1 and achieve at least 60% test accuracy.
In the homework, you should deﬁne your “Basic Block” as shown in Figure 2. For each weight layer, it should contain 3 × 3 ﬁlters for a speciﬁc number of input channels and output channels. The output of a sequence of ResNet basic blocks goes through a max pooling layer with your own choice of ﬁlter size, and then goes to a fully-connected layer. The hyperparameter speciﬁcation for each component is given in Figure 1. Note that the notation follows the notation in He et al. (2015).


## Part I: Build the required Residual Network

### Test accuracy
**`61.48%`**.
