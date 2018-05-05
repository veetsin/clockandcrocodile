from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet import image
from mxnet.gluon import nn
import mxnet as mx
import numpy as np
from time import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import os


#===========================import data , print some images , generate datasetd=================
path_clock = 'clock'
path_crocodile = 'crocodile'
img_list = []
img_list_clock = []
img_list_crocodile = [] 
for _,_,files in os.walk(path_clock):
    for file_name in files:
        if not file_name.endswith('.png'):
            continue
        img_dir = os.path.join(path_clock , file_name)
        img_arr = mx.image.imread(img_dir)
        img_arr = nd.transpose(img_arr,(2,0,1))
        img_arr = (img_arr.astype(np.float32)/127.5-1)
        img_arr = nd.array(img_arr.reshape((1,)+img_arr.shape))
        img_list.append(img_arr)
        img_list_clock.append(img_arr)
for _,_,files in os.walk(path_crocodile):
    for file_name in files:
        if not file_name.endswith('.png'):
            continue
        img_dir = os.path.join(path_crocodile , file_name)
        img_arr = mx.image.imread(img_dir)
        img_arr = nd.transpose(img_arr,(2,0,1))
        img_arr = (img_arr.astype(np.float32)/127.5-1)
        img_arr = nd.array(img_arr.reshape((1,)+img_arr.shape))
        img_list.append(img_arr)
        img_list_crocodile.append(img_arr)

train_augs=[
        image.HorizontalFlipAug(.5),
        image.BrightnessJitterAug(.5),
        image.HueJitterAug(.5),
        image.RandomCropAug((28,28)),
        ]
test_augs=[image.CenterCropAug((28,28))]

def apply_aug_list(img,augs):
    for f in augs:
        img = f(img)
    return img
    

def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx

def visualize(img_arr):
    #from CHW transpose to format HWC
    plt.imshow(((img_arr.asnumpy().transpose(1,2,0)+1.0)*127.5).astype(np.uint8)) 
    plt.axis('off')


def evaluate_accuracy(data_iterator, net, ctx):
    acc = nd.array([0])
    n = 0
    if isinstance(data_iterator, mx.io.NDArrayIter):
        data_iterator.reset()
    for batch in data_iterator:
        data = batch.data[0].as_in_context(ctx)
        label = batch.label[0].as_in_context(ctx)
        

        acc += sum((net(data).argmax(axis=1)==label)).asscalar()
        n += label.size
        
    return acc.asscalar()/n


def train(train_data, test_data, batch_size,net, loss, trainer, ctx, num_epochs, print_batches=None):
    print("Start training on ", ctx)
    his_testacc=[]
    for epoch in range(num_epochs):
        train_loss, train_acc, n, m = 0.0, 0.0, 0.0, 0.0
        if isinstance(train_data, mx.io.NDArrayIter):
            train_data.reset()
        start = time()
        if epoch == (num_epochs/4-1):
            trainer.set_learning_rate(trainer.learning_rate*.05)
        if epoch == (num_epochs/2-1):
            trainer.set_learning_rate(trainer.learning_rate*.05)
        if epoch == (num_epochs*3/4-1):
            trainer.set_learning_rate(trainer.learning_rate*.05)
        for batch in train_data:
            data = batch.data[0].as_in_context(ctx)
            label = batch.label[0].as_in_context(ctx)
            losses = []
            with autograd.record():
                outputs = net(data)
                losses = loss(outputs, label)
                losses.backward()
            train_acc += sum((outputs.argmax(axis=1)==label)).asscalar()
            train_loss += sum([l.sum().asscalar() for l in losses])
            trainer.step(batch_size)
            n += label.size
            m += label.size
#            print("Batch %d. Loss: %f, Train acc %f" % (
#                n, train_loss/n, train_acc/m
#            ))

        test_acc = evaluate_accuracy(test_data, net, ctx)
        his_testacc.append(test_acc)
        print("Epoch %d. Loss: %.3f, Train acc %.6f, Test acc %.3f, Time %.1f sec" % (
            epoch, train_loss/n, train_acc/m, test_acc, time() - start
        ))
    return train_acc/m , his_testacc , net




class Residual(nn.Block):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1,
                              strides=strides)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm()
        if not same_shape:
            self.conv3 = nn.Conv2D(channels, kernel_size=1,
                                  strides=strides)

    def forward(self, x):
        out = nd.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return nd.relu(out + x)
    
class ResNet(nn.Block):
    def __init__(self, num_classes, n ,verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.verbose = verbose
        # add name_scope on the outermost Sequential
        def stack(n):
            out = nn.Sequential()
            out.add(nn.Conv2D(4,kernel_size=3, padding=1),
                    nn.BatchNorm(),
                    nn.Activation(activation='relu')
            )
            for i in range(n):
                if i == 0:
                    out.add(Residual(4,same_shape=False))
                else: out.add(Residual(4))
            for i in range(n):
                if i == 0:
                    out.add(Residual(8,same_shape=False))
                else:out.add(Residual(8))
            for i in range(n):
                if i == 0:
                    out.add(Residual(16,same_shape=False))
                else:out.add(Residual(16))
                return out                
        with self.name_scope():
            b = nn.Sequential()
            b.add(
                nn.AvgPool2D(pool_size=4),
                nn.Dense(num_classes)
            )
            # chain all blocks together
            self.net = nn.Sequential()
            self.net.add(stack(n) , b)

    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out
    
