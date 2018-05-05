#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 01:47:44 2018

@author: veetsin
"""

import os 
import matplotlib as mpl 
import matplotlib.image as mping
from matplotlib import pyplot as plt 

import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn 
from mxnet import autograd
import numpy as np
from mxnet.ndarray import random
import utils





epochs = 201
batch_size = 50
latent_z_size = 100


ctx = utils.try_gpu()

lr = .00001
beta1 = .5
data_array  = random.shuffle(nd.concatenate(utils.img_list))
for image in data_array:
    image_tem = (nd.transpose(image,axes=(1,2,0))+1)*127.5
    image = utils.apply_aug_list(image_tem,utils.test_augs)
    image = nd.transpose(image,(2,0,1))/127.5-1
train_data = mx.io.NDArrayIter(data = data_array , batch_size = batch_size)



 

#define the networks
#=============discriminator============
netD = nn.Sequential()
with netD.name_scope():
    netD.add(nn.Conv2D(channels=64,kernel_size=3,strides=2,padding=1))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(.2))
    netD.add(nn.Conv2D(channels=128,kernel_size=3,strides=2,padding=1))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(.2))
    netD.add(nn.Conv2D(channels=256,kernel_size=3,strides=2,padding=1))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(.2))
    netD.add(nn.Conv2D(channels=512,kernel_size=3,strides=2,padding=1))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(.2))
    netD.add(nn.Conv2D(channels=1,kernel_size=2))
    
#===============generator==================
netG = nn.Sequential()
with netG.name_scope():
    netG.add(nn.Conv2DTranspose(channels=512,kernel_size=4))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation(activation='relu'))
    netG.add(nn.Conv2DTranspose(channels=256,kernel_size=3,strides=2,padding=1))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation(activation='relu'))
    netG.add(nn.Conv2DTranspose(channels=128,kernel_size=4,strides=2,padding=1))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation(activation='relu'))
    netG.add(nn.Conv2DTranspose(channels=3,kernel_size=4,strides=2,padding=1))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation(activation='tanh'))
    
#loss , initialization , trainer 
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

netG.initialize(mx.init.Normal(.02),ctx=ctx)
netD.initialize(mx.init.Normal(.02),ctx=ctx)

trainerG = gluon.Trainer(netG.collect_params(),'adam',{'learning_rate':lr,'beta1':beta1})
trainerD = gluon.Trainer(netD.collect_params(),'adam',{'learning_rate':lr,'beta1':beta1})

#training loop
import time
import logging 

real_label = nd.ones((batch_size,),ctx=ctx)
fake_label = nd.zeros((batch_size,),ctx=ctx)


#custom metric
def eveluate(pred , label):
    pred = pred.flatten()
    label = label.flatten()
    return ((pred>.5) == label).mean()
metric = mx.metric.CustomMetric(eveluate)
logging.basicConfig(level=logging.DEBUG)
his_acc = []
his_errD = []
his_errG = []


#funtion to save the netG
path_G = 'net_G'
if not os.path.exists(path_G):
    os.makedirs(path_G)

    
for epoch in range(epochs):
    start_time = time.time()
    train_data.reset()
    for batch in train_data:
        #========G fixed , train D,maxmize log(D(x)) + los(1-D(G(z)))======
        data = batch.data[0].as_in_context(ctx)
        latent_z = mx.nd.random_normal(0,1,shape=(batch_size,latent_z_size,1,1),ctx=ctx)
        
        with autograd.record():
            output = netD(data).reshape((-1,1))
            errD_real = loss(output,real_label)
            metric.update([output,],[real_label,])
            
            fake = netG(latent_z)
            output_fake = netD(fake).reshape((-1,1))
            errD_fake = loss(output_fake,fake_label)
            errD = errD_real + errD_fake
            errD.backward()
            metric.update([output_fake],[fake_label])
        
        trainerD.step(batch_size)
            
        #=======D fixed , train G, maxmize log(D(G(z)))============
        with autograd.record():
            fake = netG(latent_z)
            output_fake = netD(fake).reshape((-1,1))
            errG = loss(output_fake,real_label)
            errG.backward()
        
        trainerG.step(batch_size)
      
    end_time = time.time() 
    _,acc = metric.get()
    his_acc.append(acc)
    his_errD.append(nd.mean(errD).asscalar())
    his_errG.append(nd.mean(errG).asscalar())
    
    logging.info('epoch: %i ; discriminator loss:%f ; generator loss:%f ; training acc:%f ; time:%f'
                 %(epoch , nd.mean(errD).asscalar(),nd.mean(errG).asscalar(),acc,end_time-start_time))
    metric.reset()
    if (0 < epoch < 10) or ((epoch % 10) == 0):
        fig = plt.figure(figsize=(9,9))
        for i in range(4):
            latent_z = mx.nd.random_normal(0,1,shape=(4,latent_z_size,1,1),ctx=ctx)
            fake = netG(latent_z)
            plt.subplot(2,2,i+1)
            utils.visualize(fake[i])
        plt.show()    
    
netG.save_params('generater')
#plot the data
x_axis = np.linspace(0,epochs,len(his_acc))
plt.figure(figsize=(12,12))
plt.plot(x_axis,his_acc,label='accuracy')
plt.plot(x_axis,his_errD,label='error of Discriminator')
plt.plot(x_axis,his_errG,label='error of Generator')
plt.xlabel('epoch')
plt.legend()
plt.show()


        
    
    
