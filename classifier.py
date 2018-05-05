# -*- coding: utf-8 -*-
from mxnet import nd
from mxnet.ndarray import random
from mxnet import gluon
import mxnet as mx
from matplotlib import pyplot as plt
import numpy as np
import os
import utils



#print some images 

fig = plt.figure(figsize=(3.5,3.5))
for i in range(9):
    plt.subplot(3,3,i+1)
    utils.visualize(utils.img_list_clock[i][0])
plt.suptitle('clock')
plt.show()
fig = plt.figure(figsize=(3.5,3.5))
for i in range(9):
    plt.subplot(3,3,i+1)
    utils.visualize(utils.img_list_crocodile[i][0])
plt.suptitle('crocodile')
plt.show()


batch_size = 50
ctx = utils.try_gpu()
label = nd.stack(nd.zeros([int(len(utils.img_list)/2),1],ctx=ctx),nd.ones([int(len(utils.img_list)/2),1],ctx=ctx)).reshape([len(utils.img_list)])
data = nd.concatenate(utils.img_list)
mx.random.seed(2)
data = random.shuffle(data)
mx.random.seed(2)
label = random.shuffle(label)



epochs = 400
loss = gluon.loss.SoftmaxCrossEntropyLoss()
k_cross = 10
path_net = 'net'
if not os.path.exists(path_net):
    os.makedirs(path_net)



def train_res(k):
    his_testacc=[]
    his_trainacc=[]
    for n in range(k):
#       3*6+2=20layer
        resnet = utils.ResNet(2,3) 
        resnet.initialize(ctx=ctx, init=mx.initializer.MSRAPrelu())
        train_index = list(range(0,int(n*1000/k)))+list(range(int((n+1)*1000/k),1000))
        train_data_array = data[train_index]
        label_train_array = label[train_index]
        test_data_array = data[int(n*1000/k):int((n+1)*1000/k)]
        label_test_array = label[int(n*1000/k):int((n+1)*1000/k)]
        for image in train_data_array:
            image_tem = (nd.transpose(image,axes=(1,2,0))+1)*127.5
            image = utils.apply_aug_list(image_tem,utils.train_augs)
            image = nd.transpose(image,(2,0,1))/127.5-1
        for image in test_data_array:
            image_tem = (nd.transpose(image,axes=(1,2,0))+1)*127.5
            image = utils.apply_aug_list(image_tem,utils.test_augs)
            image = nd.transpose(image,(2,0,1))/127.5-1
        train_data = mx.io.NDArrayIter(data = train_data_array,label=label_train_array,batch_size=batch_size,shuffle=True)
        test_data = mx.io.NDArrayIter(data = test_data_array,label=label_test_array,batch_size=batch_size,shuffle=True)
        trainer = gluon.Trainer(resnet.collect_params(),
                                'sgd', {'learning_rate': .00075, 'wd':.32 ,'momentum':.8})
        train_acc , acc,net = utils.train(train_data, test_data,batch_size ,resnet, loss,
                    trainer, ctx, num_epochs=epochs)
        his_testacc.append(acc)
        his_trainacc.append(train_acc)
        net.save_params(os.path.join(path_net,str(n)))
        
    return his_trainacc ,his_testacc





train_acc , his_testacc = train_res(k_cross)
average_acc = 0
x_axis = np.linspace(0,epochs,len(his_testacc[0]))
plt.figure(figsize=(20,20))
for i in range(k_cross):
    plt.plot(x_axis,his_testacc[i],label=str(i))
    average_acc += his_testacc[i][-1]
plt.xlabel('epoch')
plt.ylabel('test_acc')
plt.legend()
plt.show()
print('The average training accuracy is : %.4f , the average test accuracy is : %.4f '% (np.mean(train_acc) , average_acc/k_cross))




max_acc = 0
max_k = 0
for i in range(k_cross):
    net = resnet = utils.ResNet(2,3)
    test_data_array = data
    for image in test_data_array:
        image_tem = (nd.transpose(image,axes=(1,2,0))+1)*127.5
        image = utils.apply_aug_list(image_tem,utils.test_augs)
        image = nd.transpose(image,(2,0,1))/127.5-1
    test_data = mx.io.NDArrayIter(data = test_data_array,label=label,batch_size=batch_size,shuffle=True)
    net.load_params(os.path.join(path_net,str(i)) , ctx = ctx)
    if utils.evaluate_accuracy(test_data,net,ctx) > max_acc:
        max_acc = utils.evaluate_accuracy(test_data,net,ctx)
        max_k = i
print('The best net is net%i , accuracy of the entire data is :%.4f '% (max_k , max_acc))
os.rename(os.path.join(path_net,str(max_k)),os.path.join(path_net,'bestnet2'))
        

net = utils.ResNet(2,3)
net.load_params(os.path.join(path_net,'bestnet2'),ctx=ctx)
data_array_clock = nd.concatenate(utils.img_list_clock)
data_array_crocodile = nd.concatenate(utils.img_list_crocodile)
threshold = 0.2


# function to print the images that looks like clock and crocodile at the sanme time 
def print_similar_image(data_array ,label_target , net , threshold , ctx):
    for image in data_array:
        image_tem = (nd.transpose(image,axes=(1,2,0))+1)*127.5
        image = utils.apply_aug_list(image_tem,utils.test_augs)
        image = nd.transpose(image,(2,0,1))/127.5-1

    DataIter = mx.io.NDArrayIter(data = data_array, batch_size = batch_size,shuffle=True)
    i = 0
    for batch in DataIter:
        data = batch.data[0].as_in_context(ctx)
        outputs = net(data)
        for n , prob in enumerate(outputs):
            if prob[label_target] > threshold:
                plt.imshow(((data_array[i + n + 1].asnumpy().transpose(1,2,0)+1.0)*127.5).astype(np.uint8)) 
            plt.show()    
        i += batch_size
 
print_similar_image(data_array_clock , 1 , net , threshold*1.2 , ctx)
print_similar_image(data_array_crocodile , 0 , net , threshold , ctx)
    

