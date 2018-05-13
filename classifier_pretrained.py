# -*- coding: utf-8 -*-
from mxnet import nd
from mxnet.ndarray import random
from mxnet import gluon
import mxnet as mx
from matplotlib import pyplot as plt
import numpy as np
import os
import utils



#==============print some images===================== 

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


#======================shuffle data and set some params======================
batch_size = 50
ctx = utils.try_gpu()
label = nd.stack(nd.zeros([int(len(utils.img_list)/2),1],ctx=ctx),nd.ones([int(len(utils.img_list)/2),1],ctx=ctx)).reshape([len(utils.img_list)])
data = nd.concatenate(utils.img_list)
mx.random.seed(2)
data = random.shuffle(data)
mx.random.seed(2)
label = random.shuffle(label)



epochs = 200
loss = gluon.loss.SoftmaxCrossEntropyLoss()
k_cross = 10
path_net = 'net'
if not os.path.exists(path_net):
    os.makedirs(path_net)


from mxnet.gluon.model_zoo import vision as models
from mxnet import init

def train_net(k):
    his_testacc=[]
    his_trainacc=[]
    for n in range(k):
        pretrained_net = models.resnet18_v2(pretrained=True)
        net = models.resnet18_v2(classes=2)
        net.features = pretrained_net.features
        net.output.initialize(init.Xavier())
        net.collect_params().reset_ctx(ctx)
#        resnet = utils.ResNet(2,1) 
#        net = utils.Perceptron(2)
#        net.initialize(ctx=ctx)

#use k cross validation
        train_index = list(range(0,int(n*1000/k)))+list(range(int((n+1)*1000/k),1000))
        train_data_array = data[train_index]
        label_train_array = label[train_index]
        test_data_array = data[int(n*1000/k):int((n+1)*1000/k)]
        label_test_array = label[int(n*1000/k):int((n+1)*1000/k)]
#use data augmentation 
        train_data_array_ori = nd.transpose(train_data_array, (0,2,3,1))*255
        train_data_array = nd.stack(*[utils.apply_aug_list(d,utils.test_augs) for d in train_data_array_ori])
        train_data_array_aug = nd.stack(*[utils.apply_aug_list(d,utils.train_augs) for d in train_data_array_ori])
        train_data_array = nd.stack(train_data_array,train_data_array_aug).reshape([int(2*(1000-1000/k)),utils.size,utils.size,3])
        train_data_array = nd.transpose(train_data_array, (0,3,1,2))/255
        label_train_array = nd.stack(label_train_array,label_train_array).reshape([int(2*(1000-1000/k))])


            
        test_data_array = nd.transpose(test_data_array, (0,2,3,1))*255
        test_data_array = nd.stack(*[utils.apply_aug_list(d,utils.test_augs) for d in test_data_array])
        test_data_array = nd.transpose(test_data_array, (0,3,1,2))/255
           

#load trainging data and test data  
        
        train_data = mx.io.NDArrayIter(data = train_data_array,label=label_train_array,batch_size=batch_size,shuffle=True)
        test_data = mx.io.NDArrayIter(data = test_data_array,label=label_test_array,batch_size=batch_size,shuffle=True)
        trainer = gluon.Trainer(net.collect_params(),
                                'sgd', {'learning_rate':.008 , 'wd':.3 })
        train_acc , acc, net_tem = utils.train(train_data, test_data,batch_size ,net, loss,
                    trainer, ctx, num_epochs=epochs)
        his_testacc.append(acc)
        his_trainacc.append(train_acc)
        net_tem.save_params(os.path.join(path_net,str(n)))#save net
        
    return his_trainacc ,his_testacc





train_acc , his_testacc = train_net(k_cross)
average_acc = 0
x_axis = np.linspace(0,epochs,len(his_testacc[0]))
plt.figure(figsize=(20,20))
# compute average training acc and test acc
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
# compare the acc of the 10 net on the entire data , get the best classifier
for i in range(k_cross):
#    net = utils.Perceptron(2)
    net = models.resnet18_v2(classes=2)
    test_data_array = data
    for image in test_data_array:
        image_tem = nd.transpose(image,axes=(1,2,0))*255
        image = utils.apply_aug_list(image_tem,utils.test_augs)
        image = nd.transpose(image,(2,0,1))/255
    test_data = mx.io.NDArrayIter(data = test_data_array,label=label,batch_size=batch_size,shuffle=True)
    net.load_params(os.path.join(path_net,str(i)) , ctx = ctx)
    if utils.evaluate_accuracy(test_data,net,ctx) > max_acc:
        max_acc = utils.evaluate_accuracy(test_data,net,ctx)
        max_k = i
print('The best net is net%i , accuracy of the entire data is :%.4f '% (max_k , max_acc))
os.rename(os.path.join(path_net,str(max_k)),os.path.join(path_net,'bestnet_pretrained'))
        











#==========use the classifier we got to get the images that look like clock and crocodile at the same time======
#net = utils.Perceptron(2)
#net = models.mobilenet0_25(classes=2)
net = models.resnet18_v2(classes=2)
net.load_params(os.path.join(path_net,'bestnet_pretrained'),ctx=ctx)
data_array_clock = nd.concatenate(utils.img_list_clock)
data_array_crocodile = nd.concatenate(utils.img_list_crocodile)
threshold = 6


# function to print the images that looks like clock and crocodile at the sanme time 
def print_similar_image(data_array ,label_target , net , threshold , ctx):
    for image in data_array:
        image_tem = nd.transpose(image,axes=(1,2,0))*255
        image = utils.apply_aug_list(image_tem,utils.test_augs)
        image = nd.transpose(image,(2,0,1))/255

    DataIter = mx.io.NDArrayIter(data = data_array, batch_size = batch_size,shuffle=True)
    i = 0
    for batch in DataIter:
        data = batch.data[0].as_in_context(ctx)
        outputs = net(data)

        for n , prob in enumerate(outputs):
            if prob[label_target] > threshold:
                plt.imshow((data_array[i + n].asnumpy().transpose(1,2,0)*255).astype(np.uint8)) 
            plt.show()    
        i += batch_size
 
print_similar_image(data_array_clock , 1 , net , threshold*2 , ctx)
print_similar_image(data_array_crocodile , 0 , net , threshold , ctx)

#=========another way to get the target images:calculate the hamming distance between images====
from PIL import Image
path_clock = 'clock'
path_crocodile = 'crocodile'
for _,_,files in os.walk(path_clock):
    for file_name in files:
        if not file_name.endswith('.png'):
            continue
        file_name = os.path.join(path_clock,file_name)
        clock_iamge = Image.open(file_name)
        for _,_,files in os.walk(path_crocodile):
            for file_name in files:
                if not file_name.endswith('.png'):
                    continue
                file_name = os.path.join(path_crocodile,file_name)
                crocodile_image = Image.open(file_name)
                hamming_distance = utils.DHash.hamming_distance(clock_iamge,crocodile_image)
                if hamming_distance < 15:
                    print('got it')
                    plt.figure(figsize=(2,2))
                    plt.imshow(clock_iamge)
                    plt.figure(figsize=(2,2))
                    plt.imshow(crocodile_image)
                    plt.show()
                crocodile_image.close()
        clock_iamge.close()
