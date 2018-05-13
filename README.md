# clockandcrocodile


## Description
### task1:classification 
- use 10 cross validation,customized resnet20/pretrained resnet18/perceptron , weight decay , data augmentation 
- The average training accuracy is : 0.9278 , 
the average test accuracy is : 0.5030 , and the highest accuracy of the entire data is :0.9240.（with customized resnet)
- The average training accuracy is : 0.9992 , the average test accuracy is : 0.5220 
The best net is net3 , accuracy of the entire data is :0.9060 (with pretrained resnet on Imagenet)


### task2:find images:

- With this classifier , find two examples that are similar with clock and crocodile at the same time(it depends on classifier 
and a hyperparameter "threshold")

clock that looks like crocodile

![](https://github.com/veetsin/clockandcrocodile/blob/master/1.png)

crocodile that looks like clock
![](https://github.com/veetsin/clockandcrocodile/blob/master/2.png)
![](https://github.com/veetsin/clockandcrocodile/blob/master/3.png)


- compare hamming distance between images[repo](https://github.com/hjaurum/DHash)

pair1
![](https://github.com/veetsin/clockandcrocodile/blob/master/hash_1.png) ![](https://github.com/veetsin/clockandcrocodile/blob/master/hash_1_1.png)


pair2
![](https://github.com/veetsin/clockandcrocodile/blob/master/hash_2.png) ![](https://github.com/veetsin/clockandcrocodile/blob/master/hash_2_1.png)

pair3
![](https://github.com/veetsin/clockandcrocodile/blob/master/hash_3.png) ![](https://github.com/veetsin/clockandcrocodile/blob/master/hash_3_1.png)



### task3:generate images(still can't generate high quality images)
- A DCGAN to generate images which are similar with clock and crocodile at the same time , but i failed , actually time is not 
enough  , otherwise i can try more architectures and hypermeters.




## Requirements
- [mxnet-cu90](https://mxnet.incubator.apache.org/install/index.html)
- [requirements](https://github.com/veetsin/clockandcrocodile/blob/master/requirements.txt)


## Usage
- classifier_pretrained : train_res(k_cross) , set lr , wd in 'Trainer'
- detector : print_similar_image(data_array ,label_target , net , threshold , ctx) / or set the hamming distance between images
- generator(can't generate human recognized images yet)











