# clockandcrocodile


## Description
- A resnet of 20 layers as a classifier , with 10 cross validation , the average training accuracy is : 0.9278 , 
the average test accuracy is : 0.5030 , and the highest accuracy of the entire data is :0.9240.
- With this classifier , find two examples that are similar with clock and crocodile at the same time(it depends on classifier 
and a hyperparameter "threshold")

clock that looks like crocodile

![](https://github.com/veetsin/clockandcrocodile/blob/master/1.png)

crocodile that looks like clock

![ ](https://github.com/veetsin/clockandcrocodile/blob/master/2.png)




- A DCGAN to generate images which are similar with clock and crocodile at the same time , but i failed , actually time is not 
enough  , otherwise i can try more architectures and hypermeters.




## Requirements
- [mxnet-cu90](https://mxnet.incubator.apache.org/install/index.html)
- [requirements](https://github.com/veetsin/clockandcrocodile/blob/master/requirements.txt)


## Usage
- classifier : train_res(k_cross)
- detector : print_similar_image(data_array ,label_target , net , threshold , ctx)
- generator(can't generate human recognized images yet)











