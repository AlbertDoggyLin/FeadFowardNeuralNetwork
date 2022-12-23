# Feadforward Neural Network prepared for genetic algorithm
## introduction
This repo is for feedforward that only provide dense layer of feed forward neural network (not even back propogation).
The activation function contains relu, sigmoid, and elu

## install it as site-packages
    pip install -i https://test.pypi.org/simple/ DNN-AlbertLin==0.0.2

## to run the code
1. create virtualenv
2. install package by requirement.txt
3. run test.py or if you want to customize the DNN more, see more detail from createSimpleModel in [src/DNN/\_\_init\_\_.py](src/DNN/__init__.py)

- git bash command(windows)
    ###
        git clone https://github.com/AlbertDoggyLin/FeedforwardNeuralNetwork
        cd FeedforwardNeuralNetwork
        pip install virtualenv
        virtualenv .
        source Scripts/activate
        pip install -r requirement.txt
        python src/test.py

- git bash command(mac or linux)
    ###
        git clone https://github.com/AlbertDoggyLin/FeedforwardNeuralNetwork
        cd FeedforwardNeuralNetwork
        pip3 install virtualenv
        virtualenv .
        source Scripts/activate
        pip3 install -r requirement.txt
        python src/test.py


