# Feed Foward Neural Network prepared for generic algorithm
## introduction
This repo is for feed forward that only provide dense layer of feed forward neural network (not even back propogation).
The activation function contains relu, sigmoid, and elu
## to run the code
1. create virtualenv
2. install package by requirement.txt
3. run test.py or if you want to customize the DNN more, see more detail from createSimpleModel in DNN.__init.py__

- git bash command(windows)
    ###
        git clone https://github.com/AlbertDoggyLin/FeadFowardNeuralNetwork
        cd FeadFowardNeuralNetwork
        pip install virtualenv
        virtualenv .
        source Scripts/activate
        pip install -r requirement.txt
        python src/test.exe

- git bash command(mac or linux)
    ###
        git clone https://github.com/AlbertDoggyLin/FeadFowardNeuralNetwork
        cd FeadFowardNeuralNetwork
        pip3 install virtualenv
        virtualenv .
        source Scripts/activate
        pip3 install -r requirement.txt
        python src/test.exe


