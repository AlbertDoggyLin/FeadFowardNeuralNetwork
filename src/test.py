from DNN import FFN, createSimpleModel

if __name__ == "__main__":
    model:FFN = createSimpleModel(24, 4, 2, 40, outputLayerActFunc='elu')
    import numpy as np
    input=np.random.random(24)
    print(model.cal(input))
    print(model.lastOutputs)
    w = model.getLayerWeight(1)
    w+=np.random.random(w.shape)
    model.setLayerWeight(1, w)
    print(model.cal(input))
    print(model.lastOutputs)