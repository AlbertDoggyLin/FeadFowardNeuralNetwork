try:
    from .FFN import *
    from .Dense import *
except:
    from FFN import *
    from Dense import *

__all__=['FFN', 'Dense', 'createSimpleModel']

def createSimpleModel(inputSize:int, outputUnits:int, hiddenLayerSize:int, hiddenLayerUnits:int, 
        hiddenLayerActFunc:str='relu', outputLayerActFunc:str='sigmoid') -> FFN:
    model = FFN()
    model.add(Dense(hiddenLayerUnits, inputSize=inputSize, act=hiddenLayerActFunc))
    for _ in range(hiddenLayerSize - 1):
        model.add(Dense(hiddenLayerUnits, act=hiddenLayerActFunc))
    model.add(Dense(outputUnits, act=outputLayerActFunc))
    return model

if __name__ == "__main__":
    model:FFN = createSimpleModel(24, 4, 2, 40)
