try:
    from .FFN import *
    from .FFN import FFN
    from .Dense import *
except:
    from FFN import *
    from Dense import *
from enum import Enum
import numpy as np

__all__=['FFN', 'Dense', 'createSimpleModel', 'get_direction', 'init_from_chromesome']

def createSimpleModel(inputSize:int, outputUnits:int, hiddenLayerSize:int, hiddenLayerUnits:int, 
        hiddenLayerActFunc:str='relu', outputLayerActFunc:str='sigmoid') -> FFN:
    model = FFN()
    model.add(Dense(hiddenLayerUnits, inputSize=inputSize, act=hiddenLayerActFunc))
    for _ in range(hiddenLayerSize - 1):
        model.add(Dense(hiddenLayerUnits, act=hiddenLayerActFunc))
    model.add(Dense(outputUnits, act=outputLayerActFunc))
    return model

class Direction(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

def get_direction(model:FFN, feature:np.ndarray)->Direction:
    res = np.argmax(model.foward(feature))
    if res==0: return Direction.UP
    elif res==1: return Direction.DOWN
    elif res==2: return Direction.LEFT
    elif res==3: return Direction.RIGHT
    else: raise Exception('error on getdirection, res=', res)
    
def init_from_chromesome(model:FFN, chromesome:list[np.ndarray[np.ndarray]]):
    for i, layer in enumerate(chromesome):
        w, bias = layer[:, 1:], layer[:, 0:1]
        model.setWeightAndBias(i, modifiedW = w, modifiedB = bias)


if __name__ == "__main__":
    model:FFN = createSimpleModel(24, 4, 2, 40)
