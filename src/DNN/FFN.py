import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try:
        from .Dense import Dense
    except:
        from Dense import Dense

__all__=['FFN']

class FFN:
    def __init__(self) -> None:
        self._layers:list['Dense']=[]

    def add(self, layer:'Dense'):
        layer.setParent(self)
        self._layers.append(layer)

    @property
    def outputDim(self)->int:
        return self._layers[-1]._units
    
    @property
    def lastOutputs(self)->list[np.ndarray]:
        return [layer._lastOutput.flatten() for layer in self._layers]

    def cal(self, x:np.ndarray):
        if needFlatten:=(x.ndim!=2):
            x=np.array([x.flatten()]).transpose()
        nextInput=x
        for layer in self._layers:
            nextInput=layer.cal(nextInput)
        if not needFlatten: return nextInput
        else: return nextInput.flatten()

    def getLayerWeight(self, layerIdx:int)->np.ndarray:
        """
        returned array on axis 0 is the weight of a neural
        example: 
        w = getLayerWeight
        w[0] is the first neural's weight
        """
        return self._layers[layerIdx]._W.copy()

    def setLayerWeight(self, layerIdx:int, modifiedW:np.ndarray):
        if self._layers[layerIdx]._W.shape==modifiedW.shape:
            self._layers[layerIdx]._W = modifiedW
        else:
            print(f'Wrong size for set weight, target w shape is {self._layers[layerIdx]._W.shape}, while yours is {modifiedW.shape}')

if __name__=="__main__":
    print(FFN())