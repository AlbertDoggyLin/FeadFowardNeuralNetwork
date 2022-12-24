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
    def info(self)->dict[str, int]:
        """
        inputDim: input dimention

        outputDim: output dimention

        layers Info: the amount of units for each layer by correspounded index
        """
        info:dict[str, int]={}
        info['inputDim']=self._layers[0]._inputSize
        info['outputDim']=self._layers[-1]._units
        info['layersInfo']=[layer._units for layer in self._layers]
        return info
    
    @property
    def lastOutputs(self)->list[np.ndarray]:
        return [layer._lastOutput.flatten() for layer in self._layers]

    def foward(self, x:np.ndarray):
        if needFlatten:=(x.ndim!=2):
            x=np.array([x.flatten()])
        nextInput=x.transpose()
        for layer in self._layers:
            nextInput=layer.cal(nextInput)
        if not needFlatten: return nextInput
        else: return nextInput.flatten()

    def getWeightAndBias(self, layerIdx:int)->tuple[np.ndarray, np.ndarray]:
        """
        returned array on axis 0 is the weight of a neural

        example:

        w, b = getWeightAndBias(1)

        w[0] and b[0] is the weights and bias of the first neural
        """
        return self._layers[layerIdx]._W.copy(), self._layers[layerIdx]._B.copy()

    def setWeightAndBias(self, layerIdx:int, modifiedW:np.ndarray| None=None, modifiedB:np.ndarray | None = None):
        if modifiedW is not None:
            if self._layers[layerIdx]._W.shape==modifiedW.shape:
                self._layers[layerIdx]._W = modifiedW.copy()
            else:
                print(f'Wrong size for set weight, target w shape is {self._layers[layerIdx]._W.shape}, while yours is {modifiedW.shape}')
        if modifiedB is not None:
            if self._layers[layerIdx]._B.shape==modifiedB.shape:
                self._layers[layerIdx]._B = modifiedB.copy()
            else:
                print(f'Wrong size for set weight, target w shape is {self._layers[layerIdx]._B.shape}, while yours is {modifiedB.shape}')

if __name__=="__main__":
    print(FFN())