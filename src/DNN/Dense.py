try:
    from .FFN import FFN
    from .actFunction import getActFunc
except:
    from FFN import FFN
    from actFunction import getActFunc

import numpy as np

__all__=['Dense']

class Dense:
    def __init__(self, units, inputSize:int | None = None, act='relu') -> None:
        self._units:int = units
        self._actf = getActFunc(act)
        self._inputSize:int | None = inputSize
        self._W:np.ndarray | None = None
        self._B:np.ndarray | None = None
        self._parent:FFN | None = None
        self._lastOutput:np.ndarray | None = None
        if inputSize is not None:
            self._W=(np.random.random((units, inputSize))-0.5)*2
            self._B=(np.random.random((units, 1))-0.5)*2
    
    def setParent(self, parent:FFN):
        self._parent=parent
        if self._W is None:
            self._inputSize=parent.info['outputDim']
            self._W=(np.random.random((self._units, self._inputSize))-0.5)*2
            self._B=(np.random.random((self._units, 1))-0.5)*2

    def cal(self, input:np.ndarray):
        if self._W is None:
            raise Exception('W not set, please specify input size or indicate parent')
        self._lastOutput = self._actf(self._W@input-self._B)
        return self._lastOutput
