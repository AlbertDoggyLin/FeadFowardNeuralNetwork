__all__=['getActFunc']

import numpy as np

actFuncDic={
    'relu':lambda x:np.where(x > 0, x, 0),
    'sigmoid':lambda x: 1/(1+np.exp(-x)),
    'elu':lambda x: np.where(x > 0, x, np.exp(x)-1),
}

def getActFunc(actName:str):
    return actFuncDic[actName]
