import math
import numpy as np

#inverse trig function through its cos and sin value
def inverse_trig(cosval, sinval):
    if cosval >= 0 and sinval >= 0:
        return math.acos(cosval)
    elif cosval < 0 and sinval < 0:
        return -math.acos(cosval)
    elif cosval >= 0 and sinval < 0:
        return math.asin(sinval)
    elif cosval < 0 and sinval >= 0:
        return math.acos(cosval)

def cartesian_product_simple_transpose(arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[i, ...] = a
    return arr.reshape(la, -1).T