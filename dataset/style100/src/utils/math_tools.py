import numpy as np
import os


def softmax(x, **kw):
    softness = kw.pop("softness", 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, **kw):
    return -softmax(-x, **kw)


def alpha(t):
    return 2.0 * t * t * t - 3.0 * t * t + 1


def lerp(a, l, r):
    return (1 - a) * l + a * r

def rect_iter_rates(rates):
    '''
        return a list where element refects the ratio of current layer when deforming a mesh.
        eg. [0.2,0.3,0.5] -> [1,0.6,0.5]
        Ah..a wrong order from the input cfg...
    '''
    _rates = rates[::-1]
    ret = [None] * len(_rates)
    pre = [0] * (len(_rates)+1)
    for i in range(len(_rates)):
        pre[i+1] = pre[i] + _rates[i]
    for i in reversed(range(len(_rates))):
        ret[i] = (pre[i+1] - pre[i]) / pre[i+1]
    return ret[::-1]

if __name__ == '__main__':
    print('hello math_tool.')
    # normilze_sk(anim=None)
