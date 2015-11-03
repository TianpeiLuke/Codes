#utf-8
import numpy as np

def wrapper(func):
    def inner(*args, **kwargs):
       res = func(*args, **kwargs)
       if res < 0:
          res = 0
       return res
    return inner

@wrapper
def linear(w, x):
    import numpy as np
    return np.dot(np.array(w), np.array(x))

w  = [1, -1, 2]
x =  [0, 1, 1]

print linear(w,x)

