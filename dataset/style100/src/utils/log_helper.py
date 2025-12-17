import datetime
from src.utils.path_parser import *
import numpy as np
try:
    import torch as tc
    from collections import Iterable
except:
    pass

from time import time

glb_st_time = time()

def printf(*args,**kwargs):
    file_desc = __file__
    if glb_code_path in file_desc:
        file_desc = file_desc[len(glb_code_path):]
    # new_args = [f'[{datetime.datetime.now()}@{file_desc}] ']
    new_args = [f'[{datetime.datetime.now()}] ']
    new_args.extend(args)
    new_kwargs = {}
    new_kwargs.update(kwargs)
    for k in ['is_log','log_name']:
        if k in new_kwargs:
            del new_kwargs[k]
    print(*new_args)
    if 'is_log' in kwargs and kwargs['is_log']:
        log_name = 'default_log.log' if 'log_name' not in kwargs else kwargs['log_name']
        with open(log_path(log_name), 'a') as f:
            for arg in new_args:
                f.write(arg)
            f.write('\n')
            f.flush()

glb_time = None
def log_tpoint(*arg):
    global glb_time
    lst = []
    lst.extend(arg)
    cur_consume = 0
    if glb_time is not None:
        cur_consume = time() - glb_time
    glb_time = time()
    lst = [f'LogTPoint[{cur_consume:.4f}]'] + lst
    print(*lst)

def mem_usage(obj:object):
    mem_cnt = 0
    if obj is None:
        return mem_cnt
    if isinstance(obj,dict):
        mem_cnt += sys.getsizeof(obj) / (1024**2) # MB
        for k in obj:
            mem_cnt += mem_usage(obj[k])
    elif isinstance(obj,np.ndarray):
        mem_cnt += sys.getsizeof(obj) / (1024 ** 2)  # MB
    elif isinstance(obj,tc.Tensor):
        mem_cnt += (obj.element_size() * obj.nelement()) / (1024**2) # MB
    elif isinstance(obj,str):
        mem_cnt += sys.getsizeof(obj) / (1024 ** 2)  # MB
    elif isinstance(obj,Iterable):
        for e_obj in obj:
            mem_cnt += mem_usage(e_obj)
    else:
        mem_cnt += sys.getsizeof(obj) / (1024 ** 2)  # MB
    return mem_cnt


if __name__ == '__main__':
    print('hello utils')
    # printf('dawda')
    # printf('111why',is_log=True)

    # print(isinstance('dawda',Iterable))
    # print(isinstance([12,2,3],Iterable))
    # print(isinstance((12,3,2),Iterable))
    # print(isinstance(np.array([1,2,3]),Iterable))
    # print(isinstance(tc.tensor([123,42]),Iterable))
    print(mem_usage([1]*100000))
    print(mem_usage('1'*100000))
    print(mem_usage(np.ones(shape=(1000,100))))
    print(mem_usage(tc.ones(size=(1000, 100))))
    a = {
        'i':[1]*100000,
        's': '1'*100000,
        'n':np.ones(shape=(1000,100)),
        't':tc.ones(size=(1000,100)),
    }
    print(mem_usage(a))


