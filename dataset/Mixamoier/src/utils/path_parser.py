import os
import platform
import sys
import yaml
from src.utils.config import Config
# import torch as tc

'''
global variable.
'''
__glb_config__ = None


def glb_cfg(cfg_path='config/moret.yaml'):
    global __glb_config__
    if __glb_config__ is None:
        retry = 4
        for _ in range(retry):
            if os.path.exists(cfg_path):
                break
            cfg_path = '../'+cfg_path
        with open(cfg_path,'r') as f:
            yaml_config = yaml.safe_load(f)
            __glb_config__ = Config(name='glb_cfg')
            for k in yaml_config:
                __glb_config__.param = (k,yaml_config[k])
    assert __glb_config__ is not None
    return __glb_config__

if platform.system().lower() == 'windows':
    glb_data_path = glb_cfg().data_path_win
    glb_code_path = glb_cfg().code_path_win
elif platform.system().lower() == 'linux':
    glb_data_path = glb_cfg().data_path_linux
    glb_code_path = glb_cfg().code_path_linux
    if not os.path.exists(glb_data_path):
        glb_data_path = glb_cfg().data_path_linux2
        glb_code_path = glb_cfg().code_path_linux2

elif platform.system().lower() == 'darwin':
    glb_data_path = glb_cfg().data_path_darwin
    glb_code_path = glb_cfg().code_path_darwin
else:
    assert False, print(f'unsupported platform {platform.system().lower()}')

def glb_device():
    device = 'cuda' if glb_cfg().cuda_preferred and tc.cuda.is_available() else 'cpu'
    return device

def download_path():
    return __glb_config__.chrome_download_path

def profile_path():
    return __glb_config__.chrome_profile_path

def safe_filename(filename:str):
    return filename.replace('\\','@') # there might not be a name with '@'

def unsafe_filename(filename):
    return filename.replace('@','\\')

def is_plt_available():
    try:
        import matplotlib.pyplot as plt
    except:
        return False
    return True

glb_with_view = is_plt_available()

def pool_config_path(*args):
    if platform.system().lower() == 'windows':
        glb_pool_config_path = os.path.join(glb_data_path,glb_cfg().pool_config_path_win)
    elif platform.system().lower() == 'linux':
        glb_pool_config_path = os.path.join(glb_data_path,glb_cfg().pool_config_path_linux)
    elif platform.system().lower() == 'darwin':
        glb_pool_config_path = os.path.join(glb_data_path, glb_cfg().pool_config_path_darwin)
    else:
        assert False, print(f'unsupported platform {platform.system().lower()}')
    if args is None or len(args) == None:
        return glb_pool_config_path
    out_path = glb_pool_config_path
    for arg in args:
        out_path = os.path.join(out_path,arg if not arg.startswith('.') else arg[2:])
    return out_path

def data_path(*args):
    if not os.path.exists(glb_data_path):
        if platform.system().lower() == 'windows':
            os.system(f'mkdir {glb_data_path}')
        elif platform.system().lower() == 'linux':
            os.system(f'mkdir {glb_data_path}')
        elif platform.system().lower() == 'darwin':
            os.system(f'mkdir {glb_data_path}')
        else:
            assert False, print(f'unsupported platform {platform.system().lower()}')
    if args is None or len(args) == None:
        return glb_data_path
    out_path = glb_data_path
    for arg in args:
        out_path = os.path.join(out_path,arg if not arg.startswith('.') else arg[2:])
    return out_path

def pretrain_path(f=None):
    cur_dir = data_path('pretrain')
    if not os.path.exists(cur_dir):
        if platform.system().lower() == 'windows':
            os.system(f'mkdir {cur_dir}')
        elif platform.system().lower() == 'linux':
            os.system(f'mkdir {cur_dir}')
        elif platform.system().lower() == 'darwin':
            os.system(f'mkdir {cur_dir}')
        else:
            assert False, print(f'unsupported platform {platform.system().lower()}')
    return cur_dir if f is None else os.path.join(cur_dir,f)

def in_path(f=None):
    cur_dir = data_path('in_data')
    if not os.path.exists(cur_dir):
        if platform.system().lower() == 'windows':
            os.system(f'mkdir {cur_dir}')
        elif platform.system().lower() == 'linux':
            os.system(f'mkdir {cur_dir}')
        elif platform.system().lower() == 'darwin':
            os.system(f'mkdir {cur_dir}')
        else:
            assert False, print(f'unsupported platform {platform.system().lower()}')
    return cur_dir if f is None else os.path.join(cur_dir,f)

def log_path(f=None):
    cur_dir = data_path('log')
    if not os.path.exists(cur_dir):
        if platform.system().lower() == 'windows':
            os.system(f'mkdir {cur_dir}')
        elif platform.system().lower() == 'linux':
            os.system(f'mkdir {cur_dir}')
        elif platform.system().lower() == 'darwin':
            os.system(f'mkdir {cur_dir}')
        else:
            assert False, print(f'unsupported platform {platform.system().lower()}')
    return cur_dir if f is None else os.path.join(cur_dir,f)

def result_path(f=None):
    cur_dir = data_path('result')
    if not os.path.exists(cur_dir):
        if platform.system().lower() == 'windows':
            os.system(f'mkdir {cur_dir}')
        elif platform.system().lower() == 'linux':
            os.system(f'mkdir {cur_dir}')
        elif platform.system().lower() == 'darwin':
            os.system(f'mkdir {cur_dir}')
        else:
            assert False, print(f'unsupported platform {platform.system().lower()}')
    return cur_dir if f is None else os.path.join(cur_dir,f)

def local_result_path(f=None):
    raise NotImplementedError

def add_sys_path(cur_file):
    sys.path.append(os.path.dirname(cur_file))
