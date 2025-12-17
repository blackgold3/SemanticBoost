
import copy

class Config:
    '''
        a generalizable class that serves parameter settings of different models.
        mainly used as an inline arg-parser.
    '''
    def __init__(self,name='default-config',**kwargs):
        self.name = name
        self._params = set([k for k in kwargs]) # only save their names.
        for k in self._params:
            setattr(self, k, kwargs[k])
        self._params_sgn = None
    @property
    def param(self):
        assert False

    @param.setter
    def param(self,value):
        assert len(value) == 2
        setattr(self,value[0],value[1])
        self._params.add(value[0])

    @property
    def params(self):
        return self._params.copy()

    @params.setter
    def params(self,value):
        assert False

    def join(self,cfg):
        assert isinstance(cfg,Config)
        params = self.params
        for k in cfg.params:
            if k in params:
                print(f'Warning: the param {k} in {self.name} has been overwritten by other configs.')
            _k = copy.deepcopy(k)
            self._params.add(_k)
            setattr(self,_k,copy.deepcopy(getattr(cfg,k)))

    def sign(self,*param_names):
        assert param_names
        sgn_str = ''
        params_sgn = []
        for param_name in param_names:
            assert param_name in self._params
            params_sgn.append(param_name)
        self._params_sgn = params_sgn

    def __call__(self,short=True, *args, **kwargs):
        sgn_str = f'{self.name}@'
        for param_name in self._params_sgn:
            val = getattr(self,param_name)
            if '_' in param_name:
                param_name = ''.join([ele[0] for ele in param_name.split('_')])
            sgn_str += f'{param_name}_{val}-'
        sgn_str = sgn_str[:-1]
        return sgn_str

    def __str__(self):
        return f'{self.name}:'+'{' + '|'.join([f'{k}={getattr(self,k)}' for k in self.params]) + '}'

if __name__ == '__main__':
    cfg1 = Config(name='cfg1', a='a1', b='b1')
    cfg2 = Config(name='cfg2',a2='a2',b='b2')
    print(cfg1)
    print(cfg2)
    print(cfg2.params)
    cfg2.param = ('c','c2')
    print(cfg2)
    print(cfg2.params)
    cfg1.join(cfg2)
    print(cfg1)
    cfg1.a = 100
    print(cfg1)
    cfg1.sign('a2','b')
    cfg1.param = ('wo_shi_hoe_wang',10086)
    cfg1.sign('wo_shi_hoe_wang')
    print(cfg1())
    print(cfg1(short=False))



