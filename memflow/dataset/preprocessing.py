import os
import yaml
import numpy as np
import torch
import joblib
import itertools
from copy import deepcopy
from abc import ABCMeta,abstractmethod
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

import memflow
from memflow.dataset.utils import recursive_tuple_to_list


class AbsScaler(metaclass=ABCMeta):
    @abstractmethod
    def transform(self,x):
        pass

    @abstractmethod
    def inverse(self,x):
        pass

    def fit(self,x):
        pass

    def save(self,f):
        return None

    @classmethod
    def load(cls,cfg):
        return None


class lowercutshift(AbsScaler):
    def __init__(self,lower_cut=None):
        self.lower_cut = lower_cut

    def transform(self,x):
        if self.lower_cut is None:
            self.lower_cut = x.min()
        return x - self.lower_cut

    def inverse(self,x):
        if self.lower_cut is None:
            raise RuntimeError('Lower cut has not been defined')
        return x + self.lower_cut

    def save(self,f):
        return {'lower_cut':self.lower_cut}

    @classmethod
    def load(cls,cfg):
        return cls(lower_cut=cfg['lower_cut'])


class logmodulus(AbsScaler):
    def transform(self,x):
        return torch.sign(x) * torch.log(1+torch.abs(x))

    def inverse(self,x):
        return torch.sign(x) * (torch.exp((x/torch.sign(x))) - 1)

    def save(self,f):
        return {}

    @classmethod
    def load(cls,cfg):
        return cls()



class SklearnScaler(AbsScaler):
    def __init__(self,obj):
        assert isinstance(obj,BaseEstimator), f'{obj} not from scikit-learn'
        self.obj = obj

    def fit(self,x):
        try:
            check_is_fitted(self.obj)
            raise RuntimeError('Scaler has already been fitted, this should not be called')
        except NotFittedError:
            self.obj.fit(x)

    def transform(self,x):
        # Check if fitted already #
        try:
            check_is_fitted(self.obj)
            y = self.obj.transform(x)
        except NotFittedError:
            raise RuntimeError(f'Scaler {self.obj} has not been fitted yet')
        # Sklearn produces np arrays #
        if isinstance(y,np.ndarray):
            if y.dtype == 'O': # typically when containing None
                y = y.astype(np.float32)
            y = torch.from_numpy(y)
        if x.dtype != y.dtype:
            y = y.to(x.dtype)
        return y

    def inverse(self,x):
        try:
            check_is_fitted(self.obj)
            y = self.obj.inverse_transform(x)
        except NotFittedError:
            raise RuntimeError('Scaler has not been fitted yet')
        # Sklearn produces np arrays #
        if isinstance(y,np.ndarray):
            if y.dtype == 'O': # typically when containing None
                y = y.astype(np.float32)
            y = torch.from_numpy(y)
        if x.dtype != y.dtype:
            y = y.to(x.dtype)
        return y

    def save(self,f):
        joblib.dump(self.obj,f)
        return {'obj':f}

    @classmethod
    def load(cls,cfg):
        obj = joblib.load(cfg['obj'])
        return cls(obj=obj)



class PreprocessingPipeline:
    """
        Pipeline class that applies several steps of preprocessing
    """
    def __init__(self):
        """
            Steps : PreprocessingStep instance list
        """
        self.steps = []

    def add_step(self,step):
        assert isinstance(step,PreprocessingStep)
        self.steps.append(step)

    def fit(self,names,xs,masks,fields):
        assert len(names) == len(xs)
        assert len(names) == len(masks)
        assert len(names) == len(fields)
        for step in self.steps:
            # Find object indices that are considered in the preprocessing step #
            indices = [i for i,name in enumerate(names) if step.applies(name)]
            if len(indices) == 0:
                print (f'Skipping step {step} (applied on {step.names}) but got only {names}')
                continue
            # Fit the scaler #
            step.fit(
                names = [names[i] for i in indices],
                xs = [xs[i] for i in indices],
                masks = [masks[i] for i in indices],
                fields = [fields[i] for i in indices],
            )
            # Apply the transform that was just fitted to get the input of the next step #
            for i in indices:
                xs[i],fields[i] = step.transform(names[i],xs[i],masks[i],fields[i])

    def transform(self,name,x,mask,fields):
        assert x.shape[-1] == len(fields), f'Mismatch between shape {x.shape} and number of fields {len(fields)}'
        for step in self.steps:
            if step.applies(name):
                x,fields = step.transform(name,x,mask,fields)
        return x,fields

    def inverse(self,name,x,mask,fields):
        assert x.shape[-1] == len(fields), f'Mismatch between shape {x.shape} and number of fields {len(fields)}'
        for step in reversed(self.steps):
            if step.applies(name):
                x,fields = step.inverse(name,x,mask,fields)
        return x,fields

    def is_processed(self,field):
        for names,step in self.steps:
            if field in step.scaler_dict.keys():
                return True
        return False

    def __str__(self):
        return "\nPreprocessing steps\n" + "".join([str(step) for step in self.steps])

    def save(self,outdir):
        if os.path.exists(outdir):
            print (f'Will overwrite what is in output directory {outdir}')
        else:
            os.makedirs(outdir)

        configs = []
        for i,step in enumerate(self.steps):
            step_cfg = step.save(os.path.join(outdir,f'step_{i}'))
            configs.append(step_cfg)

        with open(os.path.join(outdir,'preprocessing_config.yml'),'w') as handle:
            yaml.dump(configs,handle)
        print (f'Preprocessing saved in {outdir}')

    @classmethod
    def load(cls,outdir):
        if not os.path.exists(outdir):
            raise RuntimeError(f'Could not find directory {outdir}')
        config_path = os.path.join(outdir,'preprocessing_config.yml')
        if not os.path.exists(config_path):
            raise RuntimeError(f'Could not find config {config_path}')

        with open(config_path,'r') as handle:
            config = yaml.safe_load(handle)

        inst = cls()
        for step_cfg in config:
            inst.add_step(PreprocessingStep.load(step_cfg,outdir))

        return inst


class PreprocessingStep:
    """
    Class that applies a scaler to some variable as determined by the scaler_dict
    """
    def __init__(self,names,scaler_dict,fields_select=None):
        """
        Args :
              - names [str/list] : list of names for particle types to use
                Need those to keep track of multiple objects during processing in the pipeline
              - scaler_dict [dict] : dict with variable name as keys, and scalers as values
              - fields_select [list[list]/None] : list of features for each particle to restrict transform
            Example:
            ```
                scaler_dict = {'pt': logmodulus}
            ```

            Can also use the scikit-learn preprocessing:
            ```
                from sklearn.preprocessing import scale
                scaler_dict = {
                    'pt' : scale,
                    'eta' : scale,
                    'phi' : scale,
                    'mass' : scale,
                }
            ```
            in case of other arguments to provide, can use a lambda
            ```
                from sklearn.preprocessing import power_transform
                scaler_dict = {
                    'pt' : lambda x : power_transform(x,method='yeo-johnson'),
                    [...]
                }
            ```
            If provided, the fields_select must b a list with for each particle type in names,
            tells which feature to actually include.
            This is useful when some particles do not have the feature in question (eg eta in the MET)
            Example:
            ```
                names = ['jets','met'],
                scaler_dict = {'pt':<scaler>,'eta':<scaler>},
                fields_select = [['pt','eta'],['pt']]
            ```
            In this example, the pt is scaler together for jets and met, but eta is only included for jets

        """
        # Attributes #
        if isinstance(names,str):
            self.names = [names]
        elif isinstance(names,(list,tuple)):
            self.names = names
        else:
            raise TypeError
        self.names = names
        self.scaler_dict = scaler_dict
        assert isinstance(self.scaler_dict,dict)
        self.fields_select = fields_select
        # Safety checks #
        if self.fields_select is not None:
            assert isinstance(self.fields_select,(list,tuple)), f'{type(self.fields_select)}'
            if len(self.fields_select) != len(self.names):
                raise RuntimeError(f'Got {len(self.names)} objects but {len(self.fields_select)} set of fields')
            for fields in self.fields_select:
                if not isinstance(fields,(list,tuple)):
                    fields = tuple(fields)
                if len(set(fields)-set(self.keys())) > 0:
                    raise RuntimeError(f'Selecting fields ({fields}) that are not in the scaler dict ({self.keys()}): {[f for f in fields if f not in self.keys()]}')
        else:
            self.fields_select = [tuple(self.scaler_dict.keys()) for _ in range(len(self.names))]
        for key,val in self.scaler_dict.items():
            if not isinstance(val,AbsScaler):
                raise RuntimeError(f'Scaler for key `{key}` is not AbsScaler, got `{type(val)}` instead')

    def applies(self,name):
        return name in self.names

    def keys(self):
        return self.scaler_dict.keys()

    def fit(self,names,xs,masks,fields):
        # Loop over all fields of all objects #
        for field in list(set(itertools.chain.from_iterable(fields))):
            if field in self.scaler_dict.keys():
                x = []
                for j in range(len(xs)):
                    if field not in fields[j]:
                        continue
                    if names[j] not in self.names:
                        continue
                    if field not in self.fields_select[self.names.index(names[j])]:
                        continue
                    x.append(xs[j][:,:,fields[j].index(field)][masks[j]].unsqueeze(1))
                x = torch.cat(x,dim=0)
                self.scaler_dict[field].fit(x)


    def _process(self,name,x,mask,fields,direction):
        # Safety checks #
        assert direction in ['transform','inverse']
        assert len(fields) == x.shape[2], f'Feature size is {x.shape[2]}, but received {fields} feature names'
        if name not in self.names:
            return x

        x = x.clone() # avoid reference issues
        if mask.dtype != torch.bool:
            mask = mask > 0

        fields_select = self.fields_select[self.names.index(name)]
        unique_fields = list(dict.fromkeys(fields))
        field_indices = {
            field : [i for i in range(len(fields)) if fields[i]==field]
            for field in unique_fields
        }
        # Need to use inner loops here, because application of the mask linearizes the 2D tensor
        # Will also split the input by feature and concat later
        # This is because some preprocessing (eg onehot) will change the feature dimension
        # Loop over features #
        xfs = []
        proc_fields = []
        for field in unique_fields:
            indices = field_indices[field]
            xf = x[...,indices]
            proc_field = [field for _ in range(len(indices))]
            if field in fields_select:
                # Get scaling
                scaling = getattr(self.scaler_dict[field],direction)
                # Loop over particles and mask
                xps = []
                for i in range(xf.shape[1]):
                    xp = xf[:,i,:]
                    mp = mask[:,i]
                    xs = scaling(xp)
                    # Treat cases where feature dimension changes #
                    if xs.shape[-1] > xp.shape[-1]:
                        # operation increases the number of features (eg onehot encoding)
                        assert xp.shape[-1] == 1
                        # First repeat then replace unmasked
                        xp = xp.repeat_interleave(xs.shape[-1],dim=-1)
                        # repeat the field to keep track
                        proc_field = [field for _ in range(xs.shape[-1])]
                    if xs.shape[-1] < xp.shape[-1]:
                        assert xs.shape[-1] == 1
                        # first restrict xp (we had repeated it so it's fine)
                        xp = xp[:,:1]
                        proc_field = [proc_field[0]]
                    # Change the values but only unmasked particles #
                    xp[mp] = xs[mp]
                    # Record #
                    xps.append(xp.unsqueeze(1))
                # Combine all the particles
                xf = torch.cat(xps,dim=1)
            # Record feature array #
            xfs.append(xf)
            proc_fields.extend(proc_field)
        # Combine all the features #
        x = torch.cat(xfs,dim=-1)
        return x,proc_fields

    def transform(self,name,x,mask,fields):
        """
            Process a tensor x with the preprocessing scaler (needs to be provided the fields attached to each dimension), and the mask
            Args:
             - x [torch.tensor] : tensor with size [N,P,H]
             - mask [torch.tensor] : tensor with size [N,P,1]
             - fields [list] : list of feature names (size=H)
            (N = events, P = particles, H = features)
        """
        return self._process(name,x,mask,fields,'transform')

    def inverse(self,name,x,mask,fields):
        return self._process(name,x,mask,fields,'inverse')

    def __str__(self):
        s = f"Step applied to {self.names}\n"
        max_len = max([len(field) for field in self.scaler_dict.keys()])
        for field,scaler in self.scaler_dict.items():
            s += f'\t{field:{max_len+1}s} : {scaler.__class__}\n'
        max_len = max([len(name) for name in self.names])
        for name,fields in zip(self.names,self.fields_select):
            s += f'  - {name:{max_len+1}s}: {fields}\n'
        return s

    def save(self,subdir):
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        # Save the scaler_dict instances content #
        scaler_dict_cfg = {}
        for feature,instance in self.scaler_dict.items():
            instance_cfg = instance.save(os.path.join(subdir,f'{feature}.bin'))
            instance_cls = instance.__class__.__name__
            scaler_dict_cfg[feature] = {
                'class' : instance_cls,
                **instance_cfg
            }

        # Return config #
        return {
            'names': self.names,
            'scaler_dict': scaler_dict_cfg,
            'fields_select' : recursive_tuple_to_list(self.fields_select),
        }

    @classmethod
    def load(cls,config,subdir):
        scaler_dict = {}
        for feature,cfg in config['scaler_dict'].items():
            feature_cls = getattr(memflow.dataset.preprocessing,cfg.pop('class'))
                # need to find a cleaner way with importlib
            scaler_dict[feature] = feature_cls.load(cfg)

        return cls(
            names = config['names'],
            scaler_dict = scaler_dict,
            fields_select = config['fields_select'],
        )


