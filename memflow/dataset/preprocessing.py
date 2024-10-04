import numpy as np
import torch
import itertools
from copy import deepcopy
from abc import ABCMeta,abstractmethod
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

class AbsScaler(metaclass=ABCMeta):
    @abstractmethod
    def transform(self,x):
        pass

    @abstractmethod
    def inverse(self,x):
        pass

    def fit(self,x):
        pass

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

class logmodulus(AbsScaler):
    def transform(self,x):
        return torch.sign(x) * torch.log(1+torch.abs(x))

    def inverse(self,x):
        return torch.sign(x) * (torch.exp((x/torch.sign(x))) - 1)


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
            y = torch.tensor(y)
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
            y = torch.tensor(y)
        if x.dtype != y.dtype:
            y = y.to(x.dtype)
        return y

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
                xs[i] = step.transform(names[i],xs[i],masks[i],fields[i])

    def transform(self,name,x,mask,fields):
        assert x.shape[-1] == len(fields), f'Mismatch between shape {x.shape} and number of fields {len(fields)}'
        for step in self.steps:
            if step.applies(name):
                x = step.transform(name,x,mask,fields)
        return x

    def inverse(self,name,x,mask,fields):
        assert x.shape[-1] == len(fields), f'Mismatch between shape {x.shape} and number of fields {len(fields)}'
        for step in reversed(self.steps):
            if step.applies(name):
                x = step.inverse(name,x,mask,fields)
        return x

    def is_processed(self,field):
        for names,step in self.steps:
            if field in step.scaler_dict.keys():
                return True
        return False

    def __str__(self):
        return "\nPreprocessing steps\n" + "".join([str(step) for step in self.steps])


class PreprocessingStep:
    """
    Class that applies a scaler to some variable as determined by the scaler_dict
    """
    def __init__(self,names,scaler_dict,fields_select=None): # TODO : update docs
        """
        Args :
              - scaler_dict [dict] : dict with variable name as keys, and scalers as values

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
        """
        # Attributes #
        self.names = names
        self.scaler_dict = scaler_dict
        self.fields_select = fields_select
        # Safety checks #
        if self.fields_select is not None:
            if len(self.fields_select) != len(self.names):
                raise RuntimeError(f'Got {len(self.names)} objects but {len(self.fields_select)} set of fields')
            for fields in self.fields_select:
                if not isinstance(fields,(list,tuple)):
                    fields = tuple(fields)
                if len(set(fields)-set(self.keys())) > 0:
                    raise RuntimeError(f'Selecting fields that are not in the scaler dict {[f for f in fields if f not in self.keys()]}')
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

        # Need to use inner loops here, because application of the mask linearizes the 2D tensor
        # Loop over particles #
        for j in range(x.shape[1]):
            # Skip if none of the particles exist #
            if mask[:,j].sum() == 0:
                continue
            # Loop over features #
            for i,field in enumerate(fields):
                if field in fields_select and field in self.scaler_dict.keys():
                    scaling = getattr(self.scaler_dict[field],direction)
                    x[:,j,i][mask[:,j]] = scaling(x[:,j,i][mask[:,j]].unsqueeze(-1)).squeeze(-1)
        return x

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
