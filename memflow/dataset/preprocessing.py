import torch

def logmodulus(X):
    return torch.sign(X) * torch.log(1+torch.abs(X))

class PreprocessingPipeline:
    """
        Pipeline class that applies several steps of preprocessing
        Can be useful to rescale some variables, before applying a more global rescaling
    """
    def __init__(self,steps):
        """
            Steps : PreprocessingStep instance list
        """
        self.steps = steps
        # Do some type checks #
        if not isinstance(self.steps,(list,tuple)):
            raise RuntimeError(f'Expects steps to be a list/tuple of PreprocessingStep instances, got {type(self.steps)}')
        for i,step in enumerate(self.steps):
            if not isinstance(step,PreprocessingStep):
                raise RuntimeError(f'Step {i} is {type(step)}, and not PreprocessingStep instance')

    def __call__(self,x,mask,fields):
        assert x.shape[-1] == len(fields), f'Mismatch between shape {x.shape} and number of fields {len(fields)}'
        for step in self.steps:
            x = step(x,mask,fields)
        return x

class PreprocessingStep:
    """
        Class that applies a function to some variable as determined by the function_dict
    """
    def __init__(self,function_dict):
        """
            Args :
              - function_dict [dict] : dict with variable name as keys, and functions as values

            Example:
            ```
                function_dict = {'pt': logmodulus}
            ```

            Can also use the scikit-learn preprocessing:
            ```
                from sklearn.preprocessing import scale
                function_dict = {
                    'pt' : scale,
                    'eta' : scale,
                    'phi' : scale,
                    'mass' : scale,
                }
            ```
            in case of other arguments to provide, can use a lambda
            ```
                from sklearn.preprocessing import power_transform
                function_dict = {
                    'pt' : lambda x : power_transform(x,method='yeo-johnson'),
                    [...]
                }
            ```
        """
        self.function_dict = function_dict

    def __call__(self,x,mask,fields):
        """
            Process a tensor x with the preprocessing function (needs to be provided the fields attached to each dimension), and the mask
            Args:
             - x [torch.tensor] : tensor with size [N,P,H]
             - mask [torch.tensor] : tensor with size [N,P,1]
             - fields [list] : list of feature names (size=H)
            (N = events, P = particles, H = features)
        """
        assert len(fields) == x.shape[2], f'Feature size is {x.shape[2]}, but received {fields} feature names'
        for i,field in enumerate(fields): # loop over features (eg pt, eta, ... pdgid,...)
            if field in self.function_dict.keys():
                for j in range(x.shape[1]): # loop over particles
                    # Need to use a loop here, because application of the mask
                    # linearizes the 2D tensor
                    x_feat = self.function_dict[field](x[:,j,i][mask[:,j]].unsqueeze(-1))
                    if not torch.is_tensor(x_feat): # sklearn functions return np array
                        x_feat = torch.tensor(x_feat)
                    if x_feat.dtype != x.dtype:
                        x_feat = x_feat.to(x.dtype)
                    x[:,j,i][mask[:,j]] = x_feat.squeeze()
        return x
