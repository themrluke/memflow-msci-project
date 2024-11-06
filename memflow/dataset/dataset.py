import os
import copy
import glob
import awkward as ak
import torch
import numpy as np
import vector
import matplotlib.pyplot as plt
from itertools import chain
from operator import itemgetter
from functools import reduce
from hepunits.units import MeV, GeV
from abc import ABCMeta,abstractmethod
from particle import Particle


from torch.utils.data import Dataset

from memflow.read_data import utils
from memflow.dataset.data import AbsData, get_intersection_indices
from memflow.dataset.preprocessing import PreprocessingPipeline, PreprocessingStep
from memflow.phasespace.phasespace import PhaseSpace


class AcceptanceDataset(Dataset):
    """
        Dataset to train the efficiency on hard events, knowing whether they have been selected in the reco dataset
    """
    def __init__(self,hard_dataset,reco_data,intersection_branch=None):
        self.hard_dataset = hard_dataset
        self.reco_data = reco_data
        self.intersection_branch = intersection_branch

        assert isinstance(self.hard_dataset,HardDataset)
        assert isinstance(self.reco_data,AbsData)

        # Obtain the hard indices that are mathed to reco #
        if self.intersection_branch is None:
            pass
            ## Assume the trees are the same for both reco and hard,
            ## will just check their length
            #if self.hard_dataset.data is not self.reco_dataset.data:
            #    raise RuntimeError('Not the same `data` for reco and hard datasets, you should use `intersection_branch` to help resolve ambiguities')
            #assert len(self.hard_dataset) == len(self.reco_dataset), f'Different number of entries between hard ({len(self.hard_dataset)}) compared to reco ({len(self.reco_dataset)})'
            #self.N = len(self.hard_dataset)
        else:
            self.selected_idx, _ = get_intersection_indices(
                datas = [self.hard_dataset.data,self.reco_data],
                branch = self.intersection_branch,
            )
            # selected_idx are the hard events that are reconstructed in the reco dataset -> they passed the selections

        # Get input #
        self.inputs = [
                self.hard_dataset._objects[name][0]
                for name in self.hard_dataset.selection
        ]
        shapes = [inp[:,0,:].shape for inp in self.inputs] # check the #events and #features between inputs
        if len(set(shapes)) > 1:
            raise RuntimeError('Mismatch in shapes : '+', '.join([f'{name} = {shape}' for name,shape in zip(self.hard_dataset.selection,shapes)]))
        self.inputs = torch.cat(self.inputs,dim=1)

        # Make target #
        self.targets = torch.zeros((len(self),1))
        self.targets[self.selected_idx] = 1
        #self.targets = self.targets.to(torch.long)

    def __len__(self):
        return len(self.hard_dataset)

    def __getitem__(self, idx):
        """ Returns the hard-level variables and targets """
        return self.inputs[idx],self.targets[idx]

    @property
    def number_objects(self):
        return self.inputs.shape[1]

    @property
    def dim_features(self):
        return self.inputs.shape[2]

class MultiplicityDataset(Dataset):
    """
        Dataset to train the reco jet multiplicity on hard events
    """
    def __init__(self,hard_dataset,reco_data,intersection_branch,use_weights=False):
        self.hard_dataset = hard_dataset
        self.reco_data = reco_data
        self.intersection_branch = intersection_branch
        self.use_weights = use_weights

        assert isinstance(self.hard_dataset,HardDataset)
        assert isinstance(self.reco_data,AbsData)

        # Get indices for hard and reco #
        self.selected_idx, self.reco_idx = get_intersection_indices(
            datas = [self.hard_dataset.data,self.reco_data],
            branch = self.intersection_branch,
        )
        # selected_idx are the hard events that are reconstructed in the reco dataset -> they passed the selections

        # Get number of jets #
        jet_mask = torch.cat(
            [
                torch.tensor((self.reco_data[f'j{i}_E'] > 0).to_numpy()).unsqueeze(1)
                for i in range(1,10)
            ]
            ,
            dim = 1,
        )[self.reco_idx,...]
        N_jets = jet_mask.sum(dim=1)

        # Make inputs #
        # Get input #
        self.inputs = [
                self.hard_dataset._objects[name][0][self.selected_idx]
                for name in self.hard_dataset.selection
        ]
        shapes = [inp[:,0,:].shape for inp in self.inputs] # check the #events and #features between inputs
        if len(set(shapes)) > 1:
            raise RuntimeError('Mismatch in shapes : '+', '.join([f'{name} = {shape}' for name,shape in zip(self.hard_dataset.selection,shapes)]))
        self.inputs = torch.cat(self.inputs,dim=1)

        # Make target #
        self.targets = torch.nn.functional.one_hot(N_jets).to(torch.float32)

        # Make weights #
        if self.use_weights:
            # Generate weight per multiplicity #
            w = torch.nan_to_num(
                input = 1/torch.sum(self.targets,dim=0),
                nan = 0.,
                posinf = 0.,
                neginf = 0.,
            )
            w = torch.arange(0,self.targets.shape[1],1).to(torch.float32)**5
            print (w)
            # Repeat for each mult then select the corresponding one #
            w = w.unsqueeze(0).repeat_interleave(len(self),dim=0)
            self.weights = w[self.targets>0]
            # Rescale to sum of events #
            self.weights *= len(self) / self.weights.sum()
        else:
            self.weights = torch.ones((len(self)))

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        """ Returns the hard-level variables and targets """
        return self.inputs[idx],self.targets[idx], self.weights[idx]

    @property
    def max_length(self):
        return self.targets.shape[1]

    @property
    def number_objects(self):
        return self.inputs.shape[1]

    @property
    def dim_features(self):
        return self.inputs.shape[2]

class MultiCombinedDataset(Dataset):
    """
        Dataset that combines several combined datasets, typically for different processes
    """
    def __init__(self,datasets):
        # Check dataset #
        for dataset in datasets:
            assert isinstance(dataset,CombinedDataset)
        assert len(dataset)>1, 'Need to provide at least two datasets'
        self.datasets = datasets

        self.hard_datasets = [dataset.hard_dataset for dataset in self.datasets]
        self.reco_datasets = [dataset.reco_dataset for dataset in self.datasets]

        # Various sanity checks #
        self.check_features(self.hard_datasets)
        self.check_features(self.reco_datasets)

        # Equalize datasets #
        self.equalize_datasets(self.hard_datasets)
        self.equalize_datasets(self.reco_datasets)

        # Check attention masks #
        self.check_attention_masks(self.hard_datasets)
        self.check_attention_masks(self.reco_datasets)

        # Get attributes needed to find index per dataset #
        self.Ns = torch.tensor([len(dataset) for dataset in self.datasets])
        self.cumNs = torch.cumsum(self.Ns,dim=0)
        self.idx_dataset_start = self.cumNs-self.Ns

    @staticmethod
    def check_features(datasets):
        inputs_features = set([dataset.input_features for dataset in datasets])
        if len(inputs_features) != 1:
            raise RuntimeError(f'Different sets of inputs features : {inputs_features}')
        n_types = set([len(dataset.number_particles_per_type) for dataset in datasets])
        if len(n_types) != 1:
            raise RuntimeError(f'Different number of types : {n_types}')

    @staticmethod
    def check_attention_masks(datasets):
        mask_shapes = set([dataset.attention_mask.shape for dataset in datasets])
        if len(mask_shapes) != 1:
            raise RuntimeError(f'Attention masks have different shapes : {mask_shapes}')
        attention_masks = set([dataset.attention_mask for dataset in datasets])
        for i in range(1,len(datasets)):
            if not (datasets[0].attention_mask == datasets[i].attention_mask).all():
                raise RuntimeError(f'Different attention masks between entry 0 ({datasets[0].attention_mask}) and {i} ({datasets[i].attention_mask})')

    @staticmethod
    def check_datasets(datasets):
        inputs_features = set([dataset.input_features for dataset in datasets])
        if len(inputs_features) != 1:
            raise RuntimeError(f'Different sets of inputs features : {inputs_features}')
        n_types = set([len(dataset.number_particles_per_type) for dataset in datasets])
        if len(n_types) != 1:
            raise RuntimeError(f'Different number of types : {n_types}')


    @staticmethod
    def equalize_datasets(datasets):
        # calculate the max number of particles per type #
        n_types = len(datasets[0].number_particles_per_type)
        max_number_particles_per_type = [
            max(
                [
                    dataset.number_particles_per_type[i] for dataset in datasets
                ]
            )
            for i in range(n_types)
        ]
        # For each type, pad the tensors #
        for dataset in datasets:
            for i in range(n_types):
                if dataset.number_particles_per_type[i] < max_number_particles_per_type[i]:
                    # Get number of additional particles to pad and data #
                    n_diff = max_number_particles_per_type[i]-dataset.number_particles_per_type[i]
                    name = dataset.selection[i]
                    data, mask, weights = dataset._objects[name]
                    # Zero-pad the inputs data #
                    data = torch.cat(
                        [
                            data,
                            torch.zeros(
                                (
                                    data.shape[0],    # N batch
                                    n_diff,           # S sequence
                                    data.shape[2],    # F features
                                 )
                            ),
                        ],
                        dim = 1,
                    )
                    # Pad the mask with Falses #
                    mask = torch.cat(
                        [
                            mask,
                            torch.full(
                                (
                                    mask.shape[0],    # N batch
                                    n_diff,           # S sequence
                                 ),
                                fill_value = False,
                            ),
                        ],
                        dim = 1,
                    )
                    # Pad the weights with ones #
                    weights = torch.cat(
                        [
                            weights,
                            torch.ones(
                                (
                                    weights.shape[0], # N batch
                                    n_diff,           # S sequence
                                 ),
                            ),
                        ],
                        dim = 1,
                    )
                    # Replace the object #
                    dataset._objects[name] = data, mask, weights
                if dataset.number_particles_per_type[i] > max_number_particles_per_type[i]:
                    raise RuntimeError


    def __getitem__(self, idx):
        # Find which dataset to take the index from #
        idx_of_dataset = torch.searchsorted((self.cumNs-1),idx)
        # Find the index within the dataset #
        idx_in_dataset = int(idx - self.idx_dataset_start[idx_of_dataset])
        # return the corresponding item #
        return self.datasets[idx_of_dataset][idx_in_dataset]

    def __len__(self):
        return self.cumNs[-1]

    def __str__(self):
        return f'MultiCombined dataset :\n'+'\n'.join([str(dataset) for dataset in self.datasets])

class CombinedDataset(Dataset):
    """
        Dataset to combine hard and reco and provide events with both information
    """
    def __init__(self,hard_dataset,reco_dataset,intersection_branch=None):
        """
            Args:
             - hard_dataset [HardDataset] : hard scattering level dataset
             - reco_dataset [RecoDataset] : reconstructed level dataset
             - intersection_branch [str] : in case the data object are different, use this branch to resolve ambiguity (eg event number branch)
        """
        self.hard_dataset = hard_dataset
        self.reco_dataset = reco_dataset
        self.intersection_branch = intersection_branch

        assert isinstance(self.hard_dataset,HardDataset)
        assert isinstance(self.reco_dataset,RecoDataset)

        if self.intersection_branch is None:
            # Assume the trees are the same for both reco and hard,
            # will just check their length
            if self.hard_dataset.data is not self.reco_dataset.data:
                raise RuntimeError('Not the same `data` for reco and hard datasets, you should use `intersection_branch` to help resolve ambiguities')
            assert len(self.hard_dataset) == len(self.reco_dataset), f'Different number of entries between hard ({len(self.hard_dataset)}) compared to reco ({len(self.reco_dataset)})'
            self.N = len(self.hard_dataset)
        else:
            self.hard_idx, self.reco_idx = get_intersection_indices(
                datas = [self.hard_dataset.data,self.reco_dataset.data],
                branch = self.intersection_branch,
            )
            assert len(self.hard_idx) == len(self.reco_idx)
            self.N = len(self.hard_idx)

    def __getitem__(self, index):
        """ Returns the event info of both hard and reco level variables """
        if self.intersection_branch is None:
            return {
                'hard'  : self.hard_dataset[index],
                'reco' : self.reco_dataset[index],
            }
        else:
            return {
                'hard'  : self.hard_dataset[self.hard_idx[index]],
                'reco' : self.reco_dataset[self.reco_idx[index]],
            }

    def batch_by_index(self,index):
        if self.intersection_branch is None:
            return {
                'hard'  : self.hard_dataset.batch_by_index(index),
                'reco' : self.reco_dataset.batch_by_index(index),
            }
        else:
            return {
                'hard'  : self.hard_dataset.batch_by_index(self.hard_idx[index]),
                'reco' : self.reco_dataset.batch_by_index(self.reco_idx[index]),
            }

    def find_indices(self,reco_masks=[],hard_masks=[]):
        """
            Find the indices in the combined dataset that correspond to masks in both reco and hard masks
        """
        # safety checks #
        for i,hard_mask in enumerate(hard_masks):
            assert len(hard_mask) == self.hard_dataset.data.events, f'Hard mass entry {i} has length {len(hard_mask)} but hard data object has {self.hard_dataset.data.events} events'
        for i,reco_mask in enumerate(reco_masks):
            assert len(reco_mask) == self.reco_dataset.data.events, f'Reco mass entry {i} has length {len(reco_mask)} but reco data object has {self.reco_dataset.data.events} events'

        # Make total masks #
        if len(hard_masks) > 0:
            hard_mask = np.logical_and.reduce(hard_masks)
        else:
            hard_mask = np.full((self.hard_dataset.data.events),fill_value=True)
        if len(reco_masks) > 0:
            reco_mask = np.logical_and.reduce(reco_masks)
        else:
            reco_mask = np.full((self.reco_dataset.data.events),fill_value=True)

        if self.intersection_branch is None:
            # hard and reco data are the same, just select the ones passing both masks #
            mask = np.logical_and(hard_mask,reco_mask)
            indices = np.arange(self.reco_dataset.data.events)[mask]
            return indices
        else:
            # different data objects, find the intersection between mask and common events #
            # Find in hard idx the events passing the hard cut
            mask_hard_idx = hard_mask[self.hard_idx]
            # Find in reco idx the events passing the reco cut
            mask_reco_idx = reco_mask[self.reco_idx]
            # Combine masks #
            mask = np.logical_and(mask_hard_idx,mask_reco_idx)
            indices = np.arange(len(self))[mask]
            return indices


    def __len__(self):
        return self.N

    def __str__(self):
        return f'Combined dataset (extracting {len(self)} events of the following) :\n{self.hard_dataset}\n{self.reco_dataset}'


class AbsDataset(Dataset,metaclass=ABCMeta):
    """
        Abstract class for torch dataset, inherited by the Hard and Reco datasets
    """
    cartesian_fields = set(['px','py','pz','E'])
    cylindrical_fields = set(['pt','eta','phi','mass'])

    def __init__(self,data,selection,default_features=None,build=False,device=None,dtype=None,**kwargs):
        """
            Initialises class, then perform the following steps
             - init (user class)
             - load or process (user class) + save : depending on presence of files and build arg
             - finalize (user class)
             - moving to device and change type
             - standardize : in case default_features arg is used, will fill missing fields in different objects
            Args:
             - data [AbsData] : data instance
             - selection [list] : list of objects (defined in the process method) to select for batching
             - default_features [dict/int/float] : features to standardize between all the objects
                    - int/float : replace all missing fields by value
                    - dict :
                        key : field name
                        val : value to put in case field is missing fo object
                              if None, field is removed from variables
            - build [bool] : whether to force saving tensors to file
            - device [str] : device for torch tensors
            - dtype [torch.type] : torch type for torch tensors
        """
        # Attributes #
        self.data = data
        self.build = build
        self.selection = selection
        self.default_features = default_features
        assert len(self.selection) > 0, 'You need to have at least some object selected'

        # Private attributes #
        self._objects = {}
        self._fields = {}
        self._preprocessing = PreprocessingPipeline()
        self._reserved_object_names = ['ps','detjinv']

        # Calling methods #
        self.init()
        if self.processed_path is not None and os.path.exists(self.processed_path) and not self.build:
            self._load()
        else:
            self.process()
            self._save()
        self.preprocessing()
        self.finalize()
        self.to_device(device)
        self.to_dtype(dtype)
        self.standardize()

        # Safety checks #
        for name in self.selection:
            if name not in self._objects.keys():
                raise RuntimeError(f'`{name}` not in registered objects {self._objects.keys()}')
        events = []
        for key,(data,mask,weights) in self._objects.items():
            if data.shape[0] != mask.shape[0]:
                raise RuntimeError(f'Object `{key}` has {data.shape[0]} data events but {mask.shape[0]} mask events')
            events.append(data.shape[0])
        events = list(set(events))
        if len(events) != 1:
            raise RuntimeError(f'Number of events mismatch {events} for objects {list(self._objects.keys())}')
        else:
            self.events = events[0]

    ##### Coordinate helpers #####
    def match_coordinates(self,arr_from,arr_to):
        """
            Modifies the fields of arr_from object, to the fields of the arr_to object
            Can be useful to uniformize the coordinates,
            eg if jets are in (pt,eta,phi,mass) and boost (px,py,pz,E) frames
        """
        dummy_vec = vector.obj(pt=0, phi=0, theta=0, mass=0)
        vec_attributes = [att for att in dir(dummy_vec) if not att.startswith('_')]
        for field in arr_to.fields:
            if field in vec_attributes:
                arr_from[field] = getattr(arr_from,field)
        for field in arr_from.fields:
            if field not in arr_to.fields:
                del arr_from[field]

    def cartesian_to_cylindrical(self,arr):
        """ Modifies coordinates cartesian->cylindrical """
        return self.change_variable(arr,self.cartesian_fields,self.cylindrical_fields)

    def cylindrical_to_cartesian(self,arr):
        """ Modifies coordinates cylindrical->cartesian """
        return self.change_variable(arr,self.cylindrical_fields,self.cartesian_fields)

    def change_variable(self,arr,set_from,set_to):
        """ Change variables from set_from to set_to """
        # Check the fields are there #
        assert len(set_from.intersection(set(arr.fields))) == len(set_from), f'Not all {set_from} in fields {arr.fields}'
        # Add new fields #
        for field in set_to:
            arr[field] = getattr(arr,field)
        for field in arr.fields:
            if field in set_to:
                continue
            # Remove old ones #
            elif field in set_from:
                del arr[field]
            # Put the old ones that are untouched by change of variable #
            else:
                arr[field] = getattr(arr,field)
        return arr

    ##### Main methods #####
    def object_to_tensor(self,obj,fields=None):
        if not isinstance(obj,ak.Array):
            raise RuntimeError(f'Expects an awkward array, got {type(obj)}')
        # From awkward to numpy #
        arr = utils.to_flat_numpy(
            obj,
            fields = fields,
            axis = 1,
            allow_missing = False,
        )
        if arr.ndim == 3:
            arr = np.transpose(arr,(0,2,1))
        elif arr.ndim == 2:
            arr = np.expand_dims(arr,axis=1)
        else:
            raise RuntimeError(f'Array has dim {arr.ndim}')
        # numpy to torch #
        return torch.tensor(arr)

    def register_object(self,name,obj,mask=None,weights=None,fields=None):
        """
            Function called by user to register an awkward+vector array as a torch tensor
            Args:
             - name [str] : name of the object to register (eg, 'lepton', 'jets')
             - obj [ak.Array + vector] : array to convert
             - mask [ak.Array] : boolean array for missing objects (if None, will be filled with True)
             - fields [list] : which fields to extract from the object (if None, will take all the ones in the object)
             - preprocessing [PreprocessingPipeline] : preprocessing instance
            Returns : None
            Self : save object in self._objects and fields in in self._fields
        """
        # Avoid overwriting objects #
        if name in self._reserved_object_names:
            raise RuntimeError(f'Name `{name}` already reserved')
        if name in self._objects.keys():
            raise RuntimeError(f'Name `{name}` already present in objects')
        # Get fields and then data tensore #
        fields = fields if fields is not None else obj.fields
        data = self.object_to_tensor(obj,fields)
        # Compute and reshape mask #
        if mask is None:
            mask = torch.full(data.shape[:2],fill_value=True)
        else:
            mask = torch.tensor(np.array(mask))
        if mask.dim() == 1:
            mask = mask.unsqueeze(-1)
        # Process mask ##
        if weights is None:
            weights = torch.ones_like(mask)
        else:
            if not torch.is_tensor(weights):
                weights = torch.tensor(weights)
            if weights.dim() == 0:
                weights = torch.ones_like(mask) * weights
            elif weights.dim() == 1:
                weights = weights.reshape(-1,1)
            elif weights.dim() == 2:
                pass
            else:
                raise NotImplementedError(f'Dim {weights.dim()} of weights is not expected')
            if weights.shape != mask.shape:
                raise RuntimeError(f'Weights shape {weights.shape} of {name} objects differs from {mask.shape} object shape')

        # Record #
        self._objects[name] = (data,mask,weights)
        self._fields[name] = fields

    def register_preprocessing_step(self,preprocessing):
        assert isinstance(preprocessing,PreprocessingStep), f'Preprocessing must be a PreprocessingStep instance'
        assert len(set(preprocessing.keys())-set(self._objects.keys())) > 0, f'Object names {[key for key in preprocessing.keys() if key not in self._objects.keys()]} have not been registered yet'
        self._preprocessing.add_step(preprocessing)

    def preprocessing(self):
        # First fit the different steps in the pipeline #
        self._preprocessing.fit(
            names = list(self._objects.keys()),
            xs = [self._objects[name][0] for name in self._objects.keys()],
            masks = [self._objects[name][1] for name in self._objects.keys()],
            fields = [self._fields[name] for name in self._objects.keys()],
        )
        # Finally apply the preprocessing transform #
        for name in self._objects.keys():
            data, mask, weights = self._objects[name]
            data = self._preprocessing.transform(name,data,mask,self._fields[name])
            self._objects[name] = (data,mask,weights)

    @staticmethod
    def reshape(input,value,max_no=None):
        """
            Function called by user to reshape a variable length object with padding
            Args:
             - input [ak.Array] : input array (N,P,H)
             - value [ak.Array/int/float] : default value to fill missing entry
                    can be an awkward array with different values per field, or a float/it used for all the input fields
            - max_no [int] : maximum number of particles for padding, if not given takes the max value
            Returns :
            - padded+filled array [ak.Array]
            - associated mask (for missing particles) [ak.Array]
        """
        if isinstance(value,(float,int)):
            value = ak.zip(
                {
                    f: value for f in input.fields
                },
                with_name='Momentum4D',
            )
        elif isinstance(value,(ak.Array,vector.Vector)):
            pass
        else:
            raise NotImplementedError(f'Type {type(value)} not implemented')
        if max_no is None:
            max_no = ak.max(ak.num(input, axis=1))
        input_padded = ak.pad_none(
            array = input,
            target = max_no,
            axis = 1,
        )[:,:max_no]
        # Not using clip because it creates RegularType and not ListType
        # (not sure why it causes a bug though)
        mask = ~ak.is_none(input_padded,axis=1)
        input_filled = ak.fill_none(
            array = input_padded,
            value = value,
            axis = None,
        )
        return input_filled,mask

    @staticmethod
    def boost(obj,boost):
        """
            Lorentz boosts an object
            Args:
             - obj [ak.Array] : object to boost
             - boost [ak.Array] : boost object
            Returns :
             - boosted object [ak.Array]
        """
        if boost.layout.purelist_depth > 1:
            raise RuntimeError(f'Boost array depth is {boost.layout.purelist_depth}, expects one value per event')
        obj_boosted = obj.boost_p4(boost.neg3D)
        # Restore original fields #
        for field in obj.fields:
            obj_boosted[field] = getattr(obj_boosted,field)
        # Remove added fields not in original #
        for field in obj_boosted.fields:
            if field not in obj.fields:
                del obj_boosted[field]
        return obj_boosted

    @staticmethod
    def expand_tensor(data,fields,default_values):
        """
            Expand tensor with default values
            Args:
             - data [torch.tensor]: tensor to fill
             - fields [list] : list of fields associated to the axis=1 of the tensor
             - default_values [dict] : default values for missing values
                    key : field name
                    val : value to fill when missing
            Return :
             - output tensor [torch.tensor]
             - output fields [list]
        """
        out_tensor = []
        out_fields = []
        # Add default fields #
        for field,default_val in default_values.items():
            if default_val is None:
                continue
            if field in fields:
                idx = fields.index(field)
                out_tensor.append(data[:,:,idx].unsqueeze(-1))
                out_fields.append(field)
            else:
                out_tensor.append(
                    torch.full((data.shape[0],data.shape[1],1),fill_value=default_val)
                )
                out_fields.append(field)
        # Include the original fields that were not in the default #
        for field in fields:
            if field not in default_values.keys():
                idx = fields.index(field)
                out_tensor.append(data[:,:,idx].unsqueeze(-1))
                out_fields.append(field)
        # Return concat data and fields #
        return torch.cat(out_tensor,dim=-1),out_fields

    def standardize(self):
        """
            If default features asked, will fill the missing ones for each object
            Self : modifies the self._objects and self._fields
        """
        if self.default_features is None:
            return
        elif isinstance(self.default_features,(float,int)):
            # Get the comprehensive list of fields and complete with value #
            default_val = self.default_features
            all_fields = sorted(list(set(chain.from_iterable(self._fields.values()))))
            default_values = {f:default_val for f in all_fields}
        elif isinstance(self.default_features,dict):
            default_values = self.default_features
        else:
            raise NotImplementedError(f'Type {type(self.default_features)} of default_features not implemented')
        # Modify each object data #
        for name,(data,mask,weights) in self._objects.items():
            fields = self._fields[name]
            data,fields = self.expand_tensor(data,fields,default_values)
            self._objects[name] = (data,mask,weights)
            self._fields[name] = fields

    ##### Intrinsic properties #####
    @property
    def input_features(self):
        return tuple(
            tuple(self._fields[name])
            for name in self.selection
        )

    def number_particles(self,name):
        return self._objects[name][0].shape[1]

    @property
    def number_particles_per_type(self):
        return [self.number_particles(name) for name in self.selection]

    @property
    def number_particles_total(self):
        return sum(self.number_particles_per_type)

    def object_attention_mask(self,name):
        if self.attention_idx is not None and name in self.attention_idx.keys():
            return [
                True if i in self.attention_idx[name] else False
                for i in range(self.number_particles(name))
            ]
        else:
            return [True]*self.number_particles(name)

    @property
    def attention_mask(self):
        return torch.tensor(list(chain.from_iterable([self.object_attention_mask(name) for name in self.selection])))

    ##### Type and device helpers #####
    def to_device(self,device=None):
        """ Move all tensors to device """
        if device is None:
            return
        for name,(data,mask,weights) in self._objects.items():
            self._objects[name] = (data.to(device),mask.to(device),weights.to(device))

    def to_dtype(self,dtype=None):
        """ Modifies the type of all tensors """
        if dtype is None:
            return
        for name,(data,mask,weights) in self._objects.items():
            self._objects[name] = (data.to(dtype),mask.to(dtype),weights.to(dtype))

    ##### Save and load methods ######
    def _save(self):
        """ If user defines the property processed_path, will save each object to file """
        if self.processed_path is not None:
            if not os.path.exists(self.processed_path):
                os.makedirs(self.processed_path)
            for name,(data,mask,weights) in self._objects.items():
                torch.save(
                    (data,mask,weights),
                    os.path.join(self.processed_path,f'{name}.pt')
                )
        print (f'Saving objects to {self.processed_path}')

    def _load(self):
        """ If user defines the property processed_path, will load each object from file """
        if self.processed_path is None:
            raise RuntimeError('Not processed path defined, cannot load the data')
        print (f'Loading objects from {self.processed_path}')
        for f in glob.glob(os.path.join(self.processed_path,'*.pt')):
            name = os.path.basename(f).replace('.pt','')
            data,mask,weights = torch.load(f)
            self._objects[name] = (data,mask,weights)

    ##### Magic methods #####
    def __getitem__(self,idx):
        """
            For each object specified in the selections, returns the element from idx
            Joins the data and mask along new dimension (axis = 2)
        """
        return {
            'data': [self._objects[name][0][idx] for name in self.selection],
            'mask': [self._objects[name][1][idx] for name in self.selection],
            'weights': [self._objects[name][2][idx] for name in self.selection],
        }

    def __len__(self):
        """ Number of events """
        return self.events

    def __str__(self):
        """ Representation string to show all defined objects with some info """
        s = f'\nContaining the following tensors'
        names_len = max([len(name) for name in self._objects.keys()]) + 1
        for name,(data,mask,weights) in self._objects.items():
            props = mask.sum(axis=0) / mask.shape[0] * 100
            prop_str = ', '.join([f'{prop:3.2f}%' if prop>=0.01 else '<0.01%' for prop in props])
            w_sum = weights.sum(dim=0)
            w_str = ', '.join([f'{ws:.2f}' for ws in w_sum])
            s += f'\n{name:{names_len}s} : data ({list(data.shape)}), mask ({list(mask.shape)})'
            s += f'\n{" "*names_len}   Mask exist    : [{prop_str}]'
            s += f'\n{" "*names_len}   Mask corr     : {self.object_attention_mask(name)}'
            s += f'\n{" "*names_len}   Weights       : {w_str}'
            s += f'\n{" "*names_len}   Features      : {self._fields[name]}'
            s += f'\n{" "*names_len}   Selected for batches : {name in self.selection}'
        s += str(self._preprocessing)
        return s

    ##### User accessible helper methods #####
    def batch_by_index(self,idx):
        if not torch.is_tensor(idx):
            idx = torch.tensor(idx)
            if idx.dim() == 0:
                idx = idx.unsqueeze(-1)
        return {
            'data': [
                torch.index_select(
                    input = self._objects[name][0],
                    dim = 0,
                    index = idx,
                )
                for name in self.selection
            ],
            'mask': [
                torch.index_select(
                    input = self._objects[name][1],
                    dim = 0,
                    index = idx,
                )
                for name in self.selection
            ],
            'weights': [
                torch.index_select(
                    input = self._objects[name][2],
                    dim = 0,
                    index = idx,
                )
                for name in self.selection
            ],
        }

    def plot(self,raw=False,selection=False,weighted=False,log=False):
        names = self.selection if selection else self._objects.keys()
        for name in names:
            # Get objects for name #
            data,mask,weights = self._objects[name]
            fields = self._fields[name]
            # Some processing #
            if mask.dtype != torch.bool:
                mask = mask > 0
            if raw:
                data = self._preprocessing.inverse(name,data,mask,fields)
            # Generate figure #
            n_parts = data.shape[1]
            n_cols = data.shape[2]
            if n_parts > 1:
                fig,axs = plt.subplots(1,n_cols+1,figsize=(4.5*(n_cols+1),4))
            else:
                fig,axs = plt.subplots(1,n_cols,figsize=(4.5*n_cols,4))
            fig.suptitle(f'Objects : {name}')
            # Loop over features #
            for i in range(n_cols):
                bins = np.linspace(data[:,:,i].min(),data[:,:,i].max(),50)
                # Loop over particles #
                for j in range(n_parts):
                    if weighted:
                        axs[i].hist(data[:,j,i][mask[:,j]],weights=weights[:,j][mask[:,j]],bins=bins,histtype='step',linewidth=2)
                    else:
                        axs[i].hist(data[:,j,i][mask[:,j]],bins=bins,histtype='step',linewidth=2)
                axs[i].set_xlabel(fields[i])
                if log:
                    axs[i].set_yscale('log')
                    axs[i].set_ylim((0.1,None))
                else:
                    axs[i].set_ylim((0,None))
            # Add legend in last subplot #
            if n_parts > 1:
                axs[-1].axis('off')
                for j in range(n_parts):
                    axs[-1].plot([],[],label=f'Object {j}')
                axs[-1].legend(loc='center left')
            plt.show()


    ##### Abstract method to be implemented by user #####
    @property
    @abstractmethod
    def energy(self):
        """ Energy of CM, to be defined by user, should use hepunits.units """
        pass

    @property
    def attention_idx(self):
        """ """
        pass

    def init(self):
        """ User available hook """
        pass

    @abstractmethod
    def process(self):
        """ To be called by user to register all the different objects from the self.data object """
        pass

    def finalize(self):
        """ User available hook """
        pass

    @property
    def processed_path(self):
        """ If user wants to save tensor to file, should return the directory path """
        return None



class HardDataset(AbsDataset):
    """
        Specific dataset for gen/hard data
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def finalize(self):
        self.compute_PS_point()

    @property
    @abstractmethod
    def initial_states_pdgid(self):
        """ Property to be overriden by user, returning the list of the initial state pdgIds """
        pass

    @property
    @abstractmethod
    def final_states_pdgid(self):
        """ Property to be overriden by user, returning the list of the final state pdgIds """
        pass

    @property
    def final_states_object_name(self):
        """
            Names of the final state objets, as they are recorded in name arg of register_objects
            Used by compute_PS_point to extract the momentas to be fed to Rambo and then the ME
        """
        return None

    @property
    def final_states_particle(self):
        """ Returns list of particle objects from the pdgIds """
        return [Particle.from_pdgid(pdgid) for pdgid in self.final_states_pdgid]

    @property
    def final_states_mass(self):
        """ Returns list of particle masses from the particle list """
        # neutrinos have mass=None, need to feed a 0. #
        return [p.mass/GeV if p.mass is not None else 0. for p in self.final_states_particle]

    def make_boost(self,x1,x2):
        """
            From the x1 an x2 fraction of momentum of initial partons, computes the boost
            Args:
             - x1 [np.array/ak.Array] : fraction x1
             - x2 [np.array/ak.Array] : fraction x2
            Returns:
             - boost [ak.Array]
        """
        if not isinstance(x1, np.ndarray):
            x1 = x1.to_numpy()
        if not isinstance(x2, np.ndarray):
            x2 = x2.to_numpy()

        half_sqrts = self.energy / (2*GeV)
        pz = (x1-x2) * half_sqrts
        E  = (x1+x2) * half_sqrts
        zeros = np.zeros(pz.shape)

        boost = ak.Array(
            {
                "x": zeros,
                "y": zeros,
                "z": pz,
                "t": E,
            }
        )
        boost = ak.with_name(boost, name="Momentum4D")
        return boost

    def compute_PS_point(self):
        """
            Produces a PS point based on the ME information from the user
            - the initial and final pdgIds (provided in properties) + their masses (automatically extracted)
            - the momenta : by fetching in the objects the name from the final state names (from the property user)
            Will save the ps point and jacobian in the objects under names "ps" and "detjinv"
        """
        if self.final_states_object_name is None:
            print ('No final state recorded with `final_states_object_name` property, will not compute PS points')
            return
        masses = torch.Tensor(self.final_states_mass)
        names = self.final_states_object_name
        if len(masses) != len(names):
            raise RuntimeError(f'{len(names)} objects ({names}) and {len(self.final_states_pdgid)} pdgids ({final_states_pdgid})')

        momenta = []
        for name in names:
            if name not in self._objects.keys():
                raise RuntimeError(f'{name} not found in the objects recorded {self._objects.keys()}')
            momenta.append(self._objects[name][0])
        shapes = [mom.shape for mom in momenta]
        for axis in [0,2]:
            if len(set([shape[axis] for shape in shapes])) != 1:
                raise RuntimeError(f'Shapes mismatch {shapes} for axis {axis}')
        momenta = torch.cat(momenta,dim=1)

        if 'boost' not in self._objects.keys():
            raise RuntimeError(f'`boost` must have been recorded in the objects')
        boost = self._objects['boost'][0]

        phasespace = PhaseSpace(
            collider_energy = self.energy / GeV,
            initial_pdgs = self.initial_states_pdgid,
            final_pdgs = self.final_states_pdgid,
            final_masses = masses,
            dev = 'cpu',
        )

        x1 = (boost[:, 0, 0] + boost[:, 0, 3]) / (self.energy / GeV)
        x2 = (boost[:, 0, 0] - boost[:, 0, 3]) / (self.energy / GeV)
        ps, detjinv = phasespace.get_ps_from_momenta(
            momenta = momenta,
            x1 = x1,
            x2 = x2,
        )
        detjinv = detjinv.unsqueeze(-1)
        self._objects['ps'] = (
            ps.unsqueeze(dim=-1),
            torch.full(ps.shape,True)#.unsqueeze(-1),
        )
        self._fields['ps'] = [f'ps_{i}' for i in range(ps.shape[1])]
        self._objects['detjinv'] = (
            detjinv.unsqueeze(-1),
            torch.full(detjinv.shape,True)#.unsqueeze(-1),
        )
        self._fields['detjinv'] = ['det']

    def __str__(self):
        """ Specific hard representations + mother class one """
        s =  f'Parton dataset with {len(self)} events'
        s += f'\n Initial states pdgids : {self.initial_states_pdgid}'
        s += f'\n Final states pdgids   : {self.final_states_pdgid}'
        s += f'\n Final states masses   : {self.final_states_mass}'
        s += super().__str__()
        return s

class RecoDataset(AbsDataset):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def make_boost(*objs):
        """
            From multiple objects, provided as separate args, compute the boost
            Args:
             - objs [ak.Array] : different particles, can be single per event or variable (in which case, summed)
            Returns:
             - boost [ak.Array]
        """
        assert isinstance(objs,(tuple,list)), f'Expecting a list/tuple of awkward vector arrays'
        boost = None
        for i,obj in enumerate(objs):
            assert isinstance(obj,ak.Array) or isinstance(obj,vector.Vector), f'Object number {i} is {type(obj)}'
            if obj.layout.purelist_depth > 1:
                obj = ak.sum(obj,axis=1)
            if boost is None:
                boost = copy.deepcopy(obj)
            else:
                boost = boost + obj
        return boost

    def __str__(self):
        """ Specific reco representations + mother class one """
        s = f'Reco dataset with {len(self)} events'
        s += super().__str__()
        return s

