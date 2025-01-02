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

from memflow.dataset.utils import *
from memflow.dataset.preprocessing import *
from memflow.dataset.data import AbsData
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
                self.hard_dataset.objects[name][0]
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
                self.hard_dataset.objects[name][0][self.selected_idx]
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

class MultiHelperClass:
    def __init__(self,datasets):
        if isinstance(datasets,(tuple,list)):
            self.names = [str(i) for i in range(len(datasets))]
            self.datasets = datasets
        elif isinstance(datasets,dict):
            self.names = list(datasets.keys())
            self.datasets = list(datasets.values())
        else:
            raise RuntimeError(f'Datasets must be list or dict')

    def compute(self):
        # Get attributes needed to find index per dataset #
        self.Ns = torch.tensor([len(dataset) for dataset in self.datasets])
        self.cumNs = torch.cumsum(self.Ns,dim=0)
        self.idx_dataset_start = self.cumNs-self.Ns

    def __getitem__(self, idx):
        # Find which dataset to take the index from #
        idx_of_dataset = torch.searchsorted((self.cumNs-1),idx).item()
        # Find the index within the dataset #
        idx_in_dataset = int(idx - self.idx_dataset_start[idx_of_dataset])
        # return the corresponding item #
        return {
            'process' : idx_of_dataset,
            **self.datasets[idx_of_dataset][idx_in_dataset],
        }

    def __len__(self):
        return self.cumNs[-1]


class MultiCombinedDataset(MultiHelperClass,Dataset):
    """
        Dataset that combines several combined datasets, typically for different processes
    """
    def __init__(self,datasets):

        # Super #
        super().__init__(datasets)

        # Check dataset #
        for dataset in self.datasets:
            assert isinstance(dataset,CombinedDataset)
        assert len(self.datasets)>0, 'Need to provide at least one dataset'

        # plit between hard and reco #
        self.hard_datasets = [dataset.hard_dataset for dataset in self.datasets]
        self.reco_datasets = [dataset.reco_dataset for dataset in self.datasets]

        # Hard process datasets #
        print ('Processing the hard datasets')
        MultiDataset(self.hard_datasets)

        # Reco datasets #
        print ('Processing the reco datasets')
        MultiDataset(self.reco_datasets)

        # Compute indices #
        self.compute()

    def __str__(self):
        return f'MultiCombined dataset :\n'+'\n'.join([str(dataset) for dataset in self.datasets])



class MultiDataset(MultiHelperClass):
    def __init__(self,datasets):
        # Super #
        super().__init__(datasets)

        # Process #
        print ('... Undoing preprocessing')
        self.undo_preprocessing(self.datasets)
        print ('... Equalizing datasets')
        self.equalize_datasets(self.datasets)
        print ('... Matching preprocessing')
        self.match_preprocessing(self.datasets)
        print ('... Re-applying common preprocessing')
        self.do_preprocessing(self.datasets)
        print ('... Checking attention mask')
        self.check_attention_masks(self.datasets)

        # Compute indices #
        self.compute()

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
    def equalize_datasets(datasets):
        # Get all datasets selections, take unique values #
        # Not using sets because we want to keep the order #
        all_selection = []
        for dataset in datasets:
            all_selection.extend([name for name in dataset.selection if name not in all_selection])
        # calculate the max number of particles per type #
        max_number_particles_per_name = {
            name : max(
                [
                    dataset.number_particles(name)
                    for dataset in datasets
                    if name in dataset.selection
                ]
            )
            for name in all_selection
        }
        # Get number of features per name
        # As we go, also check consistency
        features_per_name = {}
        for dataset in datasets:
            for name in dataset.selection:
                features = dataset.input_features_particle(name)
                if name not in features_per_name.keys():
                    features_per_name[name] = dataset.input_features_particle(name)
                else:
                    if features_per_name[name] != features:
                        raise RuntimeError(f'Mismatch between features : {features_per_name[name]} != {features} for particles {name}')

        # Get size of each dataset to equalize with weights #
        avg_size = sum([len(dataset) for dataset in datasets]) / len(datasets)

        # For each type, pad the tensors #
        for dataset in datasets:
            # Factor for class imalance #
            factor = avg_size / len(dataset)
            print (f'Reweighting factor : {factor:.5f}')
            # Loop through each type
            for name in all_selection:
                if name in dataset.objects.keys():
                    # Particle type is present, need to check whether to pad it or not
                    if dataset.number_particles(name) < max_number_particles_per_name[name]:
                        n_diff = max_number_particles_per_name[name] - dataset.number_particles(name)
                        data, mask, weights = dataset.objects[name]
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
                    elif dataset.number_particles(name) > max_number_particles_per_name[name]:
                        # This should not happen
                        raise RuntimeError
                    else:
                        # The tensors are already at the maximum length size
                        data, mask, weights = dataset.objects[name]
                else:
                    # Particle is not present, need to make up tensors
                    N = len(dataset)
                    S = max_number_particles_per_name[name]
                    F = len(features_per_name[name])
                    # Make dummy tensors #
                    data = torch.zeros((N,S,F))
                    mask = torch.full((N,S),fill_value=False)
                    weights = torch.ones((N,S))
                    # Need to add features for dummy tensor #
                    dataset.fields[name] = features_per_name[name]
                # Reweight for class imbalance #
                weights *= factor
                # Replace the object #
                dataset.objects[name] = data, mask, weights

            # Make sure the datasets all have the same order of selection
            dataset.selection = all_selection

    @staticmethod
    def undo_preprocessing(datasets):
        for dataset in datasets:
            if dataset._preprocessed:
                dataset._undo_preprocessing()

    @staticmethod
    def do_preprocessing(datasets):
        for dataset in datasets:
            if not dataset._preprocessed:
                dataset._do_preprocessing()

    @staticmethod
    def match_preprocessing(datasets):
        # Find a common preprocessing pipeline #
        preprocessing = PreprocessingPipeline()
        N_steps = list(set([len(dataset.preprocessing.steps) for dataset in datasets]))
        if len(N_steps) != 1:
            raise RuntimeError(f'Different number of steps for preprocessing {N_steps}')
        for i in range(N_steps[0]):
            # Get all i steps #
            steps = [dataset.preprocessing.steps[i] for dataset in datasets]
            # Check the names #
            # Order does not matter, so can use sets #
            names = list(set(chain.from_iterable([step.names for step in steps])))
            # Check the fields_select
            # Check if consistency of Nones
            fields_are_none = [step.fields_select is None for step in steps]
            if all(fields_are_none):
                # No need to check further
                fields_select = None
            elif all([~is_none for is_none in fields_are_none]):
                # All different than None, need to check the features
                fields_select = []
                for name in names:
                    fields = None
                    for step in steps:
                        if name in step.names:
                            step_fields = step.fields_select[step.names.index(name)]
                            if fields is None:
                                fields = step_fields
                            else:
                                if fields != step_fields:
                                    raise RuntimeError(f'At preprocessing step {i}, for name {name}, mismatch in field_select : {fields} != {step_fields}')
                    fields_select.append(fields)
            else:
                raise RuntimeError(f'At preprocessing step {i}, mismatch in fields_select==None : {fields_are_none}')
            # Check the scaler_dict #
            # First make sure we have a match in the keys/features to use #
            keys = steps[0].scaler_dict.keys()
            for j in range(1,len(steps)):
                if keys != steps[j].scaler_dict.keys():
                    raise RuntimeError(f'At preprocessing step {i}, found different set of keys between dataset 0 ({keys}) and dataset {j} ({steps[i].scaler_dict.keys()})')
            # For each key, make sure we have same class #
            for key in keys:
                classes = list(set([step.scaler_dict[key].__class__ for step in steps]))
                if len(classes) != 1:
                    raise RuntimeError(f'At preprocessing step {i}, found different scaler classes for feature {key}: {classes}')
                # if a SklearnScaler, want to makes sure it uses the same sklearn preprocessor
                if isinstance(steps[0].scaler_dict[key],SklearnScaler):
                    sklearn_classes = list(set([step.scaler_dict[key].obj.__class__ for step in steps]))
                    if len(sklearn_classes) != 1:
                        raise RuntimeError(f'At preprocessing step {i}, found different scikit-learn classes for feature {key}: {sklearn_classes}')

            # Reset the sklearn scalers in the scaler dict
            # (kind of hack from https://github.com/scikit-learn/scikit-learn/blob/6e9039160f0dfc3153643143af4cfdca941d2045/sklearn/utils/validation.py#L1549-L1584)
            scaler_dict = deepcopy(steps[0].scaler_dict)
            # Copy because we need to inverse the preprocessing below
            for var,scaler in scaler_dict.items():
                if isinstance(scaler,SklearnScaler):
                    # Get attributes that are obtained through fit
                    vars_from_fit = [
                        v for v in vars(scaler.obj) if v.endswith("_") and not v.startswith("__")
                    ]
                    # remove them from the attributes -> reset the fit
                    for var in vars_from_fit:
                        delattr(scaler.obj,var)

            # Record the common preprocessing #
            preprocessing.add_step(
                PreprocessingStep(
                    names = names,
                    scaler_dict = scaler_dict,
                    fields_select = fields_select,
                )
            )
        print (preprocessing)

        # Get all the combined data, mask and fields #
        # Note : in principle should be unprocessed data
        datas, masks, fields = {}, {}, {}
        for dataset in datasets:
            for name in dataset.selection:
                # Extract #
                data,mask,_ = dataset.objects[name]
                if mask.dtype != torch.bool:
                    mask = mask > 0
                flds = tuple(dataset.fields[name])

                # Record #
                if name in datas.keys():
                    assert datas[name][0].shape[1:] == data.shape[1:], f'Data shape [S,F] mismatch : {datas[name][0].shape[1:]} vs {data.shape[1:]}'
                    datas[name].append(data)
                else:
                    datas[name] = [data]
                if name in masks.keys():
                    assert masks[name][0].shape[1] == mask.shape[1], f'Mask shape [S] mismatch : {masks[name][0].shape[1]} vs {mask.shape[1]}'

                    masks[name].append(mask)
                else:
                    masks[name] = [mask]
                if name in fields.keys():
                    if fields[name] != flds:
                        raise RuntimeError(f'Mismatch in fields for preprocessing for name {name} : {fields[name]} != {flds}')
                else:
                    fields[name] = flds
        # Concat
        datas = {key:torch.cat(values,dim=0) for key,values in datas.items()}
        masks = {key:torch.cat(values,dim=0) for key,values in masks.items()}

        # Fit the common preprocessing #
        names = list(datas.keys())
        preprocessing.fit(
            names = names,
            xs = [datas[name] for name in names],
            masks = [masks[name] for name in names],
            fields = [fields[name] for name in names],
        )

        # Set the common preprocessing for all datasets and apply it
        # This is needed for when we plot them separately
        for dataset in datasets:
            dataset._preprocessing = preprocessing
            dataset._save_preprocessing()

    def __str__(self):
        return f'MultiDataset :\n'+'\n'.join([str(dataset) for dataset in self.datasets])




class CombinedDataset(Dataset):
    """
        Dataset to combine hard and reco and provide events with both information
    """
    def __init__(self,hard_dataset,reco_dataset):
        """
            Args:
             - hard_dataset [HardDataset] : hard scattering level dataset
             - reco_dataset [RecoDataset] : reconstructed level dataset
        """
        self.hard_dataset = hard_dataset
        self.reco_dataset = reco_dataset

        assert isinstance(self.hard_dataset,HardDataset)
        assert isinstance(self.reco_dataset,RecoDataset)

        if self.hard_dataset.intersection_branch is None and self.reco_dataset.intersection_branch is None:
            # Assume the trees are the same for both reco and hard,
            # will just check their length and file branch
            print ('Not intersection branch for either hard or reco datasets, will assume bijection between datasets')
            if len(self.hard_dataset.metadata['file']) != len(self.reco_dataset.metadata['file']):
                raise RuntimeError(f'Different number of entries between hard ({len(self.hard_dataset.metadata["file"])}) compared to reco ({len(self.reco_dataset.metadata["file"])})')
            if not all(self.hard_dataset.metadata['file'] == self.reco_dataset.metadata['file']):
                raise RuntimeError(f'Different `file` content in hard and reco file metadata : {self.hard_dataset.metadata["file"]} and {self.reco_dataset.metadata["file"]}')
            self.N = len(self.hard_dataset.metadata['file'])
            self.hard_idx, self.reco_idx = None,None
        elif self.hard_dataset.intersection_branch is not None and self.reco_dataset.intersection_branch is not None:
            print (f'Intersection branches : `{self.hard_dataset.intersection_branch}` for hard dataset and `{self.reco_dataset.intersection_branch}` for reco dataset')
            hard_files = set(np.unique(self.hard_dataset.metadata['file']))
            reco_files = set(np.unique(self.reco_dataset.metadata['file']))
            self.hard_idx, self.reco_idx = get_metadata_intersection_indices(
                metadatas = [
                    self.hard_dataset.metadata,
                    self.reco_dataset.metadata,
                ],
                different_files = len(hard_files.intersection(reco_files)) == 0,
            )
            assert len(self.hard_idx) == len(self.reco_idx)
            self.N = len(self.hard_idx)
        else:
            msg = 'Mismatch in having an `intersection_branch` property:'
            if self.hard_dataset.intersection_branch is not None:
                msg += f'\n   - Hard dataset has {self.hard_dataset.intersection_branch}'
            else:
                msg += f'\n   - Hard dataset does not have an intersection branch'
            if self.reco_dataset.intersection_branch is not None:
                msg += f'\n   - Reco dataset has {self.reco_dataset.intersection_branch}'
            else:
                msg += f'\n   - Reco dataset does not have an intersection branch'
            raise RuntimeError(msg)

    def __getitem__(self, index):
        """ Returns the event info of both hard and reco level variables """
        if self.hard_idx is None:
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
        if self.hard_idx is None:
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

        if self.hard_idx is None:
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

    def __init__(self,data,selection,default_features=None,build=False,fit=True,device=None,dtype=None,**kwargs):
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
            - fit [bool] : whether to fit the preprocessing pipeline (if False, load if was built)
            - device [str] : device for torch tensors
            - dtype [torch.type] : torch type for torch tensors
        """
        # Public attributes #
        self.data = data
        self.build = build
        self.fit = fit
        self.selection = selection
        self.default_features = default_features
        assert len(self.selection) > 0, 'You need to have at least some object selected'

        # Internal data #
        self.preprocessing = PreprocessingPipeline()
        self.objects = {}
        self.fields = {}
        self.metadata = {}

        # Private attributes #
        self._reserved_object_names = ['ps','detjinv','metadata']
        self._forbidden_characters = [' ','/',r'\\',r'\t',r'\n']
        self._preprocessed = False

        # Calling methods #
        self.init()
        if self.processed_path is not None and os.path.exists(self.processed_path) and not self.build:
            self._load()
        else:
            self.process()
            self._get_metadata()
            self._save()
        self.finalize()
        self._standardize()
        if self.processed_path is not None and os.path.exists(self.processed_path) and self.fit:
            self._fit_preprocessing()
            self._save_preprocessing()
        else:
            self._load_preprocessing()
        self._do_preprocessing()
        self._to_device(device)
        self._to_dtype(dtype)

        # Safety checks #
        for name in self.selection:
            if name not in self.objects.keys():
                raise RuntimeError(f'`{name}` not in registered objects {self.objects.keys()}')
        events = []
        for key,(data,mask,weights) in self.objects.items():
            if data.shape[0] != mask.shape[0]:
                raise RuntimeError(f'Object `{key}` has {data.shape[0]} data events but {mask.shape[0]} mask events')
            events.append(data.shape[0])
        events = list(set(events))
        if len(events) != 1:
            raise RuntimeError(f'Number of events mismatch {events} for objects {list(self.objects.keys())}')
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
        """
            Turns object (awkward array) to tensor
            Fields are the only ones turned into the final tensor
        """
        if not isinstance(obj,ak.Array):
            raise RuntimeError(f'Expects an awkward array, got {type(obj)}')
        # From awkward to numpy #
        arr = to_flat_numpy(
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
            Self : save object in self.objects and fields in in self.fields
        """
        # Avoid overwriting objects #
        if name in self._reserved_object_names:
            raise RuntimeError(f'Name `{name}` already reserved')
        if name in self.objects.keys():
            raise RuntimeError(f'Name `{name}` already present in objects')
        for char in self._forbidden_characters:
            if char in name:
                raise RuntimeError(f'Character `{char}` forbidden in name `{name}`')
        # Get fields and then data tensore #
        fields = fields if fields is not None else obj.fields
        data = self.object_to_tensor(obj,fields)
        if data.nelement() == 0:
            raise RuntimeError(f'Trying to register an empty tensor {data.size()} for name {name}')
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
        self.objects[name] = (data,mask,weights)
        self.fields[name] = fields

    def register_preprocessing_step(self,preprocessing):
        """
            Register a PreprocessingStep (from memflow.dataset.preprocessing)
        """
        assert isinstance(preprocessing,PreprocessingStep), f'Preprocessing must be a PreprocessingStep instance'
        assert len(set(preprocessing.keys())-set(self.objects.keys())) > 0, f'Object names {[key for key in preprocessing.keys() if key not in self.objects.keys()]} have not been registered yet'
        self.preprocessing.add_step(preprocessing)

    def _get_metadata(self):
        branches = ['file','tree','sample'] # branches by default in data
        for branch in branches:
            self.metadata[branch] = self.data[branch].to_numpy()
        if self.intersection_branch is not None:
            assert isinstance(self.intersection_branch,str)
            self.metadata['intersection'] = self.data[self.intersection_branch].to_numpy()


    def _do_preprocessing(self):
        if self._preprocessed:
            raise RuntimeError('Cannot do preprocessing, data is already preprocessed')
        for name in self.objects.keys():
            data, mask, weights = self.objects[name]
            data, fields = self.preprocessing.transform(name,data,mask,self.fields[name])
            self.objects[name] = (data,mask,weights)
            self.fields[name] = fields
        self._check_tensors('Preprocessed tensors')
        self._preprocessed = True

    def _undo_preprocessing(self):
        if not self._preprocessed:
            raise RuntimeError('Cannot undo preprocessing, data has not been preprocessed')
        for name in self.objects.keys():
            data, mask, weights = self.objects[name]
            data, fields = self.preprocessing.inverse(name,data,mask,self.fields[name])
            self.objects[name] = (data,mask,weights)
            self.fields[name] = fields
        self._check_tensors('Unpreprocessed tensors')
        self._preprocessed = False

    def _fit_preprocessing(self):
        self.preprocessing.fit(
            names = list(self.objects.keys()),
            xs = [self.objects[name][0] for name in self.objects.keys()],
            masks = [self.objects[name][1] for name in self.objects.keys()],
            fields = [self.fields[name] for name in self.objects.keys()],
        )

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

    def _standardize(self):
        """
            If default features asked, will fill the missing ones for each object
            Self : modifies the self.objects and self.fields
        """
        if self.default_features is None:
            return
        elif isinstance(self.default_features,(float,int)):
            # Get the comprehensive list of fields and complete with value #
            default_val = self.default_features
            all_fields = sorted(list(set(chain.from_iterable(self.fields.values()))))
            default_values = {f:default_val for f in all_fields}
        elif isinstance(self.default_features,dict):
            default_values = self.default_features
        else:
            raise NotImplementedError(f'Type {type(self.default_features)} of default_features not implemented')
        # Modify each object data #
        for name,(data,mask,weights) in self.objects.items():
            fields = self.fields[name]
            data,fields = self.expand_tensor(data,fields,default_values)
            self.objects[name] = (data,mask,weights)
            self.fields[name] = fields

    @staticmethod
    def _check_tensor(t,flag):
        """
            Perform a few checks (inf and nan)
        """
        is_error = torch.logical_or(
            torch.isnan(t),
            torch.isinf(t)
        )
        if is_error.sum() > 0:
            msg = ''
            coords = torch.where(is_error)
            msg = f'{flag} : there are nans/infs in the tensor'
            for i in range(len(coords[0])):
                if len(coords) == 3:
                    msg += f'\nEvent {coords[0][i]}, particle {coords[1][i]} : {t[coords[0][i],coords[1][i],coords[2][i]]}'
                elif len(coords) == 2:
                    msg += f'\nEvent {coords[0][i]} : {t[coords[0][i],coords[1][i]]}'
                elif len(coords) == 1:
                    msg += f'\nEvent {coords[0][i]} : {t[coords[0][i]]}'
                else:
                    raise ValueError
            raise RuntimeError(msg)


    def _check_tensors(self,step):
        for name in self.selection:
            data, mask, weights = self.objects[name]
            # Check for nans and infs
            self._check_tensor(data,f'{step} / {name} [data]')
            self._check_tensor(mask,f'{step} / {name} [mask]')
            self._check_tensor(weights,f'{step} / {name} [weights]')


    ##### Intrinsic properties #####

    def input_features_particle(self,name):
        return tuple(self.fields[name])

    @property
    def input_features(self):
        return tuple(
            self.input_features_particle(name)
            for name in self.selection
        )

    def number_particles(self,name):
        return self.objects[name][0].shape[1]

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
    def _to_device(self,device=None):
        """ Move all tensors to device """
        if device is None:
            return
        for name,(data,mask,weights) in self.objects.items():
            self.objects[name] = (data.to(device),mask.to(device),weights.to(device))

    def _to_dtype(self,dtype=None):
        """ Modifies the type of all tensors """
        if dtype is None:
            return
        for name,(data,mask,weights) in self.objects.items():
            self.objects[name] = (data.to(dtype),mask.to(dtype),weights.to(dtype))

    ##### Save and load methods ######
    def _save(self):
        """ If user defines the property processed_path, will save each object to file """
        if self.processed_path is not None:
            print (f'Saving objects to {self.processed_path}')
            # Make output directory #
            if not os.path.exists(self.processed_path):
                os.makedirs(self.processed_path)
            # Save the different tensors #
            for name in self.objects.keys():
                data,mask,weights = self.objects[name]
                fields = self.fields[name]
                torch.save(
                    (data,mask,weights,fields),
                    os.path.join(self.processed_path,f'{name}.pt')
                )
            # Save metadata #
            torch.save(
                self.metadata,
                os.path.join(self.processed_path,'metadata.pt')
            )
        else:
            print ('No `processed_path` provided, will not save the data')
        self._check_tensors('Processed tensors')

    def _save_preprocessing(self):
        """ If user defines the property processed_path, will save the preprocessing pipeline """
        if self.processed_path is not None:
            print (f'Saving preprocessing to {self.processed_path}')
            # Make output directory #
            preprocessing_dir = os.path.join(self.processed_path,'preprocessing')
            if not os.path.exists(preprocessing_dir):
                os.makedirs(preprocessing_dir)
            # Save #
            self.preprocessing.save(preprocessing_dir)
        else:
            print ('No `processed_path` provided, will not save the preprocessing')
        self._check_tensors('Processed tensors')


    def _load(self):
        """ If user defines the property processed_path, will load each object from file """
        if self.processed_path is None:
            raise RuntimeError('Not processed path defined, cannot load the data')
        print (f'Loading objects from {self.processed_path}')
        for f in glob.glob(os.path.join(self.processed_path,'*.pt')):
            name = os.path.basename(f).replace('.pt','')
            content = torch.load(f)
            if name == 'metadata':
                self.metadata = content
            else:
                data,mask,weights,fields = content
                self.objects[name] = (data,mask,weights)
                self.fields[name] = fields
        if len(self.metadata) == 0:
            raise RuntimeError(f'Could not find {os.path.join(self.processed_path,"metadata.pt")}')
        self._check_tensors('Loaded tensors')

    def _load_preprocessing(self):
        """ If user defines the property processed_path, will load preprocessing """
        if self.processed_path is None:
            raise RuntimeError('Not processed path defined, cannot load the data')
        preprocessing_dir = os.path.join(self.processed_path,'preprocessing')
        if not os.path.exists(preprocessing_dir):
            raise RuntimeError(f'Cannot find preprocessing subdirectory {preprocessing_dir}')
        self.preprocessing.load(preprocessing_dir)

    ##### Magic methods #####
    def __getitem__(self,idx):
        """
            For each object specified in the selections, returns the element from idx
            Joins the data and mask along new dimension (axis = 2)
        """
        return {
            'data': [self.objects[name][0][idx] for name in self.selection],
            'mask': [self.objects[name][1][idx] for name in self.selection],
            'weights': [self.objects[name][2][idx] for name in self.selection],
        }

    def __len__(self):
        """ Number of events """
        return self.events

    def __str__(self):
        """ Representation string to show all defined objects with some info """
        s = f'\nContaining the following tensors'
        names_len = max([len(name) for name in self.objects.keys()]) + 1
        for name,(data,mask,weights) in self.objects.items():
            props = mask.sum(axis=0) / mask.shape[0] * 100
            prop_str = ', '.join([f'{prop:3.2f}%' if prop==0. or prop>=0.01 else '<0.01%' for prop in props])
            w_sum = weights.sum(dim=0)
            w_str = ', '.join([f'{ws:.2f}' for ws in w_sum])
            s += f'\n{name:{names_len}s} : data ({list(data.shape)}), mask ({list(mask.shape)})'
            s += f'\n{" "*names_len}   Mask exist    : [{prop_str}]'
            s += f'\n{" "*names_len}   Mask attn     : {self.object_attention_mask(name)}'
            s += f'\n{" "*names_len}   Weights       : {w_str}'
            s += f'\n{" "*names_len}   Features      : {self.fields[name]}'
            s += f'\n{" "*names_len}   Selected for batches : {name in self.selection}'
        s += str(self.preprocessing)
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
                    input = self.objects[name][0],
                    dim = 0,
                    index = idx,
                )
                for name in self.selection
            ],
            'mask': [
                torch.index_select(
                    input = self.objects[name][1],
                    dim = 0,
                    index = idx,
                )
                for name in self.selection
            ],
            'weights': [
                torch.index_select(
                    input = self.objects[name][2],
                    dim = 0,
                    index = idx,
                )
                for name in self.selection
            ],
        }

    def plot(self,fields_to_plot=None,raw=False,selection=False,weighted=False,log=False):
        names = self.selection if selection else self.objects.keys()
        if fields_to_plot is not None:
            if isinstance(fields_to_plot,(list,tuple)):
                pass
            elif isinstance(fields_to_plot,str):
                fields_to_plot = [fields_to_plot]
            else:
                raise RuntimeError(f'Type {type(fields_to_plot)} not understood')
        for name in names:
            # Get objects for name #
            data,mask,weights = self.objects[name]
            fields = self.fields[name]
            # Some processing #
            if mask.dtype != torch.bool:
                mask = mask > 0
            if raw:
                data,fields = self.preprocessing.inverse(name,data,mask,fields)
            # Generate figure #
            n_parts = data.shape[1]
            if fields_to_plot is None:
                fields_to_plot = fields
                field_indices = torch.arange(len(fields))
                n_cols = len(fields)
            else:
                fields_to_plot = [field for field in fields if field in fields_to_plot]
                field_indices = [i for i,field in enumerate(fields) if field in fields_to_plot]
                n_cols = len(fields_to_plot)
            if n_parts > 1:
                fig,axs = plt.subplots(1,n_cols+1,figsize=(4.5*(n_cols+1),4))
            else:
                fig,axs = plt.subplots(1,n_cols,figsize=(4.5*n_cols,4))
            if not isinstance(axs,np.ndarray):
                axs = np.array([axs])
            fig.suptitle(f'Objects : {name}')
            # Loop over features #
            for k,(i,field) in enumerate(zip(field_indices,fields_to_plot)):
                # Find index from field name #
                if field not in fields:
                    raise RuntimeError(f'Field {field} not in {fields}')
                # Make binning #
                bins = np.linspace(data[:,:,i].min(),data[:,:,i].max(),50)
                # Loop over particles #
                for j in range(n_parts):
                    if weighted:
                        axs[k].hist(data[:,j,i][mask[:,j]],weights=weights[:,j][mask[:,j]],bins=bins,histtype='step',linewidth=2)
                    else:
                        axs[k].hist(data[:,j,i][mask[:,j]],bins=bins,histtype='step',linewidth=2)
                axs[k].set_xlabel(field)
                if log:
                    axs[k].set_yscale('log')
                    axs[k].set_ylim((0.1,None))
                else:
                    axs[k].set_ylim((0,None))
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

    @property
    def intersection_branch(self):
        return None

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
            if name not in self.objects.keys():
                raise RuntimeError(f'{name} not found in the objects recorded {self.objects.keys()}')
            momenta.append(self.objects[name][0])
        shapes = [mom.shape for mom in momenta]
        for axis in [0,2]:
            if len(set([shape[axis] for shape in shapes])) != 1:
                raise RuntimeError(f'Shapes mismatch {shapes} for axis {axis}')
        momenta = torch.cat(momenta,dim=1)

        if 'boost' not in self.objects.keys():
            raise RuntimeError(f'`boost` must have been recorded in the objects')
        boost = self.objects['boost'][0]

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
        self.objects['ps'] = (
            ps.unsqueeze(dim=-1),
            torch.full(ps.shape,True)#.unsqueeze(-1),
        )
        self.fields['ps'] = [f'ps_{i}' for i in range(ps.shape[1])]
        self.objects['detjinv'] = (
            detjinv.unsqueeze(-1),
            torch.full(detjinv.shape,True)#.unsqueeze(-1),
        )
        self.fields['detjinv'] = ['det']

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

