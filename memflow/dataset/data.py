import os
import vector
import numpy as np
import pandas as pd
import pyarrow as pa
import awkward as ak
import dask_awkward as dak
import dask.dataframe as dd
import uproot
from collections.abc import Mapping
from abc import abstractmethod
from functools import reduce, cached_property

vector.register_awkward()

def get_intersection_indices(datas,branch,different_files=False):
    """
        Return a list of indices for each data instance that intersect on the branch provided
        Args:
         - datas [list] : list of AbsData instances
         - branch [str] : data branch on which to check for intersection (eg, event number)
        Note : this search is performed file by file, assuming different trees
    """
    # Make recasting and checks #
    if not isinstance(datas,(list,tuple)):
        datas = [datas]
    assert len(datas) > 1, f'Need at least 2 Data objects, got {len(data)}'
    for i,data in enumerate(datas):
        assert isinstance(data,AbsData), f'Data entry number {i} (type : {type(data)}) is not a Data class'

    # recover unique file names between all the Data objects #
    print ('Looking into file metadata')
    unique_files = None
    for i, data in enumerate(datas):
        assert data['file'].layout.purelist_depth == 1
        uniq = np.unique(data['file'].to_numpy())
        print (f'\tentry {i} : {uniq}')
        if unique_files is None:
            unique_files = uniq
        else:
            unique_files = [f for f in unique_files if f in uniq]
    print (f'Will only consider common files : {unique_files}')
    print ('(Note : this assumes the files have the same order between the different data objects')

    # Obtain set of indices for each Data object that intersect on the branch #
    idxs = [np.array([],dtype=np.int64) for _ in range(len(datas))]
    sizes = [0 for _ in range(len(datas))] # Need to avoid resetting indices to zero
    for i,file in enumerate(unique_files):
        arrays = [data[branch][data['file']==file].to_numpy() for data in datas]
        matched = reduce(np.intersect1d, arrays) # find common values between all arrays
        for j in range(len(datas)):
            idx = np.nonzero(np.in1d(arrays[j],matched))[0] # for specific array, get common indices with matched
            idx += sizes[j]
            idxs[j] = np.concatenate((idxs[j],idx),axis = 0)
            sizes[j] += len(arrays[j])  # keep track to add to next iteration

    # Info printout #
    for i in range(len(datas)):
        print (f'For entry {i} : from {datas[i].events} events, {len(idxs[i])} selected')

    # Safety check : make sure the intersection branch returns the same values
    for i in range(1,len(datas)):
        if not all(datas[0][branch][idxs[0]] == datas[i][branch][idxs[i]]):
            print (f'Disagreement between Data object 0 and {i} on branch {branch} after the index selection')

    return idxs



class AbsData(Mapping):
    """
        Abstract class for loading of the data from several types of files
        Loading can be complete (default) or lazily (only when asked)
        Usage : same as a dictionnary
    """
    def __init__(self,files=[],treenames=[],N=None,lazy=False):
        """
            Initialize the data

            Args:
                - files [list] : list of files (format depends on the inherited class)
                - treenames [list] : list of trees (needed for ROOT files)
                - N [int] (default=None) : maximum number of entries to load per tree (useful for testing)
                - lazy [bool] (default=False): when True, will use dask to load branches only when requesting them, then keeping them in memory
        """
        # Save inputs in self #
        self.files = files if isinstance(files,(list,tuple)) else [files]
        self.treenames = treenames if isinstance(treenames,(list,tuple)) else [treenames]
        self.lazy = lazy
        self.N = N

        # Bookkeeping objects #
        self.trees = []
        self.entries = []
        self.data = {'file':ak.Array([]),'tree':ak.Array([]),'sample':ak.Array([])}
        self.idx = None

    @abstractmethod
    def getitem(self,entries,tree):
        """ inherited class getitem function """
        pass

    def _getitem(self,key):
        """ Separate getitem to do the concat """
        arrays = []
        for entries,tree in zip(self.entries,self.trees):
            arrays.append(self.getitem(entries,tree,key))
        array = ak.concatenate(arrays,axis=0)
        if isinstance(array.layout,ak.contents.listoffsetarray.ListOffsetArray):
            pass # the typical type of awkward array we want
        if isinstance(array.layout,ak.contents.numpyarray.NumpyArray):
            pass # other typical layout we want
        elif isinstance(array.layout,ak.contents.bytemaskedarray.ByteMaskedArray):
            # arrays with a option[var* ...] type, nasty
            # pass to list then proper array
            array = ak.Array(array.tolist())
        else:
            raise NotImplementedError(f'Unexpected layout type {type(array.layout)}')
        return array

    def __getitem__(self,key):
        """
            Magic getitem method
            Looks for the keys in already loaded ones, if not found loads the branch
            If a cut has been done before, only select the corresponding indices
        """
        if key not in self.keys():
            self.data[key] = self._getitem(key)
        if self.idx is not None:
            return self.data[key][self.idx]
        else:
            return self.data[key]

    def __setitem__(self,key,val):
        """ set item with checking if key already there """
        if key in self.keys():
            print (f'Key {key} already in the data, will modify it')
        self.data[key] = val

    def __len__(self):
        """ len here means number of loaded branches/keys """
        return len(self.data)

    def __iter__(self):
        """ Iterator  """
        return self.data.__iter__()

    def __str__(self):
        """ Representation, printing all branches (loaded or not) """
        s = 'Data object\nLoaded branches:'
        for key in sorted(self.keys()):
            s += f'\n   ... {key}: {len(self[key])}'
        s += f'\nBranch in files not loaded:'
        for branch in sorted(self.branches):
            if branch not in self.keys():
                s += f'\n   ... {branch}'
        return s

    def delete(self,key):
        """ helper to remove a branch from memory """
        if key not in self.keys():
            print (f'Key {key} is not registered')
        else:
            del self.data[key]

    def cut(self,mask):
        """
            Given a mask corresponding to the current selection, update indices to select
            (eg, mask = data['my_branch'] > some_value)
        """
        # Checks to make sure mask corresponds to current state of idx #
        assert len(mask) == self.events, f'Mask has {len(mask)} elements but {self.events} events are selected so far'

        # Get index of selected events #
        idx = np.where(mask)[0]
        self._store_idx(idx)

    def _store_idx(self,idx):
        """ Update the indices """
        if self.idx is None:
            self.idx = idx
        else:
            self.idx = self.idx[idx]

    def reset_cuts(self):
        """ Remove any prior mask/selection by reseting the index """
        self.idx = None


    @abstractmethod
    def branches(self):
        """ List of branches already in file """
        pass

    @property
    def events(self):
        """ Number of events in the dataset """
        if self.idx is None:
            return sum(self.entries)
        else:
            return len(self.idx)

    def keys(self):
        """ Keys already loaded """
        return self.data.keys()

    def make_particles(self,name,link,lambda_mask=None,pad_value=None):
        """
            link links the different branches into the record fields for a Momentum4D vector
            eg:
            link = {
                'px' : ['p1_Px','p2_Px', ...],
                'py' : ['p1_Py','p2_Py', ...],
                'pz' : ['p1_Pz','p2_Pz', ...],
                'E'  : ['p1_E','p2_E', ...],
                'foo': ['p1_foo','p2_foo', ...],
            }
            Note :
            - to have a proper 4-vector, need to have either at least ['px','py','pz','E'] or ['pt','eta','phi','mass']
            - any additional variable can be included (foo, or pdg id, btag, etc)
            - if a branch is already an akward array containing the particles info, its name can be provided as a string to link
                -> in this case the only operation is to turn it into a Momentum4D record
            lambda_mask is a lambda applied on the final akward array to create the ragged array from the "rectangular" one
            (eg to remove placeholder particles that had default values filled in)
        """
        # Check if the name is already there #
        if name in self.keys():
            print (f'{name} is already in data, will not modify it')
            return self[name]
        # If dict, concatenate the particles into a vector awkward array #
        if isinstance(link,dict):
            # make sure the values are list (needed later)
            for key in link.keys():
                if not isinstance(link[key],(list,tuple)):
                    link[key] = [link[key]]
            # Safety checks #
            for key,values in link.items():
                for val in values:
                    if isinstance(val,str):
                        if val not in self.branches:
                            raise RuntimeError(f'Branch {val} not found in file')
                    elif isinstance(val,(float,int)):
                        pass
                    else:
                        raise NotImplementedError(f'Type {type(val)} of {key} not implemented')
            # Do all the getitem #
            arrays = {
                key : [
                    self._getitem(value)
                    if isinstance(value,str) else value
                    # not using self[key] because would call __getitem__
                    # and therefore include the index selection
                    # but we want the whole column, to do the index selection later
                    for value in values
                ]
                for key,values in link.items()
            }
            # process into dict of awkward arrays #
            # if numpy arrays, need to turn into awkward array
            for key in arrays.keys():
                # check uniformity of types #
                types = set([type(arr) for arr in arrays[key]])
                if len(types) != 1:
                    raise RuntimeError(f'Found multiple types {types} for {key}')
                types = list(types)[0]
                # if number, will update afterwards #
                if types == float or types == int:
                    assert len(arrays[key]) == 1
                    arrays[key] = arrays[key][0]
                # check whether we can combine the awkward arrays #
                elif types == ak.Array:
                    # Pad the array if requested #
                    if pad_value is not None:
                        arr = arrays[key][0]
                        arrays[key] = [
                            ak.fill_none(
                                ak.pad_none(
                                    arr,
                                    target = ak.max(ak.num(arr,axis=1)), # max number of entries
                                    axis = 1,
                                ),
                                value = pad_value
                            )
                            for arr in arrays[key]
                        ]
                    # handle different scenarios
                    if len(arrays[key]) == 1:
                        arrays[key] = arrays[key][0]
                    else:
                        # Avoid 3D arrays #
                        depths = [arr.layout.purelist_depth for arr in arrays[key]]
                        if any([depth > 2 for depth in depths]):
                            raise RuntimeError(f'Cannot deal with > 2 depth : {depths}')
                        # make sure we can "numpify" the arrays, ie rectangular arrays, ie same dims on axis=1
                        if not all([depth == 1 for depth in depths]):
                            nums = [np.unique(ak.num(arr,axis=1)) for arr in arrays[key]]
                            if any([len(num)>1 for num in nums]):
                                raise RuntimeError(f'Some awkward arrays for {key} have not unique counts on axis=1 ({nums}), cannot turn them into rectangular arrays and concatenate them')
                        # if all good : awkward arrays -> numpy arrays -> concat -> awkward array -> list
                        arrays[key] = ak.Array(
                            np.concatenate(
                                [
                                    arr.to_numpy().reshape(-1,1)
                                    for arr in arrays[key]
                                ],
                                axis=1,
                            )
                        )

                # if numpy arrays, concatenate and turn into awkward array
                elif types == np.ndarray:
                    arrays[key] = ak.Array(
                        np.concatenate(
                            [
                                arr.reshape(-1,1)
                                for arr in arrays[key]
                            ],
                            axis=1
                        )
                    )
                else:
                    raise TypeError(f'Type {types} not implemented')
            # Turn the floats/ints into the awkward arrays #
            # Turn into vector awkward array #
            vec = ak.zip(
                {key: array.tolist() if isinstance(array,ak.Array) else array for key,array in arrays.items()},
                # need the list to get *var* number of entries on axis=1
                with_name="Momentum4D",
            )
            if lambda_mask is not None:
                vec = vec[lambda_mask(vec)]
            self[name] = vec
        # If string, just make sure it is treated as Momentum4D #
        elif isinstance(link,str):
            # Safety checks #
            if link not in self.branches:
                raise RuntimeError(f'Branch {link} not found in file')
            if len(self[link].layout.content.parameters) > 0 and \
                    self[link].layout.content.parameters['__record__'] == 'Momentum4D':
                raise RuntimeError(f'Branch {link} already has the `Momentum4D` record')
            self[name] = ak.with_name(self._getitem(link), name="Momentum4D")
        else:
            raise NotImplementedError(f'Type {type(link)} of link not understood')

        return self[name]

class RootData(AbsData):
    """ Cache for loading data from ROOT tree """
    def __init__(self,**kwargs):
        # Call base #
        super().__init__(**kwargs)

        # Root-dependnt processing #
        for f in self.files:
            if isinstance(f,uproot.reading.ReadOnlyDirectory):
                F = f
            elif isinstance(f,str):
                if not os.path.isfile(f):
                    raise RuntimeError(f'File {f} does not exist')
                F = uproot.open(f)
            else:
                raise ValueError
            for treename in self.treenames:
                if treename not in F.keys():
                    raise RuntimeError(f'Tree {treename} is not in file {f}')
                if self.lazy:
                    tree = uproot.dask(f'{F._file._file_path}:{treename}',library='ak',step_size='100 MB')
                    n_tree = len(tree[tree.fields[0]])
                else:
                    tree = F[treename]
                    n_tree = tree.num_entries
                if self.N is not None:
                    n_tree = min(n_tree,self.N)
                self.trees.append(tree)
                self.entries.append(n_tree)
                self.data['file'] = ak.concatenate(
                    (
                        self.data['file'],
                        ak.Array([F.file_path]*n_tree),
                    ),
                    axis=0,
                )
                self.data['tree'] = ak.concatenate(
                    (
                        self.data['tree'],
                        ak.Array([tree.name]*n_tree),
                    ),
                    axis=0,
                )
                self.data['sample'] = ak.concatenate(
                    (
                        self.data['sample'],
                        ak.Array([os.path.basename(F.file_path)]*n_tree),
                    ),
                    axis=0,
                )

    @cached_property
    def branches(self):
        branches = set()
        for tree in self.trees:
            if isinstance(tree,dak.Array):
                branches.update(tree.fields)
            else:
                branches.update(tree.keys())
        return sorted(list(branches))


    def getitem(self,entries,tree,key):
        if isinstance(tree,dak.Array):
            if key not in tree.fields:
                raise KeyError(f'Key {key} is not present in tree')
            return tree[key][:entries].compute()
        else:
            if key not in tree.keys():
                raise KeyError(f'Key {key} is not present in tree {tree.name} of file {tree.file.file_path}')
            return tree[key].array()[:entries]


class ParquetData(AbsData):
    """ Cache for loading data from Parquet file """
    def __init__(self,**kwargs):
        # Call base #
        super().__init__(**kwargs)

        # Parquet-dependant processing #
        for f in self.files:
            assert isinstance(f,str)
            if self.lazy:
                df = dak.from_parquet(f)
                n_tree = ak.num(df,axis=0).compute()
                if self.N is not None:
                    n_tree = min(n_tree,self.N)
                self.trees.append(df)
            else:
                df = ak.from_parquet(f)
                n_tree = len(df)
                if self.N is not None:
                    n_tree = min(n_tree,self.N)
                for field in df.fields:
                    if field in self.data.keys():
                        self.data[field] = np.concatenate(
                            (
                                self.data[field],
                                df[field][:n_tree],
                            ),
                            axis = 0,
                        )
                    else:
                        self.data[field] = df[field][:n_tree]
                self.trees.append(None)

            self.entries.append(n_tree)
            self.data['file'] = np.concatenate(
                (
                    self.data['file'],
                    np.array([f]*n_tree),
                ),
                axis = 0,
            )
            self.data['tree'] = np.concatenate(
                (
                    self.data['tree'],
                    np.array(['tree']*n_tree),
                ),
                axis = 0,
            )
            self.data['sample'] = np.concatenate(
                (
                    self.data['sample'],
                    np.array([os.path.basename(f)]*n_tree),
                ),
                axis = 0,
            )

    @cached_property
    def branches(self):
        if self.lazy:
            return list(set([
                col
                for tree in self.trees
                for col in tree.fields
            ]))
        else:
            return self.keys()

    def getitem(self,entries,tree,key):
        if tree is None and not self.lazy:
            raise RuntimeError(f'When running without dask, the dataframe is loaded entirely, this getitem should not happen')
        else:
            assert self.lazy
            return tree[key][:entries].compute()

