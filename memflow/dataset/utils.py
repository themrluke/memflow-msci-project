import numpy as np
import awkward as ak
from functools import reduce

from memflow.dataset.data import AbsData

def to_flat_numpy(X, fields, axis=1, allow_missing=False):
    return np.stack(
        [
            ak.to_numpy(
                getattr(X,f),
                allow_missing = allow_missing,
            )
            for f in fields
        ],
        axis = axis,
    )


def recursive_tuple_to_list(obj):
    if isinstance(obj,(list,tuple)):
        return [recursive_tuple_to_list(o) for o in obj]
    elif isinstance(obj,dict):
        return {key:recursive_tuple_to_list(val) for key,val in obj.items()}
    elif isinstance(obj,(float,int,str)):
        return obj
    else:
        raise TypeError

def get_data_intersection_indices(datas,branch,different_files=False):
    """
        Return a list of indices for each data instance that intersect on the branch provided
        Args:
         - datas [list] : list of AbsData instances
         - branch [str] : data branch on which to check for intersection (eg, event number)
         - different_files [bool] : if True, will make sure to cross check the files
        Note : this search is performed file by file, assuming different trees
    """
    # Make recasting and checks #
    if not isinstance(datas,(list,tuple)):
        datas = [datas]
    assert len(datas) > 1, f'Need at least 2 Data objects, got {len(data)}'
    for i,data in enumerate(datas):
        assert isinstance(data,AbsData), f'Data entry number {i} (type : {type(data)}) is not a Data class'

    # Make metadata #
    metadatas = []
    for data in datas:
        metadata = {br: data[br].to_numpy() for br in ['file','tree','sample']}
        metadata['intersection'] = data[branch]
        metadatas.append(metadata)

    # Get the indices #
    return get_metadata_intersection_indices(metadatas,different_files)


def get_metadata_intersection_indices(metadatas,different_files):
    # Get matching between files
    print ('Looking into file metadata')
    if different_files:
        # The different trees from which the data are taken are from different files
        # Cannot make any checks here, rely on the fact the user provided them in the correct order
        common_files = [np.unique(metadata['file']) for metadata in metadatas]
        lengths = [len(files) for files in common_files]
        assert len(set(lengths))==1, f'The data objects have different number of files : {lengths}'
        print ('Will pair these files together :')
        for i in range(lengths[0]):
            print ('   - '+' <-> '.join([files[i] for files in common_files]))
    else:
        # Assume the files contain multiple trees that are extracted in the data objects
        # We want to only compare corresponding files, because the intersecton branch is probably not unique
        unique_files = None
        for i, metadata in enumerate(metadatas):
            uniq = np.unique(metadata['file'])
            print (f'\tentry {i} : {uniq}')
            if unique_files is None:
                unique_files = uniq
            else:
                unique_files = [f for f in unique_files if f in uniq]
        common_files = [unique_files] * len(metadatas)
        print (f'Will only consider common files : {unique_files}')
        print ('(Note : this assumes the files have the same order between the different metadata objects)')

    # Obtain set of indices for each metadata object that intersect on the branch #
    idxs = [np.array([],dtype=np.int64) for _ in range(len(metadatas))]
    sizes = [0 for _ in range(len(metadatas))] # Need to avoid resetting indices to zero
    for i in range(len(common_files[0])):
        arrays = [metadata['intersection'][metadata['file']==files[i]] for metadata,files in zip(metadatas,common_files)]
        matched = reduce(np.intersect1d, arrays) # find common values between all arrays
        for j in range(len(metadatas)):
            idx = np.nonzero(np.in1d(arrays[j],matched))[0] # for specific array, get common indices with matched
            idx += sizes[j]
            idxs[j] = np.concatenate((idxs[j],idx),axis = 0)
            sizes[j] += len(arrays[j])  # keep track to add to next iteration

    # Info printout #
    for i in range(len(metadatas)):
        print (f'For entry {i} : from {len(metadatas[i]["file"])} events, {len(idxs[i])} selected')

    # Safety check : make sure the intersection branch returns the same values
    for i in range(1,len(metadatas)):
        if not all(metadatas[0]['intersection'][idxs[0]] == metadatas[i]['intersection'][idxs[i]]):
            raise RuntimeError(f'Disagreement between metadata object 0 and {i} on branch `intersection` after the index selection')

    return idxs
