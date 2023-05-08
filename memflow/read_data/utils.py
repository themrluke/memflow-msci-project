import torch
import numpy as np
import numba
from numba import njit
import awkward as ak
import vector
vector.register_awkward()
vector.register_numba()

import numpy.ma as ma
from torch.utils.data import DataLoader


struct_jets = ak.zip({"pt": np.float32(0),
                      "eta": np.float32(0),
                      "phi": np.float32(0),
                      "btag": np.float32(0),
                      "m": np.float64(0),
                      "matched": bool(0),
                      "prov": -1},
                     with_name='Momentum4D')

struct_partons = ak.zip({"pt": np.float32(0),
                         "eta": np.float32(0),
                         "phi": np.float32(0),
                         "mass": np.float64(0),
                         "pdgId": bool(0),
                         "prov": -1},
                        with_name='Momentum4D')

struct_gluon = ak.zip({"pt": np.float32(1e-7),
                         "eta": np.float32(0.),
                         "phi": np.float32(0.),
                         "mass": np.float64(1e-7),
                         "pdgId": bool(0),
                         "prov": -1},
                        with_name='Momentum4D')



@njit
def sum_vectors_all(arrays):
    size = len(arrays)
    results = np.zeros((size, 4))
    for i, array in enumerate(arrays):
        total = vector.obj(px=0.0, py=0.0, pz=0.0, E=0.0)
        for v in array:
            total = total + v
        results[i, 0] = total.px
        results[i, 1] = total.py
        results[i, 2] = total.pz
        results[i, 3] = total.E
    return results


def get_vector_sum(vectors):
    out = sum_vectors_all(vectors)
    return vector.awk({
        "px": out[:, 0],
        "py": out[:, 1],
        "pz": out[:, 2],
        "E": out[:, 3],
    })


def to_flat_numpy(X, fields, axis=1, allow_missing=False):
    return np.stack([ak.to_numpy(getattr(X,f), allow_missing=allow_missing) for f in fields], axis=axis)


def to_flat_tensor(X, fields, axis=1, allow_missing=False):
    return torch.tensor(np.stack([ak.to_numpy(getattr(X,f), allow_missing=allow_missing) for f in fields], axis=axis))


class Test():
    def __init__(self):
        super().__init__()

    # compute px and py(for every frame - should be equal to 0)
    # input = batches (jet, lepton, met)
    def check_px_py(self, batches):

        no_batches = len(batches)

        for i, batch in enumerate(batches):

            # for px
            batch_px = self.check_px_per_batch(batch)
            if i == 0:
                objects_px_per_event = batch_px
            else:
                objects_px_per_event = torch.cat(
                    (objects_px_per_event, batch_px), dim=1)

            # for py
            batch_py = self.check_py_per_batch(batch)
            if i == 0:
                objects_py_per_event = batch_py
            else:
                objects_py_per_event = torch.cat(
                    (objects_py_per_event, batch_py), dim=1)

        sum_px_per_event = torch.sum(objects_px_per_event, dim=1)
        sum_py_per_event = torch.sum(objects_py_per_event, dim=1)

        return sum_px_per_event, sum_py_per_event

    # compute pz (for CM frame - should be equal to 0)
    # input = batches (jet, lepton, met)
    def check_pz(self, batches):

        for i, batch in enumerate(batches):

            # for pz
            batch_pz = self.check_pz_per_batch(batch)
            if i == 0:
                objects_pz_per_event = batch_pz
            else:
                objects_pz_per_event = torch.cat(
                    (objects_pz_per_event, batch_pz), dim=1)

        sum_pz_per_event = torch.sum(objects_pz_per_event, dim=1)

        return sum_pz_per_event

    def check_px_per_batch(self, batch):
        batch_px = np.cos(batch[:, :, 2]) * batch[:, :, 0]
        if (batch.size(1) > 1):
            batch_px = torch.sum(batch_px, dim=1)
            batch_px = torch.unsqueeze(batch_px, 1)

        return batch_px

    def check_py_per_batch(self, batch):
        batch_py = np.sin(batch[:, :, 2]) * batch[:, :, 0]
        if (batch.size(1) > 1):
            batch_py = torch.sum(batch_py, dim=1)
            batch_py = torch.unsqueeze(batch_py, 1)

        return batch_py

    def check_pz_per_batch(self, batch):
        batch_pz = np.sinh(batch[:, :, 1]) * batch[:, :, 0]
        if (batch.size(1) > 1):
            batch_pz = torch.sum(batch_pz, dim=1)
            batch_pz = torch.unsqueeze(batch_pz, 1)

        return batch_pz
