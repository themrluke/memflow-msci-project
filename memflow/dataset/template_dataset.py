import os
import torch
import awkward as ak
import numpy as np
from hepunits.units import MeV, GeV
from sklearn import preprocessing

from memflow.dataset.dataset import HardDataset, RecoDataset
from memflow.dataset.preprocessing import *


class TemplateHardDataset(HardDataset):
    # This is a template class for hard-scattering dataset
    # Will highlight below the all the required methods and properties
    # Some clues will be given on helper methods
    def __init__(self,foo,bar**kwargs):
        # The kwargs are the args required by the base HardDataset class
        # (data,selection,default_features,build,device,dtype)
        # Any user requested argument can also be defined in the __init__ args

        # Here save in self any argument you need to use further
        self.foo = foo
        self.bar = bar

        # Call abstract class #
        super().__init__(**kwargs)


    # The process method is the one where the data object is treated
    # Objects from the awkward arrays are registered and the preprocessing
    def process(self):
        # 1) : select the decay you are interested in for your process
        # -> make mask (data type dependent) and apply it
        self.data.cut(mask)

        # 2) : obtain boost from the x1 and x2 parton fractions
        # Note : this is only needed in the case of the ME integration, not the transfer-flow
        x1 = self.data[<branch_x1>]
        x2 = self.data[<branch_x2>]
        boost = self.make_boost(x1,x2)
        # Note : for the ME case, you need to register the boost as it is used in rambo
        # (see below to register)

        # 3) : Make the particles
        # Obtain the awkward arrays with Momentum4D to make your particles
        self.data.make_particles(
            <name>,
            <dict>,
            ...
        )

        # In case you want to boost your objects to the CM, see below
        <obj_boosted> = self.boost(<obj>,boost)

        # 4) : Register the objecs to do awkward->rectangular array->tensor
        self.register_object(
            name = <name>,
            obj = <awkward_array>,
            fields = <list_of_fields>,
        )
        # The <name> is the object name that will be used to identify the objects:
        #   - in the `selection` when you instantiate the class
        #   - in the preprocessing, to apply the correct functions to the objects
        # The <awkward_array> must be a Momentum4D type of array
        # The <list_of_fieds> is the list of fields from the awkward array to save in the tensor
        # this is necessary to keep track of the features in the transformer
        # typically this will be ['pt','eta','phi','mass',('pdgId')]
        # note: the fields can be from the array directly, or attributes of vector

        # In case you expect a variable number of particles (not always the case)
        # <mask> can be obtained when reshaping the awkward array as below
        # This is not always needed, for example when you know you have the same number of that particle per event

        <array_reshaped>, <array_mask> = self.reshape(
            input = <awkward_array>,
            value = <padding_value>, # number used to pad the missing entries (0 typically)
            max_no = <value/None> # maximum number of particles in the padded array
            # if max_no is None, will use the maximum number of particles in all events
        )
        self.register_object(
            name = <name>,
            obj = <array_reshaped>,
            fields = <list_of_fields>,
            mask = <array_mask>,
        )

        # Note, you can also include a weight in the object, needs to be better described

        # 5) : Reqister the preprocessing steps
        # You might want to apply multiple scaling in steps
        # Needs to be registered to be able to inverse all steps at the end
        self.register_preprocessing_step(
            PreprocessingStep(
                names = <list_of_names>,
                scaler_dict = <scaler_dict>,
                fields_select = <list_fields>,
            )
        )
        # <list_of_names> is the list of registered names in step 4
        # you can either
        #   - include multiple objects,
        #     -> preprocessing will be fit (eg using a sklearn) scaler on all the objects
        #   - register preprocessing for particles individually
        #     -> each will be fit for eac particle independently
        # <scaler_dict> : dict of
        #   - key : field as saved in the register objects
        #     -> apply different preprocessing for each feature
        #   - value : a class defined in memflow.dataset.preprocessing
        #       (see below)
        # <fields_select> : list (same length as <list_of_names>) of fields to consider
        # this can be useful when for example you include leptons and met,
        # and want to preprocess pt for both, but only eta for leptons

        # Available preprocessing :
        # - logmodulus : apply sign(x) * log(1+|x|)
        # - lowercutshift : rescale by the cut value
        # - SklearnScaler : anything from the sklearn.preprocessing
        # - anything custom, see the logic in memflow.dataset.preprocessing


    # Properties #
    @property
    def attention_idx(self):
        # Need to return a dict (or None if not needed)
        # - key : name of a registered object
        # - value : indices to consider in the attention mask (even if not present)

    @property
    def processed_path(self):
        # return path of where to store the tensors in case build=True


    # Some properties are required for the ME integration
    # This is to be linked with Rambo
    # If you are not using it right now, you can just return None

    @property
    def initial_states_pdgid(self):
        # return the pdgids of the two initial state quarks as they are in the ME
        return None

    @property
    def final_states_pdgid(self):
        # return the pdgids of the process final states as they are in the ME
        return None

    @property
    def final_states_object_name(self):
        # names associated to the final states
        # these names will be fetched in the object registered in the process method
        # (see above)
        return None


class TemplateRecoDataset(RecoDataset):
    # Most of the parts are similar to the HardDataset
    # Will only illustrate the differences here for reco
    # In particular everything related to ME can be dismissed:
    # - initial_states_pdgid
    # - final_states_pdgid
    # - final_states_object_name

    # Properties #
    @property
    def attention_idx(self):
        # same as Hard

    @property
    def processed_path(self):
        # same as Hard


    def process(self):
        # Only difference is step 2)
        # Boost can be defined as a the total particle momentum
        boost = self.make_boost(jets,electrons,muons,met)


