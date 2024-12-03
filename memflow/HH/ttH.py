import os
import torch
import awkward as ak
import numpy as np
from hepunits.units import MeV, GeV
from sklearn import preprocessing

from memflow.dataset.dataset import AbsDataset, HardDataset, RecoDataset
from memflow.HH.base import Base
from memflow.dataset.preprocessing import (
    lowercutshift,
    logmodulus,
    SklearnScaler,
    PreprocessingPipeline,
    PreprocessingStep,
)
import vector


class ttHHardDataset(Base, HardDataset):

    """
    Base takes in arguments:
        coordinates = 'cartesian' or 'cylindrical'
        apply_preprocessing = True or False
        apply_boost = True or False

    HardDataset(AbsDataset(Dataset,metaclass=ABCMeta))
    AbsDataset:
        Abstract class for torch dataset, inherited by the Hard and Reco datasets
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

    def __init__(self, is_variable_length=True, **kwargs):
        super().__init__(**kwargs)
        self.is_variable_length = is_variable_length  # Store this setting to use later

    @property
    def initial_states_pdgid(self):
        return [21, 21]  # Initial particles (gluons)

    @property
    def final_states_pdgid(self):
        # Assuming the final states of the ttH process are t, tbar, H
        return [6, -6, 25]  # PDG IDs for top, anti-top, Higgs

    @property
    def final_states_object_name(self):
        # Define the final objects (Higgs, tops)
        return ["top", "antitop", "higgs"]

    @property
    def processed_path(self):
        print("Processed path method called.")  # Debug print
        # Directory to save processed ttH data
        return os.path.join(os.getcwd(), 'ttH_hard')
    
    @property
    def energy(self): # DO I NEED THIS BIT
        # Return the energy of the center-of-mass in GeV for the ttH process
        return 13000 * GeV  # Replace 13000 GeV with the actual value if different

    @staticmethod
    def polar_to_cartesian(pt, eta, phi, mass):
        # Use the vector package to convert polar coordinates to Cartesian
        vec = vector.obj(pt=pt, eta=eta, phi=phi, mass=mass)
        return vec.px, vec.py, vec.pz, vec.E
    
    def process(self):
        print("Process method invoked.")  # Debug print
        # Set the `is_variable_length` attribute on `self.data` once it’s initialized
        self.data.is_variable_length = self.is_variable_length
        self.events = len(self.data)  # Ensure this corresponds to the number of events
        #Select relevant objects for ttH events
        self.select_objects([ # Check names here actually exist
            "top",
            "antitop",
            "higgs",
            "bquark_top",
            "bquark_antitop",
            "w_lepton_top",
            "w_neutrino_top",
            "w_lepton_antitop",
            "w_neutrino_antitop"
        ])

        # Define random boost values for particles
        x1 = np.random.random((self.data.events, 1))
        x2 = np.random.random((self.data.events, 1))
        boost = self.make_boost(x1, x2)

        # Register particles
        self.data.make_particles(
            'higgs',
            {
                'pt'  : [
                    'higgs_pt',
                ],
                'eta'  : [
                    'higgs_eta',
                ],
                'phi'  : [
                    'higgs_phi',
                ],
                'mass'  : [
                    'higgs_mass',
                ],
                'pdgid'  : [
                    'higgs_pdgId',
                ],
            },
        )
        self.data.make_particles(
            'tops',
            {
                'pt'  : [
                    'top_pt',
                    'antitop_pt',
                ],
                'eta'  : [
                    'top_eta',
                    'antitop_eta',
                ],
                'phi'  : [
                    'top_phi',
                    'antitop_phi',
                ],
                'mass'  : [
                    'top_mass',
                    'antitop_mass',
                ],
                'pdgid'  : [
                    'top_pdgId',
                    'antitop_pdgId',
                ],
            },
        )
        self.data.make_particles(
            'bottoms',
            {
                'pt'  : [
                    'bottom_pt',
                    'antibottom_pt',
                ],
                'eta'  : [
                    'bottom_eta',
                    'antibottom_eta',
                ],
                'phi'  : [
                    'bottom_phi',
                    'antibottom_phi',
                ],
                'mass'  : [
                    'bottom_mass',
                    'antibottom_mass',
                ],
                'pdgid'  : [
                    'bottom_pdgId',
                    'antibottom_pdgId',
                ],
            },
        )
        self.data.make_particles(
            'Ws',
            {
                'pt'  : [
                    'W_plus_from_top_pt',
                    'W_minus_from_antitop_pt',
                ],
                'eta'  : [
                    'W_plus_from_top_eta',
                    'W_minus_from_antitop_eta',
                ],
                'phi'  : [
                    'W_plus_from_top_phi',
                    'W_minus_from_antitop_phi',
                ],
                'mass'  : [
                    'W_plus_from_top_mass',
                    'W_minus_from_antitop_mass',
                ],
                'pdgid'  : [
                    'W_plus_from_top_pdgId',
                    'W_minus_from_antitop_pdgId',
                ],
            },
        )
        self.data.make_particles(
            'quarks',
            {
                'pt'  : [
                    'quark_from_W_plus_pt',
                    'antiquark_from_W_plus_pt',
                    'quark_from_W_minus_pt',
                    'antiquark_from_W_minus_pt',
                ],
                'eta'  : [
                    'quark_from_W_plus_eta',
                    'antiquark_from_W_plus_eta',
                    'quark_from_W_minus_eta',
                    'antiquark_from_W_minus_eta',
                ],
                'phi'  : [
                    'quark_from_W_plus_phi',
                    'antiquark_from_W_plus_phi',
                    'quark_from_W_minus_phi',
                    'antiquark_from_W_minus_phi',
                ],
                'mass'  : [
                    'quark_from_W_plus_mass',
                    'antiquark_from_W_plus_mass',
                    'quark_from_W_minus_mass',
                    'antiquark_from_W_minus_mass',
                ],
                'pdgid'  : [
                    'quark_from_W_plus_pdgId',
                    'antiquark_from_W_plus_pdgId',
                    'quark_from_W_minus_pdgId',
                    'antiquark_from_W_minus_pdgId',
                ],
            },
            pad_value = 0.,
        )
        self.data.make_particles(
            'Zs',
            {
                'pt'  : [
                    'Z_from_higgs_pt',
                ],
                'eta'  : [
                    'Z_from_higgs_eta',
                ],
                'phi'  : [
                    'Z_from_higgs_phi',
                ],
                'mass'  : [
                    'Z_from_higgs_mass',
                ],
                'pdgid'  : [
                    'Z_from_higgs_pdgId',
                ],
            },
        )
        self.data.make_particles(
            'neutrinos',
            {
                'pt'  : [
                    'neutrinos_from_Z_pt',
                ],
                'eta'  : [
                    'neutrinos_from_Z_eta',
                ],
                'phi'  : [
                    'neutrinos_from_Z_phi',
                ],
                'mass'  : [
                    'neutrinos_from_Z_mass',
                ],
                'pdgid'  : [
                    'neutrinos_from_Z_pdgId',
                ],
            },
        )

        # Determine fields based on chosen coordinate system
        if self.coordinates == 'cylindrical':
            fields = ['pt', 'eta', 'phi', 'mass', 'pdgid']
        elif self.coordinates == 'cartesian':
            fields = ['px', 'py', 'pz', 'E', 'pdgid']
        else:
            raise ValueError("Invalid coordinate system specified. Choose either 'cylindrical' or 'cartesian'.")

        # Define particles to be registered for ttH
        for name in ['higgs', 'tops', 'bottoms', 'Ws', 'quarks', 'Zs', 'neutrinos']:
            obj = self.data[name]
            if self.coordinates == 'cartesian':
                obj = self.polar_to_cartesian(
                    obj['pt'], obj['eta'], obj['phi'], obj['mass']
                )
            obj = self.change_coordinates(obj)
            
            # Apply boost if specified
            if self.apply_boost:
                obj = self.boost(obj, boost)
            
            # Pad and mask particles to ensure consistent data shape
            obj_padded, obj_mask = self.reshape(
                input=obj,
                value=0.0,
                ax=1
            )
            
            # Register the object with its padded data and mask
            self.register_object(
                name=name,
                obj=obj_padded,
                mask=obj_mask,
                fields=fields
            )

        # Match coordinate systems between the boost and particles
        self.match_coordinates(boost, obj)

        # Register the boost object
        self.register_object(
            name='boost',
            obj=boost
        )

       # Convert and register Cartesian coordinates for Rambo phase space (ME_fields)
        ME_fields = ['E', 'px', 'py', 'pz']

        # Convert and register specific particles in Cartesian coordinates if needed
        if self.coordinates == 'cartesian':
            # Higgs particle
            higgs_px, higgs_py, higgs_pz, higgs_E = self.polar_to_cartesian(
                self.data['higgs'][:, 0]['pt'],
                self.data['higgs'][:, 0]['eta'],
                self.data['higgs'][:, 0]['phi'],
                self.data['higgs'][:, 0]['mass']
            )
            self.register_object(
                name='higgs',
                obj={'E': higgs_E, 'px': higgs_px, 'py': higgs_py, 'pz': higgs_pz},
                fields=ME_fields
            )

            # Top particle
            top_px, top_py, top_pz, top_E = self.polar_to_cartesian(
                self.data['tops'][:, 0]['pt'],
                self.data['tops'][:, 0]['eta'],
                self.data['tops'][:, 0]['phi'],
                self.data['tops'][:, 0]['mass']
            )
            self.register_object(
                name='top',
                obj={'E': top_E, 'px': top_px, 'py': top_py, 'pz': top_pz},
                fields=ME_fields
            )

            # Antitop particle
            antitop_px, antitop_py, antitop_pz, antitop_E = self.polar_to_cartesian(
                self.data['tops'][:, 1]['pt'],
                self.data['tops'][:, 1]['eta'],
                self.data['tops'][:, 1]['phi'],
                self.data['tops'][:, 1]['mass']
            )
            self.register_object(
                name='antitop',
                obj={'E': antitop_E, 'px': antitop_px, 'py': antitop_py, 'pz': antitop_pz},
                fields=ME_fields
            )

            # Additional particles (b-quarks and W bosons)
            bquark_top_px, bquark_top_py, bquark_top_pz, bquark_top_E = self.polar_to_cartesian(
                self.data['bottoms'][:, 0]['pt'],
                self.data['bottoms'][:, 0]['eta'],
                self.data['bottoms'][:, 0]['phi'],
                self.data['bottoms'][:, 0]['mass']
            )
            self.register_object(
                name='bquark_top',
                obj={'E': bquark_top_E, 'px': bquark_top_px, 'py': bquark_top_py, 'pz': bquark_top_pz},
                fields=ME_fields
            )

            bquark_antitop_px, bquark_antitop_py, bquark_antitop_pz, bquark_antitop_E = self.polar_to_cartesian(
                self.data['bottoms'][:, 1]['pt'],
                self.data['bottoms'][:, 1]['eta'],
                self.data['bottoms'][:, 1]['phi'],
                self.data['bottoms'][:, 1]['mass']
            )
            self.register_object(
                name='bquark_antitop',
                obj={'E': bquark_antitop_E, 'px': bquark_antitop_px, 'py': bquark_antitop_py, 'pz': bquark_antitop_pz},
                fields=ME_fields
            )

            # W+ and W- bosons
            W_plus_px, W_plus_py, W_plus_pz, W_plus_E = self.polar_to_cartesian(
                self.data['Ws'][:, 0]['pt'],
                self.data['Ws'][:, 0]['eta'],
                self.data['Ws'][:, 0]['phi'],
                self.data['Ws'][:, 0]['mass']
            )
            self.register_object(
                name='W_plus',
                obj={'E': W_plus_E, 'px': W_plus_px, 'py': W_plus_py, 'pz': W_plus_pz},
                fields=ME_fields
            )

            W_minus_px, W_minus_py, W_minus_pz, W_minus_E = self.polar_to_cartesian(
                self.data['Ws'][:, 1]['pt'],
                self.data['Ws'][:, 1]['eta'],
                self.data['Ws'][:, 1]['phi'],
                self.data['Ws'][:, 1]['mass']
            )
            self.register_object(
                name='W_minus',
                obj={'E': W_minus_E, 'px': W_minus_px, 'py': W_minus_py, 'pz': W_minus_pz},
                fields=ME_fields
            )


        # Apply preprocessing if enabled
        if self.apply_preprocessing:
            if self.coordinates == 'cylindrical':
                # Register preprocessing steps for cylindrical coordinates
                self.register_preprocessing_step(
                    PreprocessingStep(
                        names = ['higgs', 'tops', 'bottoms', 'Ws'],
                        scaler_dict = {
                            'pt': logmodulus(),
                        },
                    )
                )
                self.register_preprocessing_step(
                    PreprocessingStep(
                        names = ['higgs', 'tops', 'bottoms', 'Ws'],
                        scaler_dict={
                            'pt': SklearnScaler(preprocessing.StandardScaler()),
                            'eta': SklearnScaler(preprocessing.StandardScaler()),
                            'phi': SklearnScaler(preprocessing.StandardScaler()),
                            'm': SklearnScaler(preprocessing.StandardScaler()),
                        },
                    )
                )
            elif self.coordinates == 'cartesian':
                raise NotImplementedError("Cartesian coordinates preprocessing is not implemented.")


class ttHRecoDataset(Base, RecoDataset):
    @property
    def processed_path(self):
        return os.path.join(
            os.getcwd(),
            'ttH_reco',  # Adjusted directory name for ttH-specific processed data
        )

    @property
    def energy(self): # DO I NEED THIS BIT
        # Return the energy of the center-of-mass in GeV for the ttH process
        return 13000 * GeV  # Replace 13000 GeV with the actual value if different

    def process(self):
        # Select reconstructed objects specific to ttH events
        self.select_objects([
            'jets',  # Reconstructed jets from tops and W bosons
            'met',   # Missing transverse energy (MET), representing neutrinos
            'leptons'  # Leptons from W decays in ttH events
        ])

        # Create particles based on available reco-level branches
        self.data.make_particles(
            'jets',
            {
                'pt'      : 'cleanedJet_pt',
                'eta'     : 'cleanedJet_eta',
                'phi'     : 'cleanedJet_phi',
                'mass'    : 'cleanedJet_mass',
                'btag'    : 'cleanedJet_btagDeepFlavB',
            },

        )
        self.data.make_particles(
            'met',
            {
                'pt'      : 'InputMet_pt',
                'eta'     : 0.,
                'phi'     : 'InputMet_phi',
                'mass'    : 0.,
            },
        )
        self.data.make_particles(
            'leptons',
            {
                'pt': ['lep_pt'],
                'eta': ['lep_eta'],
                'phi': ['lep_phi'],
                'mass': ['lep_mass'],
            },
        )

        # Set up fields based on the selected coordinate system
        if self.coordinates == 'cylindrical':
            fields = ['pt', 'eta', 'phi', 'mass']
        elif self.coordinates == 'cartesian':
            fields = ['px', 'py', 'pz', 'E']
        else:
            raise ValueError("Invalid coordinate system specified. Choose either 'cylindrical' or 'cartesian'.")

        # Register particles with transformations
        for name in ['jets', 'met', 'leptons']:
            obj = self.data[name]
            obj = self.change_coordinates(obj)
            obj_padded, obj_mask = self.reshape(
                input=obj,
                value=0.0,
                ax=1
            )
            self.register_object(
                name=name,
                obj=obj_padded,
                mask=obj_mask,
                fields=fields
            )

        # Preprocessing steps for cylindrical coordinates if enabled
        if self.apply_preprocessing:
            if self.coordinates == 'cylindrical':
                self.register_preprocessing_step(
                    PreprocessingStep(
                        names=['jets', 'leptons', 'met'],
                        scaler_dict={
                            'pt': logmodulus(),
                        },
                    )
                )
                self.register_preprocessing_step(
                    PreprocessingStep(
                        names=['jets', 'leptons', 'met'],
                        scaler_dict={
                            'pt': SklearnScaler(preprocessing.StandardScaler()),
                            'eta': SklearnScaler(preprocessing.StandardScaler()),
                            'phi': SklearnScaler(preprocessing.StandardScaler()),
                            'mass': SklearnScaler(preprocessing.StandardScaler()),
                        },
                    )
                )
            elif self.coordinates == 'cartesian':
                raise NotImplementedError("Cartesian coordinates preprocessing is not implemented.")

