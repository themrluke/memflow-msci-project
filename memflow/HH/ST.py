import os
import torch
import awkward as ak
import numpy as np
from hepunits.units import MeV, GeV
from sklearn import preprocessing

from memflow.HH.base import HardBase, RecoDoubleLepton

class STPlusDoubleLeptonHardDataset(HardBase):
    @property
    def initial_states_pdgid(self):
        return [21,21]

    @property
    def final_states_pdgid(self):
        return [6,-6,11,-12,-11,12]

    @property
    def final_states_object_name(self):
        return None
        #return ['b','bbar','l-','vbar','l+','v']

    @property
    def processed_path(self):
        return os.path.join(
            self.build_dir,
            'st_plus_hard'
        )

    def process(self):
        # Safety checks #
        # Note : in ST samples the top is not required to decay to bottom quarks
        # In rare cases it decays to charm quarks (even more rare into up, if any)
        self.check_quantities(
            {
                'top' : 1,
                'bottom' : [0,1],
                'W_plus_from_top' : 1,
                (
                    'lep_plus_from_top',
                    'neutrino_from_top',
                ) : [0,2],
                (
                    'quark_up_from_top',
                    'antiquark_down_from_top',
                ) : [0,2],
                (
                    'lep_plus_from_top',
                    'neutrino_from_top',
                    'quark_up_from_top',
                    'antiquark_down_from_top',
                ) : 2,
                'W_minus_prompt': 1,
                (
                    'lep_minus_from_prompt_W',
                    'antineutrino_from_prompt_W',
                ) : [0,2],
                (
                    'quark_down_from_prompt_W',
                    'antiquark_up_from_prompt_W',
                ) : [0,2],
                (
                    'lep_minus_from_prompt_W',
                    'antineutrino_from_prompt_W',
                    'quark_down_from_prompt_W',
                    'antiquark_up_from_prompt_W',
                ) : 2,
            },
            verbose=True,
        )

        # Apply the cuts to select fully leptonic case #
        self.select_present_particles(
            [
                'top',
                'bottom',
                'W_plus_from_top',
                'lep_plus_from_top',
                'neutrino_from_top',
                'W_minus_prompt',
                'lep_minus_from_prompt_W',
                'antineutrino_from_prompt_W',
            ]
        )

        # For now exclude the tau decays of the Ws #
        #mask_tau_veto = np.logical_and(
        #    self.data['lep_plus_from_top_pdgId'] != -15,
        #    self.data['lep_minus_from_prompt_W_pdgId'] != +15,
        #)
        #print (f'From {len(mask_tau_veto)}, removing {sum(~mask_tau_veto)} tau decay events')
        #self.data.cut(mask_tau_veto)


        # Make generator info #
        boost = self.make_boost(
            self.data['Generator_x1'],
            self.data['Generator_x2'],
        )

        # Register ISR #
        # Need to be done before final states if a cut on N(ISR) is done
        self.register_ISR()

        # Make particles #
        self.data.make_particles(
            'final_states',
            {
                'px'  : [
                    'bottom_Px',
                    'lep_plus_from_top_Px',
                    'neutrino_from_top_Px',
                    'lep_minus_from_prompt_W_Px',
                    'antineutrino_from_prompt_W_Px',
                ],
                'py'  : [
                    'bottom_Py',
                    'lep_plus_from_top_Py',
                    'neutrino_from_top_Py',
                    'lep_minus_from_prompt_W_Py',
                    'antineutrino_from_prompt_W_Py',
                ],
                'pz'  : [
                    'bottom_Pz',
                    'lep_plus_from_top_Pz',
                    'neutrino_from_top_Pz',
                    'lep_minus_from_prompt_W_Pz',
                    'antineutrino_from_prompt_W_Pz',
                ],
                'E'  : [
                    'bottom_E',
                    'lep_plus_from_top_E',
                    'neutrino_from_top_E',
                    'lep_minus_from_prompt_W_E',
                    'antineutrino_from_prompt_W_E',
                ],
                'pdgId'  : [
                    'bottom_pdgId',
                    'lep_plus_from_top_pdgId',
                    'neutrino_from_top_pdgId',
                    'lep_minus_from_prompt_W_pdgId',
                    'antineutrino_from_prompt_W_pdgId',
                ],
            },
            lambda vec: vec.E > 0.,
        )
        self.register_particles(['final_states'])

        self.match_coordinates(boost,self.data['final_states']) # need to be done after the boost

class STMinusDoubleLeptonHardDataset(HardBase):
    @property
    def initial_states_pdgid(self):
        return [21,21]

    @property
    def final_states_pdgid(self):
        return [6,-6,11,-12,-11,12]

    @property
    def final_states_object_name(self):
        return None
        #return ['b','bbar','l-','vbar','l+','v']

    @property
    def processed_path(self):
        return os.path.join(
            self.build_dir,
            'st_minus_hard'
        )

    def process(self):
        # Safety checks #
        # Note : in ST samples the top is not required to decay to bottom quarks
        # In rare cases it decays to charm quarks (even more rare into up, if any)
        self.check_quantities(
            {
                'antitop' : 1,
                'antibottom' : [0,1],
                'W_minus_from_antitop' : 1,
                (
                    'lep_minus_from_antitop',
                    'antineutrino_from_antitop',
                ) : [0,2],
                (
                    'quark_down_from_antitop',
                    'antiquark_up_from_antitop',
                ) : [0,2],
                (
                    'lep_minus_from_antitop',
                    'antineutrino_from_antitop',
                    'quark_down_from_antitop',
                    'antiquark_up_from_antitop',
                ) : 2,
                'W_plus_prompt': 1,
                (
                    'lep_plus_from_prompt_W',
                    'neutrino_from_prompt_W',
                ) : [0,2],
                (
                    'quark_up_from_prompt_W',
                    'antiquark_down_from_prompt_W',
                ) : [0,2],
                (
                    'lep_plus_from_prompt_W',
                    'neutrino_from_prompt_W',
                    'quark_up_from_prompt_W',
                    'antiquark_down_from_prompt_W',
                ) : 2,
            },
            verbose=True,
        )

        # Apply the cuts to select fully leptonic case #
        self.select_present_particles(
            [
                'antitop',
                'antibottom',
                'W_minus_from_antitop',
                'lep_minus_from_antitop',
                'antineutrino_from_antitop',
                'W_plus_prompt',
                'lep_plus_from_prompt_W',
                'neutrino_from_prompt_W',
            ]
        )

        # For now exclude the tau decays of the Ws #
        #mask_tau_veto = np.logical_and(
        #    self.data['lep_minus_from_antitop_pdgId'] != -15,
        #    self.data['lep_plus_from_prompt_W_pdgId'] != +15,
        #)
        #print (f'From {len(mask_tau_veto)}, removing {sum(~mask_tau_veto)} tau decay events')
        #self.data.cut(mask_tau_veto)


        # Make generator info #
        boost = self.make_boost(
            self.data['Generator_x1'],
            self.data['Generator_x2'],
        )

        # Register ISR #
        # Need to be done before final states if a cut on N(ISR) is done
        self.register_ISR()

        # Make particles #
        self.data.make_particles(
            'final_states',
            {
                'px'  : [
                    'antibottom_Px',
                    'lep_minus_from_antitop_Px',
                    'antineutrino_from_antitop_Px',
                    'lep_plus_from_prompt_W_Px',
                    'neutrino_from_prompt_W_Px',
                ],
                'py'  : [
                    'antibottom_Py',
                    'lep_minus_from_antitop_Py',
                    'antineutrino_from_antitop_Py',
                    'lep_plus_from_prompt_W_Py',
                    'neutrino_from_prompt_W_Py',
                ],
                'pz'  : [
                    'antibottom_Pz',
                    'lep_minus_from_antitop_Pz',
                    'antineutrino_from_antitop_Pz',
                    'lep_plus_from_prompt_W_Pz',
                    'neutrino_from_prompt_W_Pz',
                ],
                'E'  : [
                    'antibottom_E',
                    'lep_minus_from_antitop_E',
                    'antineutrino_from_antitop_E',
                    'lep_plus_from_prompt_W_E',
                    'neutrino_from_prompt_W_E',
                ],
                'pdgId'  : [
                    'antibottom_pdgId',
                    'lep_minus_from_antitop_pdgId',
                    'antineutrino_from_antitop_pdgId',
                    'lep_plus_from_prompt_W_pdgId',
                    'neutrino_from_prompt_W_pdgId',
                ],
            },
            lambda vec: vec.E > 0.,
        )
        self.register_particles(['final_states'])

        self.match_coordinates(boost,self.data['final_states']) # need to be done after the boost


class STPlusDoubleLeptonRecoDataset(RecoDoubleLepton):
    @property
    def processed_path(self):
        return os.path.join(
            self.build_dir,
            'st_plus_reco',
        )


class STMinusDoubleLeptonRecoDataset(RecoDoubleLepton):
    @property
    def processed_path(self):
        return os.path.join(
            self.build_dir,
            'st_minus_reco',
        )

