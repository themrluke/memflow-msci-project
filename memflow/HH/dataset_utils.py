import os

from memflow.dataset.data import RootData
from memflow.dataset.dataset import *

from memflow.HH.HH import *
from memflow.HH.ttbar import *
from memflow.HH.DY import *
from memflow.HH.ZZ import *
from memflow.HH.ZH import *
from memflow.HH.ST import *

def make_combined_dataset(files,treename,hard_cls,reco_cls,selection,build_dir,build=False,fit=True,N=None,n_ISR=None):
    if build:
        # Hard data #
        print ('Hard data loading')
        hard_data = RootData(
            files = files,
            treenames = treename,
            lazy = True,
            N = N,
        )
        print (f'\t... done : {hard_data.events} events')
        # Reco data #
        print ('Reco data loading')
        reco_data = RootData(
            files = files,
            treenames = ['reco_DL;1'],
            lazy = True,
            N = N,
        )
        print (f'\t... done : {reco_data.events} events')
    else:
        hard_data = None
        reco_data = None

    # Hard dataset #
    print ('Hard dataset')
    hard_dataset = hard_cls(
        data = hard_data,
        selection = selection,
        coordinates = 'cylindrical',
        apply_boost = False,
        apply_preprocessing = True,
        n_ISR = n_ISR,
        build = build,
        fit = fit,
        build_dir = build_dir,
        dtype = torch.float32,
    )
    print ('\t... done')
    print (hard_dataset)

    # Hard dataset #
    print ('Reco dataset')
    reco_dataset = reco_cls(
        data = reco_data,
        selection = [
            'muons',
            'electrons',
            'met',
            'jets',
        ],
        coordinates = 'cylindrical',
        topology = 'resolved',
        apply_boost = False,
        apply_preprocessing = True,
        default_features = {
            'pt': 0.,
            'eta': 0.,
            'phi': 0.,
            'mass': 0.,
            'btag' : 0.,
            'btagged': None,
            'pdgId' : 0.,
            'charge' : 0.
        },
        build = build,
        fit = fit,
        build_dir = build_dir,
        dtype = torch.float32,
    )
    print ('\t... done')
    print (reco_dataset)
    # Combined dataset #
    print ('Combined dataset')
    combined_dataset = CombinedDataset(
        hard_dataset = hard_dataset,
        reco_dataset = reco_dataset,
    )
    print ('\t... done')
    return combined_dataset


def select_dataset(suffix,build,fit,N):
    print ("="*100)
    print (f'Loading dataset for {suffix}')
    dirs = [
        '/home/ucl/cp3/fbury/scratch/MEM_data/Transfermer_v7_2016/results/'
    ]
    build_dir = '/nfs/scratch/fynu/fbury/MEMFlow_data/transfer_flow_v7'
    if suffix == 'HH':
        return make_combined_dataset(
            files = [
                os.path.join(dir,sample)
                for dir in dirs
                for sample in [
                    'GluGluToHHTo2B2VTo2L2Nu_node_cHHH0.root',
                    'GluGluToHHTo2B2VTo2L2Nu_node_cHHH1.root',
                    'GluGluToHHTo2B2VTo2L2Nu_node_cHHH2p45.root',
                    'GluGluToHHTo2B2VTo2L2Nu_node_cHHH5.root',
                ]
            ],
            treename = 'gen_HH;1',
            hard_cls = HHbbWWDoubleLeptonHardDataset,
            reco_cls = HHbbWWDoubleLeptonRecoDataset,
            selection = [
                'final_states',
                'ISR',
            ],
            n_ISR = 1,
            N = N,
            build = build,
            fit = fit,
            build_dir = build_dir,
        )
    elif suffix == 'TT':
        return make_combined_dataset(
            files = [
                os.path.join(dir,sample)
                for dir in dirs
                for sample in [
                    'TTTo2L2Nu.root',
                ]
            ],
            treename = 'gen_TT;1',
            hard_cls = TTDoubleLeptonHardDataset,
            reco_cls = TTDoubleLeptonRecoDataset,
            selection = [
                    'final_states',
                    'ISR',
                ],
            n_ISR = 1,
            N = N,
            build = build,
            fit = fit,
            build_dir = build_dir,
        )
    elif suffix == 'DY':
        return make_combined_dataset(
            files = [
                os.path.join(dir,sample)
                for dir in dirs
                for sample in [
                    'DYJetsToLL_M-10to50.root',
                    'DYJetsToLL_M-50.root',
                    #'DYToLL_0J.root',
                    'DYToLL_1J.root',
                    'DYToLL_2J.root',
                ]
            ],
            treename = 'gen_DY;1',
            hard_cls = DYDoubleLeptonHardDataset,
            reco_cls = DYDoubleLeptonRecoDataset,
            selection = [
                'final_states',
                'ISR',
            ],
            N = N,
            n_ISR = 2, # TODO
            build = build,
            fit = fit,
            build_dir = build_dir,
        )
    elif suffix == 'ZZ':
        return make_combined_dataset(
            files = [
                os.path.join(dir,sample)
                for dir in dirs
                for sample in [
                    'ZZTo2L2Q.root',
                ]
            ],
            treename = 'gen_ZZ;1',
            hard_cls = ZZDoubleLeptonHardDataset,
            reco_cls = ZZDoubleLeptonRecoDataset,
            selection = [
                'final_states',
            ],
            N = N,
            n_ISR = 0,
            build = build,
            fit = fit,
            build_dir = build_dir,
        )
    elif suffix == 'ZH':
        return make_combined_dataset(
            files = [
                os.path.join(dir,sample)
                for dir in dirs
                for sample in [
                    'ZH_HToBB_ZToLL.root',
                ]
            ],
            treename = 'gen_ZH;1',
            hard_cls = ZHDoubleLeptonHardDataset,
            reco_cls = ZHDoubleLeptonRecoDataset,
            selection = [
                'final_states',
                'ISR',
            ],
            N = N,
            n_ISR = 2,
            build = build,
            fit = fit,
            build_dir = build_dir,
        )
    elif suffix == 'STplus':
        return make_combined_dataset(
            files = [
                os.path.join(dir,sample)
                for dir in dirs
                for sample in [
                    'ST_tW_top_5f.root',
                ]
            ],
            treename = 'gen_ST;1',
            hard_cls = STPlusDoubleLeptonHardDataset,
            reco_cls = STPlusDoubleLeptonRecoDataset,
            selection = [
                'final_states',
                'ISR',
            ],
            N = N,
            n_ISR = 1,
            build = build,
            fit = fit,
            build_dir = build_dir,
        )
    elif suffix == 'STminus':
        return make_combined_dataset(
            files = [
                os.path.join(dir,sample)
                for dir in dirs
                for sample in [
                    'ST_tW_antitop_5f.root',
                ]
            ],
            treename = 'gen_ST;1',
            hard_cls = STMinusDoubleLeptonHardDataset,
            reco_cls = STMinusDoubleLeptonRecoDataset,
            selection = [
                'final_states',
                'ISR',
            ],
            N = N,
            n_ISR = 1,
            build = build,
            fit = fit,
            build_dir = build_dir,
        )
    else:
        raise RuntimeError(f'Unknown suffix {suffix}')


