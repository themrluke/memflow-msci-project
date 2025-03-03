import math
import inspect
import torch
import vector
import numpy as np
import awkward as ak
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import Callback

from abc import ABCMeta, abstractmethod

from transfer_flow.utils import *

vector.register_awkward()

class BaseCallback(Callback,metaclass=ABCMeta):
    def __init__(
            self,
            dataset,
            preprocessing = None,
            N_sample = 1,
            frequency = 1,
            bins = 50,
            hexbin = False,
            kde = False,
            log_scale = False,
            suffix = '',
            label_names = {},
            feature_rng = {},
            device = None,
    ):

        super().__init__()

        # Attributes #
        self.dataset = dataset
        self.N_sample = N_sample
        self.frequency = frequency
        self.preprocessing = preprocessing
        self.bins = bins
        self.hexbin = hexbin
        self.kde = kde
        self.log_scale = log_scale
        self.label_names = label_names
        self.device = device
        self.suffix = suffix
        self.label_names = label_names
        self.feature_rng = feature_rng


    def on_validation_epoch_end(self,trainer,pl_module):
        if trainer.sanity_checking:  # optional skip
            return
        if trainer.current_epoch == 0:
            return
        if trainer.current_epoch % self.frequency != 0:
           return

        # Get figures #
        figs = self.make_plots(pl_module,disable_tqdm=True,show=False)

        # Log them #
        for figure_name,figure in figs.items():
            trainer.logger.experiment.log_figure(
                figure_name = figure_name,
                figure = figure,
                overwrite = True,
                step = trainer.current_epoch,
            )
            plt.close(figure)


    def predict(self,model,disable_tqdm=False,only_flow_variables=True):
       # Get samples for whole dataset #
        if self.device is None:
            device = model.device
        else:
            device = self.device
        model = model.to(device)
        model.eval()

        # Predict #
        samples = []
        true   = []
        mask    = []
        for batch_idx, batch in tqdm(enumerate(self.loader),desc='Predict',disable=disable_tqdm,leave=True,total=min(self.N_batch,len(self.loader)),position=0):
            # Transfer batches #
            batch = model.transfer_batch_to_device(batch,model.device,batch_idx)
            if batch_idx >= self.N_batch:
                break

            # Sample #
            with torch.no_grad():
                batch_samples = [
                    sample.permute(1,0,2,3).cpu()
                    for sample in model.sample(
                        batch['hard']['data'],
                        batch['hard']['mask'],
                        batch['reco']['data'],
                        batch['reco']['mask'],
                        self.N_sample,
                    )
                ]

            # Select variables #
            batch = model.transfer_batch_to_device(batch,'cpu',batch_idx)
            reco_data = batch['reco']['data']
            reco_mask_exist = batch['reco']['mask']

            if only_flow_variables:
                batch_samples = [
                    batch_samples[i][...,model.flow_indices[i]]
                    for i in range(len(batch_samples))
                ]
                reco_data = [
                    reco_data[i][...,model.flow_indices[i]]
                    for i in range(len(reco_data))
                ]


            # Record #
            if len(samples) == 0:
                for i in range(len(batch_samples)):
                    samples.append([batch_samples[i]])
                    true.append([reco_data[i]])
                    mask.append([reco_mask_exist[i]])
            else:
                for i in range(len(batch_samples)):
                    samples[i].append(batch_samples[i])
                    true[i].append(reco_data[i])
                    mask[i].append(reco_mask_exist[i])

        # Concat the whole lists #
        samples = [torch.cat(sample,dim=0) for sample in samples]
        true    = [torch.cat(t,dim=0) for t in true]
        mask    = [torch.cat(m,dim=0) for m in mask]


        return true,mask,samples


    def undo_preprocessing(self,model,true,mask,samples,only_flow_variables=True):
        for i in range(len(true)):
            name = model.reco_names[i]
            fields = model.decoder_embeddings.features_per_type[i]
            if only_flow_variables:
                fields = [fields[idx] for idx in model.flow_indices[i]]
            true[i], _ = self.preprocessing.inverse(
                name = name,
                x = true[i],
                mask = mask[i],
                fields = fields,
            )
            # preprocessing expects :
            #   data = [events, particles, features]
            #   mask = [events, particles]
            # samples dims = [events, samples, particles, features]
            # will merge event*samples and unmerge later
            samples[i] = self.preprocessing.inverse(
                name = name,
                x = samples[i].reshape(
                    samples[i].shape[0]*self.N_sample,
                    samples[i].shape[2],
                    samples[i].shape[3],
                ),
                mask = mask[i].unsqueeze(1).repeat_interleave(self.N_sample,dim=1).reshape(mask[i].shape[0]*self.N_sample,mask[i].shape[1]),
                fields = fields,
            )[0].reshape(
                samples[i].shape[0],
                self.N_sample,
                samples[i].shape[2],
                samples[i].shape[3],
            )




class SamplingCallback(BaseCallback):
    def __init__(self,idx_to_monitor,**kwargs):
        super().__init__(**kwargs)

        # Call batch getting #
        self.set_idx(idx_to_monitor)

    def set_idx(self,idx_to_monitor):
        self.idx_to_monitor = idx_to_monitor
        # Checks #
        if not torch.is_tensor(self.idx_to_monitor):
            self.idx_to_monitor = torch.tensor(self.idx_to_monitor)
        if self.idx_to_monitor.dim() < 1:
            self.idx_to_monitor = self.idx_to_monitor.reshape(-1)
        self.N_event = self.idx_to_monitor.shape[0]

        # Get batch of data #
        self.batch = {
            'hard': {
                'data' : [],
                'mask' : [],
            },
            'reco': {
                'data' : [],
                'mask' : [],
            },
        }
        for idx in self.idx_to_monitor:
            entry = self.dataset[idx]
            self.batch['hard']['data'].append(entry['hard']['data'])
            self.batch['hard']['mask'].append(entry['hard']['mask'])
            self.batch['reco']['data'].append(entry['reco']['data'])
            self.batch['reco']['mask'].append(entry['reco']['mask'])

        from torch.utils.data._utils.collate import default_collate
        self.batch['hard']['data'] = default_collate(self.batch['hard']['data'])
        self.batch['hard']['mask'] = default_collate(self.batch['hard']['mask'])
        self.batch['reco']['data'] = default_collate(self.batch['reco']['data'])
        self.batch['reco']['mask'] = default_collate(self.batch['reco']['mask'])

    def plot_particle(self,sample,reco,features,title):
        # sample (N,F)
        # reco (F)

        labels = [
            self.label_names[feature] if feature in self.label_names.keys() else feature
            for feature in features
        ]
        fig,axs = pairplot(
            true = reco,
            sample = sample,
            features = features,
            labels = labels,
            bins = self.bins,
            title = title,
            feature_rng = self.feature_rng,
            hexbin = self.hexbin,
            kde = self.kde,
            log_scale = self.log_scale,
        )

        return fig

    def make_plots(self,model,show=False,**kwargs):
        # Select device #
        if self.device is None:
            device = model.device
        else:
            device = self.device
        model = model.to(device)
        model.eval()

        # Split batch #
        hard_data = [data.to(device) for data in self.batch['hard']['data']]
        hard_mask_exist = [mask.to(device) for mask in self.batch['hard']['mask']]
        reco_data = [data.to(device) for data in self.batch['reco']['data']]
        reco_mask_exist = [mask.to(device) for mask in self.batch['reco']['mask']]

        # Sample #
        with torch.no_grad():
            samples = model.sample(hard_data,hard_mask_exist,reco_data,reco_mask_exist,self.N_sample)

        # Put all on cpu #
        samples = [
            samples[i].permute(1,0,2,3)[...,model.flow_indices[i]].to('cpu')
            for i in range(len(samples))
        ]
        reco_data = [
            reco_data[i][...,model.flow_indices[i]].to("cpu")
            for i in range(len(reco_data))
        ]
        reco_mask_exist = [mask.to("cpu") for mask in reco_mask_exist]

        # Inverse preprocessing #
        if self.preprocessing is not None:
            self.undo_preprocessing(model,reco_data,reco_mask_exist,samples)

        # Loop over events #
        figs = {}
        for event in range(self.N_event):
            # Loop over types #
            for i in range(len(reco_data)):
                # Loop over particles #
                for j in range(reco_data[i].shape[1]):
                    if reco_mask_exist[i][event,j]:
                        fig = self.plot_particle(
                            sample = samples[i][event,:,j,:],
                            reco = reco_data[i][event,j,:],
                            features = model.flow.flow_features[i],
                            title = f'{model.reco_names[i]} #{j} (event #{event})',
                        )
                        if show:
                            plt.show()
                        figure_name = f'event_{event}_obj_{model.reco_names[i]}_{j}'
                        if len(self.suffix) > 0:
                            figure_name += f'_{self.suffix}'
                        figs[figure_name] = fig
        return figs


class BiasCallback(BaseCallback):
    def __init__(self,points=20,N_batch=math.inf,batch_size=1024,**kwargs):
        super().__init__(**kwargs)

        # Attributes #
        self.points = points
        self.loader = DataLoader(self.dataset,batch_size=batch_size,shuffle=False)
        self.N_batch = N_batch

    def plot_1D(self,true,samples,features,title):
        N = true.shape[-1]
        fig = plt.figure(figsize=(4.5*N,5))
        fig.suptitle(title,fontsize=16)

        # Make global gridspec #
        gs = GridSpec(1, N, width_ratios=[1]*N, wspace=0.6, bottom=0.3)
        labels = [
            self.label_names[feature] if feature in self.label_names.keys() else feature
            for feature in features
        ]

        for i in range(N):
            bins = get_bins(arrays=[true[...,i].ravel(),samples[...,i].ravel()],feature=features[i],N=self.bins)
            plot_ratio(
                fig = fig,
                gs = gs[i],
                true = true[...,i].ravel(),
                sample = samples[...,i].ravel(),
                bins = bins,
                density = True,
                label = labels[i],
                log_scale = self.log_scale,
            )
        return fig

    def plot_2D(self,true,samples,features,title):
        N = true.shape[-1]
        fig = plt.figure(figsize=(4.5*N,5))
        fig.suptitle(title,fontsize=16)

        # Repeat and ravel #
        true = true.unsqueeze(dim=1).repeat_interleave(repeats=samples.shape[1],dim=1)
        true = true.reshape(true.shape[0]*true.shape[1],true.shape[2])
        samples = samples.reshape(samples.shape[0]*samples.shape[1],samples.shape[2])

        # Make global gridspec #
        gs = GridSpec(1, N, width_ratios=[1]*N, wspace=0.6, bottom=0.3)
        labels = [
            self.label_names[feature] if feature in self.label_names.keys() else feature
            for feature in features
        ]

        for i in range(N):
            bins = get_bins(arrays=[true[...,i].ravel(),samples[...,i].ravel()],feature=features[i],N=self.bins)
            plot_2D_projections(
                fig = fig,
                gs = gs[i],
                true = true[:,i],
                sample = samples[:,i],
                bins = bins,
                hexbin = self.hexbin,
                label_x = labels[i],
                label_y = labels[i],
                log_scale = self.log_scale,
            )

        return fig

    def plot_bias(self,true,samples,features,title):
        N = true.shape[-1]
        fig,axs = plt.subplots(ncols=N,nrows=1,figsize=(4.5*N,4))
        plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,hspace=0.1,wspace=0.4)
        fig.suptitle(title,fontsize=16)

        # Compute diff #
        true = true.unsqueeze(dim=1).repeat_interleave(repeats=samples.shape[1],dim=1)
        diff = samples-true
        true = true.reshape(true.shape[0]*true.shape[1],true.shape[2])
        diff = diff.reshape(diff.shape[0]*diff.shape[1],diff.shape[2])

        # Deal with cyclic phi #
        if 'phi' in features:
            idx_phi = features.index('phi')
            diff[...,idx_phi] = (diff[...,idx_phi] + math.pi) % (2 * math.pi) - math.pi

        # Loop over features #
        labels = [
            self.label_names[feature] if feature in self.label_names.keys() else feature
            for feature in features
        ]

        for i in range(N):
            plot_diff(
                ax = axs[i],
                true = true[:,i].ravel(),
                diff = diff[:,i].ravel(),
                points = self.points,
                label = labels[i],
                relative = False,
            )
        return fig


    def plot_quantiles(self,true,samples,features,title):
        samples = samples.reshape(samples.shape[0]*samples.shape[1],samples.shape[2])
        labels = [
            self.label_names[feature] if feature in self.label_names.keys() else feature
            for feature in features
        ]

        fig,ax = plt.subplots(1,1,figsize=(6,5))
        plt.suptitle(title,fontsize=16)

        plot_quantiles(
            ax = ax,
            true = true,
            sample = samples,
            points = self.points,
            labels = labels,
        )

        return fig

    def make_plots(self,model,show=False,disable_tqdm=False):
        true,mask,samples = self.predict(model,disable_tqdm)

        # Inverse preprocessing if raw #
        if self.preprocessing is not None:
            self.undo_preprocessing(model,true,mask,samples)

        # Make figure plots #
        figs = {}
        for i,(true_type,mask_type,samples_type) in enumerate(zip(true,mask,samples)):
            for j in range(true_type.shape[1]):
                # Get present particles #
                mask_part = mask_type[:,j]
                true_part = true_type[:,j,:][mask_part]
                sample_part = samples_type[:,:,j,:][mask_part]

                # Make figure name #
                figure_name = f'{model.reco_names[i]}_{j}'
                if len(self.suffix) > 0:
                    figure_name += f'_{self.suffix}'
                figure_name += '_{type}'

                # Plots #
                fig = self.plot_1D(
                    true = true_part,
                    samples = sample_part,
                    features = model.flow.flow_features[i],
                    title = f'{model.reco_names[i]} #{j}',
                )
                figs[figure_name.format(type='1D')] = fig
                fig = self.plot_2D(
                    true = true_part,
                    samples = sample_part,
                    features = model.flow.flow_features[i],
                    title = f'{model.reco_names[i]} #{j}',
                )
                figs[figure_name.format(type='2D')] = fig
                fig = self.plot_bias(
                    true = true_part,
                    samples = sample_part,
                    features = model.flow.flow_features[i],
                    title = f'{model.reco_names[i]} #{j}',
                )
                figs[figure_name.format(type='bias')] = fig
                fig = self.plot_quantiles(
                    true = true_part,
                    samples = sample_part,
                    features = model.flow.flow_features[i],
                    title = f'{model.reco_names[i]} #{j}',
                )
                figs[figure_name.format(type='qq')] = fig


                # Show #
                if show:
                    plt.show()

        return figs

class HighLevelVariableCallback(BaseCallback):
    def __init__(self,var_functions,N_batch=math.inf,batch_size=1024,**kwargs):
        super().__init__(**kwargs)

        # Attributes #
        self.var_functions = var_functions
        self.loader = DataLoader(self.dataset,batch_size=batch_size,shuffle=False)
        self.N_batch = N_batch

        assert self.preprocessing is not None, f'Need preprocessing pipeline'

    def make_plots(self,model,show=False,disable_tqdm=False):
        # Predict #
        true,mask,samples = self.predict(model,disable_tqdm,only_flow_variables=False)

        # Inverse preprocessing if raw #
        if self.preprocessing is not None:
            self.undo_preprocessing(model,true,mask,samples,only_flow_variables=False)

        # Turn into awkward array #
        features_per_type = model.decoder_embeddings.features_per_type
        if model.reco_names is None:
            names = [f'type_{i}' for i in range(len(true))]
        else:
            names = model.reco_names
        true_particles = {}
        sample_particles = {}
        for i in range(len(true)):
            features = features_per_type[i]
            assert len(features) == true[i].shape[-1], f'Particle type {names[i]} : {len(features)} features [{features}], but true.shape[-1] = {true[i].shape[-1]}'
            # Record true particles, turnes into awkward array #
            true_particles[names[i]] = ak.drop_none(
                ak.mask(
                    ak.zip(
                        {
                            feat : true[i][...,features.index(feat)]
                            for feat in features
                        },
                        with_name="Momentum4D",
                    ),
                    mask[i].numpy(),
                )
            )
            # Ravel the samples then same #
            sample_particles[names[i]] = ak.drop_none(
                ak.mask(
                    ak.zip(
                        {
                            feat : samples[i][...,features.index(feat)].reshape(
                                samples[i].shape[0] * samples[i].shape[1],
                                samples[i].shape[2]
                            )
                            for feat in features
                        },
                        with_name="Momentum4D",
                    ),
                    torch.repeat_interleave(
                        mask[i].unsqueeze(1),
                        samples[i].shape[1],
                        dim = 1,
                    ).reshape(
                        mask[i].shape[0] * self.N_sample,
                        mask[i].shape[1],
                    ).numpy(),
                )
            )

        # Produce variables #
        true_vars = {}
        sample_vars = {}
        for var_name, func in self.var_functions.items():
            # Check input arguments #
            args = inspect.getfullargspec(func)[0]
            assert set(args) == set(names), f'Function for var {var_name}, expects {args} as arguments, but particles sampled are {names}'
            # Make variables #
            try:
                true_vars[var_name] = func(**true_particles).to_numpy()
            except Exception as err:
                raise RuntimeError(f'Could not compute `{var_name}` for true particles because of `{err}`')
            try:
                sample_vars[var_name] = func(**sample_particles).to_numpy()
            except Exception as err:
                raise RuntimeError(f'Could not compute `{var_name}` for sampled particles because of `{err}`')

        # Make plots #
        figs = {}
        for var_name in self.var_functions.keys():
            fig = plt.figure(figsize=(12,6))
            gs = GridSpec(
                nrows = 1, ncols = 2,
                figure = fig,
                left = 0.1, bottom = 0.1, right = 0.9, top = 0.9,
                hspace = 0.05,
                wspace = 0.3,
            )
            bins = get_bins(
                [
                    true_vars[var_name],
                    sample_vars[var_name],
                ],
                var_name,
                self.bins,
            )
            label = var_name if var_name not in self.label_names.keys() else self.label_names[var_name]
            plot_ratio(
                fig = fig,
                gs = gs[0],
                true = true_vars[var_name],
                sample = sample_vars[var_name],
                bins = bins,
                density = True,
                label = label,
                log_scale = self.log_scale,
            )
            plot_2D_projections(
                fig = fig,
                gs = gs[1],
                true = true_vars[var_name].repeat(self.N_sample),
                sample = sample_vars[var_name],
                bins = bins,
                hexbin = self.hexbin,
                label_x = label,
                label_y = label,
                log_scale = self.log_scale,
            )
            figs[var_name] = fig


        if show:
            plt.show()

        return figs


