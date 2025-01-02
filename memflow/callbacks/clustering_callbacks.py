import math
import torch
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import Callback

from abc import ABCMeta, abstractmethod

class ReconstructionCallback(Callback):
    def __init__(self,dataset,names,features,number_particles_per_type,generative=True,preprocessing=None,frequency=1,raw=False,bins=None,log_scale=False,device=None,batch_size=1024):
        super().__init__()

        # Attributes #
        self.dataset = dataset
        self.names = names
        self.features = features
        self.number_particles_per_type = number_particles_per_type
        self.preprocessing = preprocessing
        self.loader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
        self.generative = generative
        self.frequency = frequency
        self.raw = raw
        self.bins = bins
        self.log_scale = log_scale
        self.device = device

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
        for name,fig in figs.items():
            trainer.logger.experiment.log_figure(
                figure_name = name,
                figure = fig,
                overwrite = True,
                step = trainer.current_epoch,
            )
            plt.close(fig)

    def predict(self,model,disable_tqdm=False):
        # Select device #
        if self.device is None:
            device = model.device
        else:
            device = self.device
        model = model.to(device)

        # Loop over batch #
        x_init = []
        x_reco = []
        mask = []
        for batch_idx, batch in tqdm(enumerate(self.loader),desc='Predict',disable=disable_tqdm,leave=True,total=len(self.loader),position=0):
            xi = torch.cat(batch['data'],dim=1).to(model.device)
            m = torch.cat(batch['mask'],dim=1).to(model.device)
            if m.dtype != torch.bool:
                m = m > 0
            with torch.no_grad():
                if self.generative:
                    latent = model.encode(xi,m)
                    xr = model.decode(latent,m)
                else:
                    xr = model(xi,m)
            x_init.append(xi.cpu())
            x_reco.append(xr.cpu())
            mask.append(m.cpu())

        # Concat #
        x_init = torch.cat(x_init,dim=0)
        x_reco = torch.cat(x_reco,dim=0)
        mask = torch.cat(mask,dim=0)

        return x_init,x_reco,mask

    def make_plots(self,model,show=True,disable_tqdm=False):
        # Predict #
        x_init,x_reco,mask = self.predict(model,disable_tqdm)

        # Split between particles #
        x_init = torch.split(x_init,self.number_particles_per_type,dim=1)
        x_reco = torch.split(x_reco,self.number_particles_per_type,dim=1)
        mask = torch.split(mask,self.number_particles_per_type,dim=1)

        # Loop over particle types #
        figs = {}
        for xi,xr,m,name,features in zip(x_init,x_reco,mask,self.names,self.features):
            # Preprocessing #
            if self.preprocessing is not None:
                xi,_ = self.preprocessing.inverse(
                    x = xi,
                    mask = m,
                    name = name,
                    fields = features,
                )
                xr,features = self.preprocessing.inverse(
                    x = xr,
                    mask = m,
                    name = name,
                    fields = features,
                )
            # Loop over particles #
            for i in range(xi.shape[1]):
                n = len(features)
                fig,axs = plt.subplots(nrows=2,ncols=n,figsize=(n*4,6))
                fig.suptitle(f'{name} #{i}')
                plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.5,hspace=0.4)
                # Loop over features #
                for j in range(n):
                    # tensor and binning #
                    xii = xi[:,i,j][m[:,i]]
                    xrr = xr[:,i,j][m[:,i]]
                    x_min = min(xii.min(),xrr.min())
                    x_max = max(xii.max(),xrr.max())
                    if x_min == x_max:
                        if x_min < 0:
                            x_min *= 1.1
                            x_max *= 0.9
                        elif x_min > 0:
                            x_min *= 0.9
                            x_max *= 1.1
                        else:
                            x_min = -0.1
                            x_max = +0.1
                    bins = np.linspace(x_min,x_max,self.bins)
                    # 1D plot #
                    axs[0,j].hist(xii,bins=bins,histtype='step',color='b',label="Initial")
                    axs[0,j].hist(xrr,bins=bins,histtype='step',color='orange',label="Reconstructed")
                    axs[0,j].legend(loc='upper right',fontsize=6)
                    if self.log_scale:
                        axs[0,j].set_yscale('log')
                    axs[0,j].set_xlabel(features[j])
                    # 2D plot #
                    h = axs[1,j].hist2d(
                        xii,
                        xrr,
                        bins = (bins,bins),
                        norm = matplotlib.colors.LogNorm() if self.log_scale else None,
                    )
                    axs[1,j].set_xlabel(f'Initial {features[j]}')
                    axs[1,j].set_ylabel(f'Reconstructed {features[j]}')
                    plt.colorbar(h[3],ax=axs[1,j])
                # Record #
                if self.generative:
                    figs[f'AE_gen_{name}_{i}'] = fig
                else:
                    figs[f'AE_reco_{name}_{i}'] = fig

        if show:
            plt.show()
        return figs

