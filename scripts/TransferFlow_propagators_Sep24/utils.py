from hist import Hist
import torch
import hist
import awkward as ak
import numpy as np
import os
import mplhep
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl

M_HIGGS = 125.25
M_TOP = 172.76

def check_mass(particle, mean, std):

    particle = particle*std + mean
    particle = torch.sign(particle)*(torch.exp(torch.abs(particle)) - 1)

    particle_try = vector.array(
        {
            "E": particle[:,0].detach().numpy(),
            "px": particle[:,1].detach().numpy(),
            "py": particle[:,2].detach().numpy(),
            "pz": particle[:,3].detach().numpy(),
        }
    )
    
    print(particle_try.mass)

def constrain_energy(higgs, thad, tlep, ISR, mean, std):

    unscaled_higgs = higgs*std[1:] + mean[1:]
    unscaled_thad = thad*std[1:] + mean[1:]
    unscaled_tlep = tlep*std[1:] + mean[1:]
    unscaled_ISR = ISR*std[1:] + mean[1:]

    regressed_higgs = torch.sign(unscaled_higgs)*(torch.exp(torch.abs(unscaled_higgs)) - 1)
    regressed_thad = torch.sign(unscaled_thad)*(torch.exp(torch.abs(unscaled_thad)) - 1)
    regressed_tlep = torch.sign(unscaled_tlep)*(torch.exp(torch.abs(unscaled_tlep)) - 1)
    regressed_ISR = torch.sign(unscaled_ISR)*(torch.exp(torch.abs(unscaled_ISR)) - 1)

    E_higgs = torch.sqrt(M_HIGGS**2 + regressed_higgs[:,0]**2 + \
                        regressed_higgs[:,1]**2 + regressed_higgs[:,2]**2).unsqueeze(dim=1)
            
    E_thad = torch.sqrt(M_TOP**2 + regressed_thad[:,0]**2 + \
                        regressed_thad[:,1]**2 + regressed_thad[:,2]**2).unsqueeze(dim=1)

    E_tlep = torch.sqrt(M_TOP**2 + regressed_tlep[:,0]**2 + \
                        regressed_tlep[:,1]**2 + regressed_tlep[:,2]**2).unsqueeze(dim=1)

    E_ISR = torch.sqrt(regressed_ISR[:,0]**2 + regressed_ISR[:,1]**2 + \
                        regressed_ISR[:,2]**2).unsqueeze(dim=1)

    logE_higgs = torch.log(1 + E_higgs)
    logE_thad = torch.log(1 + E_thad)
    logE_tlep = torch.log(1 + E_tlep)
    logE_ISR = torch.log(1 + E_ISR)

    logE_higgs = (logE_higgs - mean[0])/std[0]
    logE_thad = (logE_thad - mean[0])/std[0]
    logE_tlep = (logE_tlep - mean[0])/std[0]
    logE_ISR = (logE_ISR - mean[0])/std[0]

    return logE_higgs, logE_thad, logE_tlep, logE_ISR

def total_mom(higgs, thad, tlep, ISR, mean, std):

    unscaled_higgs = higgs*std[1:] + mean[1:]
    unscaled_thad = thad*std[1:] + mean[1:]
    unscaled_tlep = tlep*std[1:] + mean[1:]
    unscaled_ISR = ISR*std[1:] + mean[1:]

    regressed_higgs = torch.sign(unscaled_higgs)*(torch.exp(torch.abs(unscaled_higgs)) - 1)
    regressed_thad = torch.sign(unscaled_thad)*(torch.exp(torch.abs(unscaled_thad)) - 1)
    regressed_tlep = torch.sign(unscaled_tlep)*(torch.exp(torch.abs(unscaled_tlep)) - 1)
    regressed_ISR = torch.sign(unscaled_ISR)*(torch.exp(torch.abs(unscaled_ISR)) - 1)

    sum_px = regressed_higgs[:,0] + regressed_thad[:,0] + regressed_tlep[:,0] + regressed_ISR[:,0]
    sum_py = regressed_higgs[:,1] + regressed_thad[:,1] + regressed_tlep[:,1] + regressed_ISR[:,1]
    sum_pz = regressed_higgs[:,2] + regressed_thad[:,2] + regressed_tlep[:,2] + regressed_ISR[:,2]

    logsum_px = torch.log(1 + torch.abs(sum_px))
    logsum_py = torch.log(1 + torch.abs(sum_py))
    logsum_pz = torch.log(1 + torch.abs(sum_pz))

    return logsum_px, logsum_py, logsum_pz

def alter_variables(difference, object_no, variable_altered, target_var, mask_target, log_mean, log_std, no_max_objects, device, reco=1):
    # object_no: number of the object/objects that I want to alter
    # variable altered: 1 for pt; 2 for eta and 3 for phi for jets
    # variable altered: 0 for pt; 1 for eta and 2 for phi for partons

    if variable_altered == 0 and reco==1:
        raise Exception('On first position: always the exist flag - check logScaledReco')

    unscaled_var = torch.clone(target_var) # need prov when compute the log_prob
    unscaled_var[:,:,:3] = target_var[:,:,:3]*log_std + log_mean # unscale
    unscaled_var[:,:,0] = torch.exp(unscaled_var[:,:,0]) - 1 # unscale pt

    for i, object in enumerate(object_no):
        for j, var in enumerate(variable_altered):
            unscaled_var[:, object, var] = unscaled_var[:, object, var] + difference[i*len(object_no) + j] # add difference for each var
        
    unscaled_var[:,:,0] = torch.log(unscaled_var[:,:,0] + 1) # log pt

    unscaled_var[:,:,:3] = (unscaled_var[:,:,:3] - log_mean)/log_std # scale back
    unscaled_var = unscaled_var*mask_target.unsqueeze(dim=2) # because padding jets are not 0 anymore
            
    return unscaled_var

def alter_variables_tensor(difference, object_no, variable_altered, target_var, mask_target, log_mean, log_std, no_max_objects, device, reco=1):
    # object_no: number of the object/objects that I want to alter
    # variable altered: 1 for pt; 2 for eta and 3 for phi for jets
    # variable altered: 0 for pt; 1 for eta and 2 for phi for partons

    if variable_altered == 0 and reco==1:
        raise Exception('On first position: always the exist flag - check logScaledReco')

    unscaled_var = torch.clone(target_var) # need prov when compute the log_prob
    unscaled_var[:,:,:3] = target_var[:,:,:3]*log_std + log_mean # unscale
    unscaled_var[:,:,0] = torch.exp(unscaled_var[:,:,0]) - 1 # unscale pt

    unscaled_var[:, object_no, variable_altered] = unscaled_var[:, object_no, variable_altered] + difference # add difference for each var
        
    unscaled_var[:,:,0] = torch.log(unscaled_var[:,:,0] + 1) # log pt

    unscaled_var[:,:,:3] = (unscaled_var[:,:,:3] - log_mean)/log_std # scale back
    unscaled_var = unscaled_var*mask_target.unsqueeze(dim=2) # because padding jets are not 0 anymore
            
    return unscaled_var

class SavePlots:
    
    def __init__(self, nameDir):
        
        self.nameDir = nameDir
        if os.path.exists(self.nameDir):
            raise Exception("Directory already exists! Use another path")
        else:
            os.mkdir(self.nameDir)
            
    
    def plot_var2d(self, higgs_var1, higgs_var2, thad_var1, thad_var2, tlep_var1, tlep_var2,
             ISR_var1, ISR_var2, name1, name2, nameFig, start1=0, stop1=1000, start2=0, stop2=1000, bins1=100, bins2=100,
             higgs_mask=1, thad_mask=1, tlep_mask=1, ISR_mask=1, neg_Mask=False, log=False, nameunit=''):
        
        
        if neg_Mask:
            higgs_mask = np.logical_not(higgs_mask)
            thad_mask = np.logical_not(thad_mask)
            tlep_mask = np.logical_not(tlep_mask)
            ISR_mask = np.logical_not(ISR_mask)

        # Quick construction, no other imports needed:
        hist2d_var_higgs = (
          Hist.new
          .Reg(bins=bins1, start=start1, stop=stop1, name=name1, label=name1)
          .Reg(bins=bins2, start=start2, stop=stop2, name=name2, label=name2)
          .Double())

        hist2d_var_thad = (
          Hist.new
          .Reg(bins=bins1, start=start1, stop=stop1, name=name1, label=name1)
          .Reg(bins=bins2, start=start2, stop=stop2, name=name2, label=name2)
          .Double())

        hist2d_var_tlep = (
          Hist.new
          .Reg(bins=bins1, start=start1, stop=stop1, name=name1, label=name1)
          .Reg(bins=bins2, start=start2, stop=stop2, name=name2, label=name2)
          .Double())

        hist2d_var_ISR = (
          Hist.new
          .Reg(bins=bins1, start=start1, stop=stop1, name=name1, label=name1)
          .Reg(bins=bins2, start=start2, stop=stop2, name=name2, label=name2)
          .Double())

        hist2d_var_higgs.fill(higgs_var1[higgs_mask],
                            higgs_var2[higgs_mask])

        hist2d_var_thad.fill(thad_var1[thad_mask],
                            thad_var2[thad_mask])

        hist2d_var_tlep.fill(tlep_var1[tlep_mask],
                            tlep_var2[tlep_mask])

        hist2d_var_ISR.fill(ISR_var1[ISR_mask],
                            ISR_var2[ISR_mask])

        colormap='viridis'
        my_viridis = mpl.colormaps[colormap].with_extremes(under="white")
        
        fontsize = 20
        labelsize=14
        labels = ['higgs', 'thad', 'tlep', 'ISR']
        hist2d_list = [hist2d_var_higgs, hist2d_var_thad, hist2d_var_tlep, hist2d_var_ISR]
        
        
        fig, axs = plt.subplots(1, 4, figsize=(16, 6))

        if log:
            for i in range(4):
                w, x, y = hist2d_list[i].to_numpy()
                mesh = axs[i].pcolormesh(x, y, w.T, cmap=my_viridis, norm=mpl.colors.LogNorm(vmin=1))
                axs[i].set_xlabel(f"{name1} {nameunit}", fontsize=fontsize)
                axs[i].set_ylabel(f"{name2} {nameunit}", fontsize=fontsize)
                axs[i].set_title(f"{labels[i]}", fontsize=fontsize)
                axs[i].tick_params(axis='both', which='major', labelsize=labelsize)
                cbar = fig.colorbar(mesh)
                cbar.ax.tick_params(labelsize=labelsize)

        else:
            for i in range(4):
                w, x, y = hist2d_list[i].to_numpy()
                mesh = axs[i].pcolormesh(x, y, w.T, cmap=my_viridis, vmin=1)
                axs[i].set_xlabel(f"{name1} {nameunit}", fontsize=fontsize)
                axs[i].set_ylabel(f"{name2} {nameunit}", fontsize=fontsize)
                axs[i].set_title(f"{labels[i]}", fontsize=fontsize)
                axs[i].tick_params(axis='both', which='major', labelsize=labelsize)
                cbar = fig.colorbar(mesh)
                cbar.ax.tick_params(labelsize=labelsize)

        plt.tight_layout()
        plt.savefig(self.nameDir + '/' + nameFig)            
                
    def plot_var2d_old(self, higgs_var1, higgs_var2, thad_var1, thad_var2, tlep_var1, tlep_var2,
             ISR_var1, ISR_var2, name1, name2, nameFig, start1=0, stop1=1000, start2=0, stop2=1000, bins1=100, bins2=100,
             higgs_mask=1, thad_mask=1, tlep_mask=1, ISR_mask=1, neg_Mask=False, log=False):
        
        
        if neg_Mask:
            higgs_mask = np.logical_not(higgs_mask)
            thad_mask = np.logical_not(thad_mask)
            tlep_mask = np.logical_not(tlep_mask)
            ISR_mask = np.logical_not(ISR_mask)

        # Quick construction, no other imports needed:
        hist2d_var_higgs = (
          Hist.new
          .Reg(bins=bins1, start=start1, stop=stop1, name=name1, label=name1)
          .Reg(bins=bins2, start=start2, stop=stop2, name=name2, label=name2)
          .Double())

        hist2d_var_thad = (
          Hist.new
          .Reg(bins=bins1, start=start1, stop=stop1, name=name1, label=name1)
          .Reg(bins=bins2, start=start2, stop=stop2, name=name2, label=name2)
          .Double())

        hist2d_var_tlep = (
          Hist.new
          .Reg(bins=bins1, start=start1, stop=stop1, name=name1, label=name1)
          .Reg(bins=bins2, start=start2, stop=stop2, name=name2, label=name2)
          .Double())

        hist2d_var_ISR = (
          Hist.new
          .Reg(bins=bins1, start=start1, stop=stop1, name=name1, label=name1)
          .Reg(bins=bins2, start=start2, stop=stop2, name=name2, label=name2)
          .Double())

        hist2d_var_higgs.fill(higgs_var1[higgs_mask],
                            higgs_var2[higgs_mask])

        hist2d_var_thad.fill(thad_var1[thad_mask],
                            thad_var2[thad_mask])

        hist2d_var_tlep.fill(tlep_var1[tlep_mask],
                            tlep_var2[tlep_mask])

        hist2d_var_ISR.fill(ISR_var1[ISR_mask],
                            ISR_var2[ISR_mask])

        fig, axs = plt.subplots(1, 4, figsize=(16, 8))

        if log:
            mplhep.hist2dplot(hist2d_var_higgs, cmap="cividis", norm=mpl.colors.LogNorm(vmin=1), ax=axs[0])

            mplhep.hist2dplot(hist2d_var_thad, cmap="cividis", norm=mpl.colors.LogNorm(vmin=1), ax=axs[1])

            mplhep.hist2dplot(hist2d_var_tlep, cmap="cividis", norm=mpl.colors.LogNorm(vmin=1), ax=axs[2])

            mplhep.hist2dplot(hist2d_var_ISR, cmap="cividis", norm=mpl.colors.LogNorm(vmin=1), ax=axs[3])

        else:
            mplhep.hist2dplot(hist2d_var_higgs, cmap="cividis", cmin=1, ax=axs[0])

            mplhep.hist2dplot(hist2d_var_thad, cmap="cividis", cmin=1, ax=axs[1])

            mplhep.hist2dplot(hist2d_var_tlep, cmap="cividis", cmin=1, ax=axs[2])

            mplhep.hist2dplot(hist2d_var_ISR, cmap="cividis", cmin=1, ax=axs[3])

        plt.tight_layout()
        plt.savefig(self.nameDir + '/' + nameFig)

    def plot_var1d(self, higgs_var1, thad_var1, tlep_var1, ISR_var1, name1, nameFig, start1=0, stop1=1000, bins1=100,
             higgs_mask=1, thad_mask=1, tlep_mask=1, ISR_mask=1, neg_Mask=False, log=False):
        
        if neg_Mask:
            higgs_mask = np.logical_not(higgs_mask)
            thad_mask = np.logical_not(thad_mask)
            tlep_mask = np.logical_not(tlep_mask)
            ISR_mask = np.logical_not(ISR_mask)

        # Quick construction, no other imports needed:
        hist1d_var_higgs = Hist(hist.axis.Regular(bins=bins1, start=start1, stop=stop1, name=name1))
        
        hist1d_var_thad = Hist(hist.axis.Regular(bins=bins1, start=start1, stop=stop1, name=name1))

        hist1d_var_tlep = Hist(hist.axis.Regular(bins=bins1, start=start1, stop=stop1, name=name1))

        hist1d_var_ISR = Hist(hist.axis.Regular(bins=bins1, start=start1, stop=stop1, name=name1))


        hist1d_var_higgs.fill(higgs_var1[higgs_mask])

        hist1d_var_thad.fill(thad_var1[thad_mask])

        hist1d_var_tlep.fill(tlep_var1[tlep_mask])

        hist1d_var_ISR.fill(ISR_var1[ISR_mask])

        fig, axs = plt.subplots(1, 4, figsize=(16, 8))

        if log:
            mplhep.histplot(hist1d_var_higgs, ax=axs[0])

            mplhep.histplot(hist1d_var_thad, ax=axs[1])

            mplhep.histplot(hist1d_var_tlep, ax=axs[2])

            mplhep.histplot(hist1d_var_ISR, ax=axs[3])

        else:
            mplhep.histplot(hist1d_var_higgs, ax=axs[0])

            mplhep.histplot(hist1d_var_thad, ax=axs[1])

            mplhep.histplot(hist1d_var_tlep, ax=axs[2])

            mplhep.histplot(hist1d_var_ISR, ax=axs[3])

        plt.tight_layout()
        plt.savefig(self.nameDir + '/' + nameFig)
        
    def plot_particle(self, particleCorrect, particle, nameFig, particle_mask=1, neg_Mask=False, log=False, nameunit=''):
        
        if neg_Mask:
            particle_mask = np.logical_not(particle_mask)

        hist2d_E_thad = (
          Hist.new
          .Reg(bins=150, start=100, stop=2000, name="E-correct", label="E-correct")
          .Reg(bins=150, start=100, stop=2000, name="E-regressed", label="E-regressed")
          .Double())

        hist2d_px_thad = (
          Hist.new
          .Reg(bins=100, start=-1500, stop=1500, name="px-correct", label="px-correct")
          .Reg(bins=100, start=-1500, stop=1500, name="px-regressed", label="px-regressed")
          .Double())

        hist2d_py_thad = (
          Hist.new
          .Reg(bins=100, start=-1500, stop=1500, name="py-correct", label="py-correct")
          .Reg(bins=100, start=-1500, stop=1500, name="py-regressed", label="py-regressed")
          .Double())

        hist2d_pz_thad = (
          Hist.new
          .Reg(bins=100, start=-2500, stop=2500, name="pz-correct", label="pz-correct")
          .Reg(bins=100, start=-2500, stop=2500, name="pz-regressed", label="pz-regressed")
          .Double())

        hist2d_E_thad.fill(particleCorrect.E[particle_mask],
                        particle.E[particle_mask])

        hist2d_px_thad.fill(particleCorrect.px[particle_mask],
                        particle.px[particle_mask])

        hist2d_py_thad.fill(particleCorrect.py[particle_mask],
                        particle.py[particle_mask])

        hist2d_pz_thad.fill(particleCorrect.pz[particle_mask],
                        particle.pz[particle_mask])
        
        colormap='viridis'
        my_viridis = mpl.colormaps[colormap].with_extremes(under="white")
        
        fontsize = 20
        labelsize=14
        hist2d_list = [hist2d_E_thad, hist2d_px_thad, hist2d_py_thad, hist2d_pz_thad]
        labels = ['E', 'px', 'py', 'pz']
        fig, axs = plt.subplots(2, 2, figsize=(20, 10))

        if log:

            for i in range(2):
                for j in range(2):
                    w, x, y = hist2d_list[i].to_numpy()
                    mesh = axs[i, j].pcolormesh(x, y, w.T, cmap=my_viridis, norm=mpl.colors.LogNorm(vmin=1))
                    axs[i, j].set_xlabel(f"{labels[2*i+j]}-correct {nameunit}", fontsize=fontsize)
                    axs[i, j].set_ylabel(f"{labels[2*i+j]}-regressed {nameunit}", fontsize=fontsize)
                    axs[i, j].tick_params(axis='both', which='major', labelsize=labelsize)
                    cbar = fig.colorbar(mesh)
                    cbar.ax.tick_params(labelsize=labelsize)

        else:
            for i in range(2):
                for j in range(2):
                    w, x, y = hist2d_list[i].to_numpy()
                    mesh = axs[i, j].pcolormesh(x, y, w.T, cmap=my_viridis, vmin=1)
                    axs[i, j].set_xlabel(f"{labels[2*i+j]}-correct {nameunit}", fontsize=fontsize)
                    axs[i, j].set_ylabel(f"{labels[2*i+j]}-regressed {nameunit}", fontsize=fontsize)
                    axs[i, j].tick_params(axis='both', which='major', labelsize=labelsize)
                    cbar = fig.colorbar(mesh)
                    cbar.ax.tick_params(labelsize=labelsize)

        plt.tight_layout()
        plt.savefig(self.nameDir + '/' + nameFig)
        
        
    def plot_rambo(self, rambo_correct, rambo_regressed, nameFig, typePlot=0, log=False):
        if (typePlot > 2):
            typePlot = 2

        hist2d_rambo_0 = (
          Hist.new
          .Reg(bins=200, start=-0.05, stop=1, name=f"rambo-correct-{4*typePlot}", label=f"rambo-correct-{4*typePlot}")
          .Reg(bins=200, start=-0.05, stop=1, name=f"rambo-regressed-{4*typePlot}", label=f"rambo-regressed-{4*typePlot}")
          .Double())

        hist2d_rambo_1 = (
          Hist.new
          .Reg(bins=200, start=-0.05, stop=1, name=f"rambo-correct-{4*typePlot + 1}", label=f"rambo-correct-{4*typePlot + 1}")
          .Reg(bins=200, start=-0.05, stop=1, name=f"rambo-regressed-{4*typePlot + 1}", label=f"rambo-regressed-{4*typePlot + 1}")
          .Double())

        hist2d_rambo_0.fill(rambo_correct[:,4*typePlot],
                        rambo_regressed[:,4*typePlot])

        hist2d_rambo_1.fill(rambo_correct[:,4*typePlot + 1],
                        rambo_regressed[:,4*typePlot + 1])

        num_plots = 2
    
        if typePlot < 2:
            hist2d_rambo_2 = (
              Hist.new
              .Reg(bins=200, start=-0.05, stop=1, name=f"rambo-correct-{4*typePlot + 2}", label=f"rambo-correct-{4*typePlot + 2}")
              .Reg(bins=200, start=-0.05, stop=1, name=f"rambo-regressed-{4*typePlot + 2}", label=f"rambo-regressed-{4*typePlot + 2}")
              .Double())

            hist2d_rambo_3 = (
              Hist.new
              .Reg(bins=200, start=-0.05, stop=1, name=f"rambo-correct-{4*typePlot + 3}", label=f"rambo-correct-{4*typePlot + 3}")
              .Reg(bins=200, start=-0.05, stop=1, name=f"rambo-regressed-{4*typePlot + 3}", label=f"rambo-regressed-{4*typePlot + 3}")
              .Double())

            hist2d_rambo_2.fill(rambo_correct[:,4*typePlot + 2],
                        rambo_regressed[:,4*typePlot + 2])

            hist2d_rambo_3.fill(rambo_correct[:,4*typePlot + 3],
                            rambo_regressed[:,4*typePlot + 3])

            num_plots = 4


        fig, axs = plt.subplots(1, num_plots, figsize=(16, 8))
    
        if log:
            mplhep.hist2dplot(hist2d_rambo_0, cmap="cividis", norm=mpl.colors.LogNorm(vmin=1), ax=axs[0])
            mplhep.hist2dplot(hist2d_rambo_1, cmap="cividis", norm=mpl.colors.LogNorm(vmin=1), ax=axs[1])

            if typePlot < 2:
                mplhep.hist2dplot(hist2d_rambo_2, cmap="cividis", norm=mpl.colors.LogNorm(vmin=1), ax=axs[2])
                mplhep.hist2dplot(hist2d_rambo_3, cmap="cividis", norm=mpl.colors.LogNorm(vmin=1), ax=axs[3])

        else:
            mplhep.hist2dplot(hist2d_rambo_0, cmap="cividis", cmin=1, ax=axs[0])
            mplhep.hist2dplot(hist2d_rambo_1, cmap="cividis", cmin=1, ax=axs[1])

            if typePlot < 2:
                mplhep.hist2dplot(hist2d_rambo_2, cmap="cividis", cmin=1, ax=axs[2])
                mplhep.hist2dplot(hist2d_rambo_3, cmap="cividis", cmin=1, ax=axs[3])

        plt.tight_layout()
        plt.savefig(self.nameDir + '/' + nameFig)
        
def return_Interval(regressedVar, correctVar, targetVar, lim=(0,1), ratio=False):
    diffRegression = regressedVar - correctVar
    if ratio==True:
        diffRegression = diffRegression/correctVar

    indices = np.where((targetVar >= lim[0])  & (targetVar < lim[1]))[0]
    ratioRegression_interval = diffRegression[indices]
    
    mean_ratioRegression = np.mean(ratioRegression_interval)
    std1sigma = (np.quantile(ratioRegression_interval, 0.84)  - np.quantile(ratioRegression_interval, 0.16))/2
    std2sigma = (np.quantile(ratioRegression_interval, 0.975)  - np.quantile(ratioRegression_interval, 0.025))/2    
    
    return mean_ratioRegression, std1sigma, std2sigma, ratioRegression_interval, indices
      
def plot_regressionFactor(regressedVar, correctVar, targetVar, matched, limTarget, bins, intervalTargetVar=0,
                            xerr_plot=False, ylim=[-10, 10],
                            xname='x', yname='y', eta=False, ratio=False, nameDir='', nameFig='1.png'):
    mean_result = []
    std_result = []
    std2_result = []

    if matched is not None:
        regressedVar = regressedVar[matched]
        correctVar = correctVar[matched]
        targetVar = targetVar[matched]
    
    if intervalTargetVar == 0:
        intervalTargetVar = np.linspace(start=limTarget[0], stop=limTarget[1], num=bins+1)
    
    target_axis = [(intervalTargetVar[i]+intervalTargetVar[i+1])/2 for i in range(0, len(intervalTargetVar)-1)]
    
    for i in range(len(intervalTargetVar) - 1):
        mean, std, std2, results, indices = return_Interval(regressedVar, correctVar, targetVar,
                                             lim=(intervalTargetVar[i],intervalTargetVar[i+1]), ratio=ratio)
        mean_result.append(mean)
        std_result.append(std)
        std2_result.append(std2)
        
    mean_result = np.array(mean_result)
    std_result = np.array(std_result)
    std2_result = np.array(std2_result)
    
    fontsize = 22
    labelsize = 16
    
    plt.plot(target_axis, mean_result, linestyle='-', marker='o', color='k')
    if eta:
        plt.axvline(x=-1.5, linestyle='--', color='k', alpha=0.2,)
        plt.axvline(x=+1.5, linestyle='--', color='k', alpha=0.2,)
        plt.axvline(x=-3, linestyle='--', color='b', alpha=0.2,)
        plt.axvline(x=+3, linestyle='--', color='b', alpha=0.2,)
        
    plt.fill_between(target_axis, mean_result - std_result, mean_result + std_result,
                     color='r', alpha=0.2, label='68% CL')
    plt.fill_between(target_axis, mean_result - std2_result, mean_result + std2_result,
                     color='b', alpha=0.2, label='95% CL')
        
    plt.ylim(ylim)
    plt.xlabel(xname, fontsize=fontsize)
    plt.ylabel(yname, fontsize=fontsize)
    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.tight_layout()
    if nameDir != '':
        plt.savefig(nameDir + '/' + nameFig)
    plt.show()
    
def get_quantiles(find_quantile, interval, regressed_var):
    if find_quantile < interval/2.:
        quantile_left = 0.
        find_quantile_left = np.quantile(regressed_var, quantile_left)
        quantile_right = interval
        find_quantile_right = np.quantile(regressed_var, quantile_right)

    elif find_quantile > 1.0-interval/2.:
        quantile_left = 1.0-interval
        find_quantile_left = np.quantile(regressed_var, quantile_left)
        quantile_right = 1.0
        find_quantile_right = np.quantile(regressed_var, quantile_right)

    else:
        quantile_left = find_quantile - 0.34
        find_quantile_left = np.quantile(regressed_var, quantile_left)
        quantile_right = find_quantile + 0.34
        find_quantile_right = np.quantile(regressed_var, quantile_right)
        
    return find_quantile_left, find_quantile_right


from numba import jit
@jit(nopython=True)
def get_central_smallest_interval(array, xrange, nbins, Ntrial=10000, perc=0.68):
    H = np.histogram(array, bins=nbins, range=xrange)
    xmax = H[1][np.argmax(H[0])]
    deltax = (xrange[1]-xrange[0])/(2*Ntrial)
    
    N = len(array)
    xd = xmax-deltax
    xu = xmax+deltax
    for i in range(Ntrial):
        q = np.sum((array>xd) &(array<xu))/ N
        if q>=perc: 
            break
        xd = xd-deltax
        xu = xu+deltax
    return xmax, xd, xu
    

def plot_mode_quantile(regressed_var, target_var, target_values, window, nbins, range,
                      xlabel='', ylabel=''):
    
    regressed_modes = []
    quantile_right_list = []
    quantile_left_list = []
    quantile_right_list_95 = []
    quantile_left_list_95 = []
    
    for target_value in target_values:
        mask = ((target_var > (target_value-window)) & (target_var < (target_value+window))).to_numpy()

        hist, bin_edges = np.histogram(regressed_var[mask], bins=nbins, range=range)
        xmax_left = bin_edges[np.argmax(hist)]
        xmax_right = bin_edges[np.argmax(hist)+1]
        regressed_mode = (xmax_left + xmax_right)/2
        regressed_modes.append(regressed_mode)

        find_quantile = np.count_nonzero(regressed_var[mask] < regressed_mode) / len(regressed_var[mask])

        find_quantile_left, find_quantile_right = get_quantiles(find_quantile=find_quantile,
                                                                interval=0.68,
                                                                regressed_var=regressed_var[mask])

        quantile_right_list.append(find_quantile_right)
        quantile_left_list.append(find_quantile_left)
        
        find_quantile_left_95, find_quantile_right_95 = get_quantiles(find_quantile=find_quantile,
                                                                interval=0.95,
                                                                regressed_var=regressed_var[mask])

        quantile_right_list_95.append(find_quantile_right_95)
        quantile_left_list_95.append(find_quantile_left_95)
        
    offset_left = np.array(regressed_modes) - np.array(quantile_left_list) # need offset
    offset_right = np.array(quantile_right_list) - np.array(regressed_modes)
    
    offset_left_95 = np.array(regressed_modes) - np.array(np.array(quantile_left_list_95)) # need offset
    offset_right_95 = np.array(quantile_right_list_95) - np.array(regressed_modes)
    
    regressed_modes_diff = regressed_modes - target_values
        
    plt.plot(target_values, regressed_modes_diff, linestyle='-', marker='o', color='k')
        
    plt.fill_between(target_values, regressed_modes_diff - offset_left, regressed_modes_diff + offset_right,
                     color='r', alpha=0.2, label='68% CL')
    plt.fill_between(target_values, regressed_modes_diff - offset_left_95, regressed_modes_diff + offset_right_95,
                     color='b', alpha=0.2, label='95% CL')
    plt.legend(fontsize=10)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    plt.show()
    
def plot_diff_mode_quantile(regressed_var, target_var, referance_var, target_values, window, nbins, range,
                      xlabel='', ylabel='', title=''):
    
    regressed_modes = []
    quantile_right_list = []
    quantile_left_list = []
    quantile_right_list_95 = []
    quantile_left_list_95 = []
    
    for target_value in target_values:
        mask = ((referance_var > (target_value-window)) & (referance_var < (target_value+window))).to_numpy()

        diff_var = regressed_var[mask] - target_var[mask]
        
        hist, bin_edges = np.histogram(diff_var, bins=nbins, range=range)
        xmax_left = bin_edges[np.argmax(hist)]
        xmax_right = bin_edges[np.argmax(hist)+1]
        regressed_mode = (xmax_left + xmax_right)/2
        regressed_modes.append(regressed_mode)

        find_quantile = np.count_nonzero(diff_var < regressed_mode) / len(diff_var)

        find_quantile_left, find_quantile_right = get_quantiles(find_quantile=find_quantile,
                                                                interval=0.68,
                                                                regressed_var=diff_var)

        quantile_right_list.append(find_quantile_right)
        quantile_left_list.append(find_quantile_left)
        
        find_quantile_left_95, find_quantile_right_95 = get_quantiles(find_quantile=find_quantile,
                                                                interval=0.95,
                                                                regressed_var=diff_var)

        quantile_right_list_95.append(find_quantile_right_95)
        quantile_left_list_95.append(find_quantile_left_95)
        
    offset_left = np.array(regressed_modes) - np.array(quantile_left_list) # need offset
    offset_right = np.array(quantile_right_list) - np.array(regressed_modes)
    
    offset_left_95 = np.array(regressed_modes) - np.array(np.array(quantile_left_list_95)) # need offset
    offset_right_95 = np.array(quantile_right_list_95) - np.array(regressed_modes)
            
    plt.plot(target_values, regressed_modes, linestyle='-', marker='o', color='k')
        
    plt.fill_between(target_values, regressed_modes - offset_left, regressed_modes + offset_right,
                     color='r', alpha=0.2, label='68% CL')
    plt.fill_between(target_values, regressed_modes - offset_left_95, regressed_modes + offset_right_95,
                     color='b', alpha=0.2, label='95% CL')
    plt.legend(fontsize=10)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.title(title)
    
    plt.show()
    
class FindMasks:

    def higgs_mask(self, jets):
        prov1_jets = jets[jets.prov == 1]
        prov1 = prov1_jets["prov"]
        higgs_mask = ak.count(prov1, axis=1) == 2
        
        higgs_mask = ak.to_numpy(higgs_mask)
        return higgs_mask
    
    def thad_mask(self, jets):
        prov2_jets = jets[jets.prov == 2] # b from hadronic top decay
        prov2 = prov2_jets["prov"]
        hadb_mask = ak.count(prov2, axis=1) == 1
        
        prov5_jets = jets[jets.prov == 5] # q from hadronic W decay
        prov5 = prov5_jets["prov"]
        hadW_mask = ak.count(prov5, axis=1) == 2
        
        hadb_mask = ak.to_numpy(hadb_mask)
        hadW_mask = ak.to_numpy(hadW_mask)
        
        thad_mask = np.logical_and(hadb_mask, hadW_mask)
        return thad_mask
    
    def tlep_mask(self, jets):
        prov3_jets = jets[jets.prov == 3] # b from lept top decay
        prov3 = prov3_jets["prov"]
        
        blep_mask = ak.count(prov3, axis=1) == 1
        tlep_mask = ak.to_numpy(blep_mask)
        
        return tlep_mask
    
    def ISR_mask(self, jets):
        prov4_jets = jets[jets.prov == 4]
        prov4 = prov4_jets["prov"]
        
        ISR_mask = ak.count(prov4, axis=1) == 1
        ISR_mask = ak.to_numpy(ISR_mask)
        
        return ISR_mask


from numba import jit
@jit(nopython=True)
def get_central_smallest_interval(array, xrange, nbins, Ntrial=10000, perc=0.68):
    H = np.histogram(array, bins=nbins, range=xrange)
        # trying to smooth out noisy histo

    xmax_l = H[1][np.argmax(H[0])-5]
    xmax_h = H[1][np.argmax(H[0])+5]
    xmax = np.mean(array[(array>xmax_l)&(array<xmax_h)])
    
    deltax = (xrange[1]-xrange[0])/(2*Ntrial)

    absmax = np.quantile(array, 0.99)
    absmin = np.quantile(array, 0.01)
    
    N = array.shape[0]
    xd = xmax-deltax
    xu = xmax+deltax
    for i in range(Ntrial):
        q = np.sum((array>xd) &(array<xu))/ N
        if q>=perc: 
            break
        if xd > absmin:
            xd = xd-deltax
        if xu < absmax:
            xu = xu+deltax
    return xmax, xd, xu


def plot_diff_mode_quantile(Y, X, cat_var, bins,
                      xlabel='', ylabel='', title='', debug=False,
                    xlim=None, ylim=None, nbins_mode=100):
    
    Y_mode  = []
    X_avg = []
    quantile_right_list = []
    quantile_left_list = []
    quantile_right_list_95 = []
    quantile_left_list_95 = []

    for i in range(len(bins)-1):
        mask = ((cat_var >= bins[i])&(cat_var< bins[i+1]))
        Ymask = Y[mask]
        X_avg.append(np.mean(X[mask]))

        #print((np.min(Ymask), np.max(Ymask)))
        mode, left, right = get_central_smallest_interval(Ymask, xrange=(np.quantile(Ymask, 0.03), np.quantile(Ymask, 0.97)),
                                      nbins=Ymask.size//nbins_mode, Ntrial=2000, perc=0.68)

        _, left95, right95 = get_central_smallest_interval(Ymask, xrange=(np.quantile(Ymask, 0.03), np.quantile(Ymask, 0.97)),
                                      nbins=Ymask.size//nbins_mode, Ntrial=2000, perc=0.95)

        if debug:
            f = plt.figure()
            plt.hist(Ymask, range=(np.quantile(Ymask, 0.03), np.quantile(Ymask, 0.97)),   bins=Ymask.size//100, histtype="step")
            plt.axvline(left, c='r', label="0.68")
            plt.axvline(right, c='r')
            plt.axvline(left95, c='orange', label="0.95")
            plt.axvline(right95, c='orange')
            plt.axvline(mode, label="mode", c="green")
            plt.title(f"Bins {bins[i]:.2f}-{bins[i+1]:.2}")
            plt.legend()
            plt.show()

        
        Y_mode.append(mode)
        quantile_left_list.append(left)
        quantile_right_list.append(right)
        quantile_left_list_95.append(left95)
        quantile_right_list_95.append(right95)
        #print(Y_mode, left, right)

    offset_left = np.array(Y_mode) - np.array(quantile_left_list) # need offset
    offset_right = np.array(quantile_right_list) - np.array(Y_mode)
    
    offset_left_95 = np.array(Y_mode) - np.array(np.array(quantile_left_list_95)) # need offset
    offset_right_95 = np.array(quantile_right_list_95) - np.array(Y_mode)
        
    f = plt.figure(figsize=(6,5),dpi = 100)
    plt.plot(X_avg, Y_mode, linestyle='-', marker='o', color='k')
    plt.grid(axis="y")
        
    plt.fill_between(X_avg, Y_mode - offset_left, Y_mode + offset_right,
                     color='r', alpha=0.2, label='68% CL')
    plt.fill_between(X_avg, Y_mode - offset_left_95, Y_mode + offset_right_95,
                     color='b', alpha=0.2, label='95% CL')
    plt.legend(fontsize=12)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    plt.title(title, fontsize=18)
    