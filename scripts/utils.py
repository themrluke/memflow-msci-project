from hist import Hist
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