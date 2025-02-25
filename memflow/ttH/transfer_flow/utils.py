import torch
import numpy as np
from scipy.stats import gaussian_kde

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

EPS = 1e-12

def make_kde_xy(values,h=None,N=1000):
    # Fit with kde #
    density = gaussian_kde(values)
    x = np.linspace(values.min(),values.max(),N)
    y = density(x)
    # If h (provided by the pyplot hist)
    if h is not None:
        w = h[1][1:]-h[1][:-1]
        integral= (h[0]*w).sum()
        y *= integral
    return x,y

def get_bins(arrays,feature,N=40):
    if not isinstance(arrays,(tuple,list)):
        arrays = [arrays]
    for i in range(len(arrays)):
        if isinstance(arrays[i],(int,float)):
            arrays[i] = np.array([arrays[i]])
        elif isinstance(arrays[i],np.ndarray) or torch.is_tensor((arrays[i])):
            pass
        else:
            raise NotImplementedError
    if feature in ['E','m','mass','pt']:
        bins = np.linspace(
            0.,
            max(
                [
                    torch.quantile(array,0.999,interpolation='higher')
                    for array in arrays
                ]
            ),
            N,
        )
    elif feature in ['px','py','pz']:
        max_rng = max(
            [
                abs(torch.quantile(array,0.999,interpolation='higher'))
                for array in arrays
            ] + \
            [
                abs(torch.quantile(array,0.001,interpolation='lower'))
                for array in arrays
            ]
        )
        bins = np.linspace(-max_rng,max_rng,N)
    elif feature in ['eta','phi']:
        bins = np.linspace(
            min([array.min() for array in arrays]),
            max([array.max() for array in arrays]),
            N,
        )
    else:
        bins = np.linspace(
            min([array.min() for array in arrays]),
            max([array.max() for array in arrays]),
            N,
        )
    return bins

def compute_diff_coverage(true,diff,bins,relative=False):
    centers = []
    quantiles = []
    for x_min,x_max in zip(bins[:-1],bins[1:]):
        mask = torch.logical_and(true<=x_max,true>=x_min)
        y = diff[mask]
        if relative:
            y /= true[mask]
        if y.sum() == 0:
            continue
        quantiles.append(
            torch.quantile(
                y,
                q = torch.tensor([0.02275,0.1587,0.5,0.8413,0.97725]).to(y.dtype),
            ).unsqueeze(0)
        )
        centers.append((x_max+x_min)/2)
    return torch.tensor(centers),torch.cat(quantiles,dim=0)



def compute_diff_means(true,diff,bins,relative=False):
    means = []
    for x_min,x_max in zip(bins[:-1],bins[1:]):
        mask = torch.logical_and(true<=x_max,true>=x_min)
        y = diff[mask]
        if relative:
            y /= true[mask]
        if y.sum() == 0:
            continue
        means.append(y.mean())
    return torch.tensor(means)

def compute_diff_modes(self,true,diff,bins,relative=False):
    modes = []
    for x_min,x_max in zip(bins[:-1],bins[1:]):
        mask = torch.logical_and(true<=x_max,true>=x_min)
        y = diff[mask]
        if relative:
            y /= true[mask]
        if y.sum() == 0:
            continue
        y_binned,y_bins = torch.histogram(y,bins=21)
        # https://www.cuemath.com/data/mode-of-grouped-data/
        y_idxmax = y_binned.argmax()
        f0 = y_binned[max(0,y_idxmax-1)]
        f1 = y_binned[y_idxmax]
        f2 = y_binned[min(len(y_binned)-1,y_idxmax+1)]
        h = y_bins[1]-y_bins[0]
        L = y_bins[y_idxmax]
        modes.append(L + (f1-f0)/(2*f1-f0-f2) * h)
    return torch.tensor(modes)


def integral(cont,bins):
    w = bins[1:]-bins[:-1]
    return (cont*w).sum()

def plot_ratio(fig,gs,true,sample,bins,density=True,label=None,log_scale=False):

    # Make axes #
    gs_sub = GridSpecFromSubplotSpec(
        nrows = 2,
        ncols = 1,
        subplot_spec = gs,
        height_ratios = [1.0,0.2],
    )
    ax1 = fig.add_subplot(gs_sub[0,0])
    ax2 = fig.add_subplot(gs_sub[1,0])

    # Compute histogram content #
    true_content = np.histogram(true,bins)[0].astype(np.float32)
    true_var     = np.sqrt(true_content)
    if density:
        true_integral = integral(true_content,bins)
        true_var      = true_var / true_integral
        true_content  = true_content / true_integral
    true_up = true_content + true_var
    true_down = true_content - true_var

    sample_content = np.histogram(sample,bins)[0].astype(np.float32)
    sample_var     = np.sqrt(sample_content)
    if density:
        sample_integral = integral(sample_content,bins)
        sample_var      = sample_var / sample_integral
        sample_content  = sample_content / sample_integral
    sample_up = sample_content + sample_var
    sample_down = sample_content - sample_var

    ratio = sample_content / (true_content+EPS)
    ratio_true_var = true_var / (true_content+EPS)
    ratio_true_up = 1+abs(ratio_true_var)
    ratio_true_down = 1-abs(ratio_true_var)
    ratio_sample_var = sample_var / (true_content+EPS)
    ratio_sample_up = ratio + ratio_sample_var
    ratio_sample_down = ratio - ratio_sample_var

    # Hist #
    ax1.stairs(
        values = true_content,
        edges = bins,
        color = 'royalblue',
        label = 'Truth',
        linewidth = 2,
    )
    ax1.fill_between(
        x = bins,
        y1 = np.r_[true_down,true_down[-1]],
        y2 = np.r_[true_up,true_up[-1]],
        color = 'royalblue',
        alpha = 0.5,
        step = 'post',
    )
    ax1.stairs(
        values = sample_content,
        edges = bins,
        color = 'orange',
        label = 'Generated',
        linewidth = 2,
    )
    ax1.fill_between(
        x = bins,
        y1 = np.r_[sample_down,sample_down[-1]],
        y2 = np.r_[sample_up,sample_up[-1]],
        color = 'orange',
        alpha = 0.5,
        step = 'post',
    )
    # Ratio #
    ax2.step(
        x = bins,
        y = np.r_[ratio[0],ratio],
        linewidth = 2,
        color = 'orange',
    )
    ax2.fill_between(
        x = bins,
        y1 = np.r_[ratio_true_down,ratio_true_down[-1]],
        y2 = np.r_[ratio_true_up,ratio_true_up[-1]],
        color = 'royalblue',
        alpha = 0.5,
        step = 'post',
    )
    ax2.fill_between(
        x = bins,
        y1 = np.r_[ratio_sample_down,ratio_sample_down[-1]],
        y2 = np.r_[ratio_sample_up,ratio_sample_up[-1]],
        color = 'orange',
        alpha = 0.5,
        step = 'post',
    )

    # Esthetic #
    ax1.legend(loc='upper right',fontsize=12)
    ax1.set_xticklabels([])
    ax1.set_ylabel('Frequency',fontsize=14)

    ax2.set_ylim(0.5,1.5)
    ax2.grid(visible=True,which='major',axis='y')
    if label is not None:
        ax2.set_xlabel(label,fontsize=14)
    ax2.set_ylabel(r'$\frac{\text{Generated}}{\text{True}}$',fontsize=14)
    if log_scale:
        ax1.set_yscale('log')
        ax1.set_ylim(
            min(true_content[true_content>0].min(),sample_content[sample_content>0].min()),
            max(true_content.max(),sample_content.max())*2.,
        )
    else:
        ax1.set_ylim(0,max(true_content.max(),sample_content.max())*1.4)

def plot_2D_projections(fig,gs,true,sample,bins,hexbin=False,kde=False,label_x=None,label_y=None,log_scale=False):
    gs_sub = GridSpecFromSubplotSpec(
        nrows = 3,
        ncols = 2,
        width_ratios = (3, 1),
        height_ratios = (2, 5, 2),
        wspace = 0.05,
        hspace = 0.05,
        subplot_spec = gs,
    )
    ax       = fig.add_subplot(gs_sub[1, 0])
    ax_histx = fig.add_subplot(gs_sub[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs_sub[1, 1], sharey=ax)
    ax_bar   = fig.add_subplot(gs_sub[2, 0])

    # 2D plot #
    if hexbin:
        h = ax.hexbin(
            true,
            sample,
            gridsize = len(bins) - 1,
            mincnt = 1,
            extent = [bins[0],bins[-1],bins[0],bins[-1]],
            bins = 'log' if log_scale else None,
        )
        fig.colorbar(
            h,
            ax = ax_bar,
            location = 'bottom',
        )
    else:
        h = ax.hist2d(
            true,
            sample,
            bins = (bins,bins),
            cmin = 1,
            norm = matplotlib.colors.LogNorm() if log_scale else None,
        )
        fig.colorbar(
            h[3],
            ax = ax_bar,
            location = 'bottom',
        )

    ax.set_xlabel(f'{label_x} (true)',fontsize=14)
    ax.set_ylabel(f'{label_y} (generated)',fontsize=14)

    # Projection plots #
    hx = ax_histx.hist(
        true,
        bins = bins,
        color = 'royalblue',
        alpha = 0.7,
    )
    hy = ax_histy.hist(
        sample,
        bins = bins,
        orientation = 'horizontal',
        color = 'royalblue',
        alpha = 0.7,
    )
    if kde:
        x,y = make_kde_xy(true,hx,1000)
        ax_histx.plot(x,y,color='royalblue',linewidth=2)
        x,y = make_kde_xy(sample,hy,1000)
        ax_histy.plot(y,x,color='royalblue',linewidth=2)
    if log_scale:
        ax_histx.set_yscale('log')
        ax_histy.set_xscale('log')
        ax_histx.set_ylim(1e0,hx[0].max()*10)
        ax_histy.set_xlim(1e0,hy[0].max()*10)
    else:
        ax_histx.set_ylim(0,hx[0].max()*1.1)
        ax_histy.set_xlim(0,hy[0].max()*1.1)
    ax_histx.tick_params(axis="both", which='both', labelbottom=False,labelleft=False)
    ax_histy.tick_params(axis="both", which='both', labelbottom=False,labelleft=False)
    ax_bar.axis('off')

def plot_diff(ax,true,diff,points,label=None,relative=False):
    # Get bins based on quantiles #
    bins = torch.quantile(true,q=torch.linspace(0.,1.,points+1).to(true.dtype))

    # Compute quantities to plot #
    centers,coverages = compute_diff_coverage(
        true = true,
        diff = diff,
        bins = bins,
        relative = relative,
    )
    means = compute_diff_means(
        true = true,
        diff = diff,
        bins = bins,
        relative = relative,
    )

    # Plot
    ax.plot(
        centers,
        means,
        linestyle='dashed',
        linewidth=2,
        marker='o',
        markersize = 2,
        color='k',
        label="mean",
    )
    ax.plot(
        centers,
        coverages[:,2],
        linestyle='-',
        marker='o',
        markersize = 2,
        color='k',
        label="median",
    )
    ax.fill_between(
        x = centers,
        y1 = coverages[:,1],
        y2 = coverages[:,3],
        color='r',
        alpha = 0.2,
        label="68% quantile",
    )
    ax.fill_between(
        x = centers,
        y1 = coverages[:,0],
        y2 = coverages[:,4],
        color='b',
        alpha = 0.2,
        label="95% quantile",
    )
    cov_max = abs(coverages).max()
    ax.set_ylim(-cov_max,cov_max)
    ax.legend(loc='upper right',facecolor='white',framealpha=1)
    ax.set_xlabel(f'{label} (true)',fontsize=14)
    label = label.replace('$','')
    if relative:
        ax.set_ylabel(fr'$\frac{{{label} \text{{ (generated)}} - {label}(true)}}{{{label} \text{{ (true)}}}}$',fontsize=14)
    else:
        ax.set_ylabel(fr'${label} \text{{ (generated)}} - {label} \text{{ (true)}}$ [GeV]',fontsize=14)


def plot_quantiles(ax,true,sample,points,labels):
    assert sample.shape[-1] == true.shape[-1]
    assert sample.shape[-1] == len(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(labels)))
    for j in range(len(labels)):
        # Calculate qq plot #
        true_quantiles = torch.linspace(0,1,points)
        true_cuts = torch.quantile(true[:,j],true_quantiles)
        sampled_quantiles = torch.zeros_like(true_quantiles)
        for k,cut in enumerate(true_cuts):
            sampled_quantiles[k] = (sample[:,j].ravel() <= cut).sum() / sample.shape[0]
        # Plot #
        ax.plot(
            true_quantiles,
            sampled_quantiles,
            marker = 'o',
            markersize = 3,
            color = colors[j],
            label = labels[j],
        )
    ax.plot(
        [0,1],
        [0,1],
        linestyle = '--',
        color = 'k',
    )
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel('Quantile')
    ax.set_ylabel('Fraction sampled events')
    ax.legend(loc='lower right',fontsize=16)

def pairplot(true,sample,features,labels,title,bins,feature_rng={},log_scale=False,hexbin=False,kde=False):
    assert sample.dim() == 2
    assert true.dim() == 1
    assert sample.shape[-1] == true.shape[-1]
    N = true.shape[-1]

    fig,axs = plt.subplots(N,N,figsize=(4*N,3*N))
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,hspace=0.3,wspace=0.4)
    fig.suptitle(title,fontsize=16)

    for i in range(N):
        if feature_rng is not None and features[i] in feature_rng.keys():
            bins_x = np.linspace(*feature_rng[features[i]],bins)
        else:
            bins_x = get_bins(arrays=[sample[:,i],true[i]],feature=features[i],N=bins)
        label_x = f'{labels[i]}'
        for j in range(N):
            if feature_rng is not None and features[j] in feature_rng.keys():
                bins_y = np.linspace(*feature_rng[features[j]],bins)
            else:
                bins_y = get_bins(arrays=[sample[:,j],true[j]],feature=features[j],N=bins)
            label_y = f'{labels[j]}'
            if j > i:
                axs[i,j].axis('off')
            elif i == j:
                h = axs[i,j].hist(
                    sample[:,i],
                    bins = bins_x,
                    color = 'royalblue',
                    alpha = 0.7,
                )
                if kde:
                    x,y = make_kde_xy(sample[:,i],h,1000)
                    axs[i,j].plot(
                        x,y,
                        color = 'royalblue',
                        linewidth = 2,
                    )
                axs[i,j].axvline(true[i],color='r',linewidth=2)
                axs[i,j].set_xlabel(label_x,fontsize=14)
                axs[i,j].set_xlim(bins_x[0],bins_x[-1])
                if log_scale:
                    axs[i,j].set_yscale('log')
                    axs[i,j].set_ylim(1e0,h[0].max()*2)
                else:
                    axs[i,j].set_ylim(0,h[0].max()*1.1)
            else:
                if hexbin:
                    h = axs[i,j].hexbin(
                        sample[:,i],
                        sample[:,j],
                        gridsize = bins,
                        mincnt = 1,
                        extent = [bins_x[0],bins_x[-1],bins_y[0],bins_y[-1]],
                        bins = 'log' if log_scale else None
                    )
                    plt.colorbar(h, ax=axs[i,j])
                else:
                    h = axs[i,j].hist2d(
                        sample[:,i],
                        sample[:,j],
                        bins = (bins_x,bins_y),
                        norm = matplotlib.colors.LogNorm() if log_scale else None,
                    )
                    plt.colorbar(h[3], ax=axs[i,j])
                axs[i,j].scatter(true[i],true[j],marker='x',color='r',s=40)
                axs[i,j].set_xlabel(label_x,fontsize=14)
                axs[i,j].set_ylabel(label_y,fontsize=14)
                axs[i,j].set_xlim(bins_x[0],bins_x[-1])
                axs[i,j].set_ylim(bins_y[0],bins_y[-1])
    return fig,axs


