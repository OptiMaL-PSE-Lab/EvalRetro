import json
import logging
import math
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from ast import literal_eval

from scipy.stats import gaussian_kde, norm, skewnorm
from scipy.spatial.distance import jensenshannon
from scipy.optimize import curve_fit

# Create logger for error handling
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

colors = [
    (31/255, 119/255, 180/255),
    (255/255, 127/255, 14/255),
    (44/255, 160/255, 44/255),
    (214/255, 39/255, 40/255),
    (148/255, 103/255, 189/255),
    (140/255, 86/255, 75/255),
    (227/255, 119/255, 194/255),
    (127/255, 127/255, 127/255),
    (188/255, 189/255, 34/255),
    (23/255, 190/255, 207/255),
    (174/255, 199/255, 232/255),
    (255/255, 187/255, 120/255)
]

project_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(project_dir, 'results')
fig_dir = os.path.join(project_dir, "figs")
config_path = os.path.join(project_dir, "config")

with open(os.path.join(config_path,"par.json"), 'r') as f:
    configs = json.load(f)

alg_data = [(configs[alg]["name"], configs[alg]["type"]) for alg in configs.keys()]
# sort alg_data by type
alg_data.sort(key=lambda x: x[1])
algorithms = [alg[0] for alg in alg_data]

def my_round(x, decimals=2):
    multiplier = 10 ** decimals
    threshold = 0.5 / multiplier
    rounded_value = round(x * multiplier)
    if rounded_value - x * multiplier < threshold:
        return math.floor(x * multiplier) / multiplier
    return math.ceil(x * multiplier) / multiplier

def plot_layout(cnt):
    """
    Returns the layout for the plots
    """
    if cnt % 3 == 0:
        nrows, ncols = 3, cnt // 3
    elif cnt % 2 == 0:
        nrows, ncols = cnt // 2, 2
    else:
        nrows, ncols = cnt, 1

    size = (6*ncols, 6*nrows)
    fig, axs = plt.subplots(nrows, ncols, figsize=(size))
    fig.tight_layout(pad=5.0)
    axs = axs.reshape((nrows, ncols))  

    return nrows, ncols, fig, axs

def skewed_normal_pdf(x, mean, std, skewness):
    # Calculate the z-score
    z = (x - mean) / std

    # Calculate the PDF of the standard normal distribution
    pdf_standard_normal = norm.pdf(z)

    # Apply the skewness parameter to the PDF
    pdf_skewed_normal = 2 / std * pdf_standard_normal * norm.cdf(skewness * z)

    return pdf_skewed_normal

def skewed_normal_cdf(x, mean, std_dev, skew):
    # Create a skew-normal distribution object
    skew_normal = skewnorm(skew, loc=mean, scale=std_dev)

    # Calculate the CDF for the given value
    cdf = skew_normal.cdf(x)

    return cdf

def plot_rt(algorithms=algorithms, fig_dir=fig_dir, results_dir=results_dir):
    """
    Plots the round-trip metrics for each algorithm
    """
    cnt = 0
    alg_acc = {alg:[] for alg in algorithms}
    alg_cov = {alg:[] for alg in algorithms}
    alg_acc_mean = {alg:[] for alg in algorithms}
    # Extract results from each algorithm
    for retro_alg in algorithms:
        try:
            rt_data = pd.read_csv(os.path.join(results_dir, retro_alg.lower(), "Round-trip.csv"))
            rt_data = rt_data.iloc[:]
            rt_cov = rt_data.groupby('cov_total').size().div(len(rt_data))
            rt_cov = rt_cov.to_dict()
            rt_cov = rt_cov[1]
            alg_cov[f'{retro_alg}'] = rt_cov
            bins = np.linspace(0, 1, 11)
            rt_data['bin'] = pd.cut(rt_data['acc_mean'], bins=bins, include_lowest=False)
            rt_acc = rt_data.loc[:,['acc_mean', 'bin']]
            counts = rt_acc.groupby('bin', observed=False).count()
            # Get extra column where the bin is set to min of bin range
            counts['bin'] = counts.index.map(lambda x: x.left)
            alg_acc[f'{retro_alg}'] = counts
            cnt +=1
            alg_acc_mean[f'{retro_alg}'] = rt_data['acc_mean'].mean()
        except:
            logger.error(f'{retro_alg} does not have a Round-trip.csv file')
            continue
    
    # Plot results and save plot
    nrows, ncols, fig, axs = plot_layout(cnt)  

    for i, (name, acc) in enumerate(alg_acc.items()):
        row_idx, col_idx = divmod(i, ncols)
        try:
            ax = axs[row_idx, col_idx]
            ax.bar(x=acc.bin, height=acc.acc_mean, edgecolor='black', align='edge', width=0.1)
            ax.set_title(name, fontsize=16)
            ax.set_xlabel("Accuracy", fontsize=14)
            ax.set_ylabel("Frequency", fontsize=14)
            # Add horizontal line to plot
            x = alg_acc_mean[f'{name}']
            # Add mean to legend
            ax.axvline(x=x, color='r', linestyle='dashed', linewidth=1)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.legend([f'Mean: {x:.2f}'])
            
        except Exception as e:
           pass
    plt.savefig(os.path.join(fig_dir, 'rt_acc.png'), dpi=300, bbox_inches='tight')

    return alg_cov, alg_acc_mean

def plot_sc(algorithms=algorithms, fig_dir=fig_dir, results_dir=results_dir):
    """  
    Plot the ScScore metrics for each algorithm
    """
    # Plot pdf for distributions
    cnt = 0
    alg_sc = {alg:[] for alg in algorithms}
    alg_sc_mean = {alg:[] for alg in algorithms}

    for retro_alg in algorithms:
        try:
            sc_data = pd.read_csv(os.path.join(results_dir, retro_alg.lower(), "SCScore.csv"))
            alg_sc[f'{retro_alg}'] = sc_data['SCScore']
            alg_sc_mean[f'{retro_alg}'] = sc_data['SCScore'].mean()
            cnt +=1
        except:
            logger.error(f'{retro_alg} does not have a ScScore.csv file')
            continue
    
    # Plot results and save plot
    nrows, ncols, fig, axs = plot_layout(cnt)  
    popts = []
    names = []
    # Fit gaussain kde to each data point in retro_alg and plot
    for i, (name, sc) in enumerate(alg_sc.items()):
        row_idx, col_idx = divmod(i, ncols)
        try:
            kde =  gaussian_kde(sc, bw_method='silverman')
            xs = np.linspace(min(sc), max(sc), 200, endpoint=False)
            density = kde(xs)
            density = density/np.trapz(density, xs)
            ax = axs[row_idx, col_idx]
            ax.scatter(xs, density, s=8)
            ax.set_title(name, fontsize=16)
            ax.set_xlabel("ScScore Difference", fontsize=14)
            ax.set_ylabel("Density", fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=12)
            # Add horizontal line to plot
            x = alg_sc_mean[f'{name}']
            # Add mean to legend
            popt, _ = curve_fit(skewed_normal_pdf,xs,density, maxfev=5000)
            popts.append(popt), names.append(name)
            std, skew = popt[1], popt[2]
            ax.axvline(x=x, color='r', linestyle='-', linewidth=1, label=f'Mean={x:.2f}')
            # Add std to plot as line from mean with length of +/- std
            ax.axvline(x=x+std, color='r', linestyle='dashed', linewidth=0.5, label=f'Std={std:.2f}')
            ax.axvline(x=x-std, color='r', linestyle='dashed', linewidth=0.5)
            y_fit = skewed_normal_pdf(xs, *popt)
            ax.plot(xs,y_fit,'k',label=f'Fit (Swd:{skew:.1f})')
            ax.set_xlim(-1.5,2)

            ax.legend(loc='upper left')

        except Exception as e:
            pass
    
    plt.savefig(os.path.join(fig_dir, 'scscore_pdf.png'), dpi=300, bbox_inches='tight')
    
    fig1, ax1 = plt.subplots()
    ax1.set_ylabel('Probability', fontsize=10)
    ax1.set_xlabel('SCScore Difference', fontsize=10)
    fig1.set_size_inches([7,6])
    left_1, bottom_1, left_2, bottom_2, width, heigth = [0.6, 0.5, 0.46, 0.18, 0.27, 0.27]
    ax2 = fig1.add_axes([left_1, bottom_1, width, heigth])
    ax3= fig1.add_axes([left_2, bottom_2, width, heigth])
    ax1.set_prop_cycle(color=colors)
    ax2.set_prop_cycle(color=colors)
    ax3.set_prop_cycle(color=colors)
    ax3.set_yscale('log')

    for popt,name in zip(popts,names):
        x = np.linspace(-1.5,2,200)
        y = skewed_normal_cdf(x, *popt)
        ax1.plot(x,y, label=name, linewidth=1.3)
        ax2.plot(x,y, linewidth=0.8)
        ax3.plot(x,y, linewidth=0.8)
    ax1.set_xlim([-0.5, 2])
    ax2.set_xlim([0.9,1.4])
    ax2.set_ylim([0.85,1])
    ax3.set_xlim([-0.4,0])
    ax3.set_ylim([0.01,0.12])
    ax2.set_yticks(np.arange(0.9, 1, 0.05))
    ax2.tick_params(axis='both', which='both', labelsize=7)
    ax3.tick_params(axis='both', which='both', labelsize=7)

    ax1.legend(fontsize=9)
    # draw box around inset
    mark_inset(ax1, ax2, loc1=2, loc2=1, linestyle='--', linewidth=0.5)
    mark_inset(ax1, ax3, loc1=2, loc2=3, linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(fig_dir, 'scscore_cdf.png'), dpi=300, bbox_inches='tight')

    return alg_sc_mean

def plot_div(algorithms=algorithms, fig_dir=fig_dir, results_dir=results_dir):
    """ 
    Plot the Diversity metrics for each algorithm
    """
    cnt = 0
    alg_div = {alg:[] for alg in algorithms}
    alg_div_mean = {alg:[] for alg in algorithms}

    for retro_alg in algorithms:
        try:
            div_data = pd.read_csv(os.path.join(results_dir, retro_alg.lower(), "Diversity.csv"))
            alg_div_mean[f'{retro_alg}'] = div_data['No_classes'].mean()
            bins = list(range(0, 9))
            div_data['bin'] = pd.cut(div_data['No_classes'], bins=bins, include_lowest=True)
            div = div_data.loc[:,['No_classes', 'bin']]
            counts = div.groupby('bin', observed=False).count()
            alg_div[f'{retro_alg}'] = counts
            cnt +=1
        except:
            logger.error(f'{retro_alg} does not have a Diversity.csv file')
            continue
    # Plot results and save plot
    nrows, ncols, fig, axs = plot_layout(cnt)  
    # Group data by number of classes 
    for i, (name, div) in enumerate(alg_div.items()):
        row_idx, col_idx = divmod(i, ncols)
        try: 
            ax = axs[row_idx, col_idx]
            div.plot.bar(ax = ax, edgecolor='black', align='center', width=1)
            ax.set_title(name, fontsize=16)
            ax.set_xlabel("No. of Reaction Classes per Target", fontsize=14)
            ax.set_ylabel("Frequency", fontsize=14)
            # Add horizontal line to plot
            x = alg_div_mean[f'{name}']
            # Modify the x-axis labels for 1-9 number of classes 
            ax.set_xticklabels([f'{i}' for i in range(1, 9)])
            # Add mean to legend
            ax.axvline(x=x, color='r', linestyle='dashed', linewidth=1)
            ax.legend([f'Mean: {x:.2f}'])
        except Exception as e:
            pass
    
    plt.savefig(os.path.join(fig_dir, 'div.png'), dpi=300, bbox_inches='tight')

    return alg_div_mean

def plot_div_dist(algorithms=algorithms, fig_dir=fig_dir, results_dir=results_dir):
    
    cnt = 0
    alg_div_dist = {f"{i}":[] for i in range(1,11)}
    alg_sim = {f"{alg}":0 for alg in algorithms}
    
    rxn_freq = [0.303,0.238,0.113,0.018,0.013,0.165,0.092,0.016,0.037,0.005]
    for retro_alg in algorithms:
        class_dist = {str(i):0 for i in range(1,11)}
        try:
            div_data = pd.read_csv(os.path.join(results_dir, retro_alg.lower(), "Diversity.csv"))
            for row in div_data.iterrows():
                classes = row[1]['Classes']
                classes = literal_eval(classes)
                counts = row[1]["Counts"]
                counts = literal_eval(counts)
                class_dict = dict(zip(classes,counts))
                for key, value in class_dict.items():
                    class_dist[f"{key}"]+= value
            # Update class_dist with counts from each dict in dicts
            cnt +=1
            # get total count of all classes in class_dist
            total = sum(class_dist.values())
            # divide each value in class_dist by total to get frequency
            class_dist = {k: v/total for k, v in class_dist.items()}
            # put each value for key 
            for key, value in class_dist.items():
                alg_div_dist[f"{key}"].append(value)

            # finally calculate the similarity between the two distributions 
            # using the Jensen-Shannon divergence
            freq = np.array(list(class_dist.values()))
            freq_true = np.array(rxn_freq)
            jsd = jensenshannon(freq, freq_true)
            similarity = 1 - jsd
            similarity = round(similarity, 2)
            alg_sim[f"{retro_alg}"] = similarity
        except:
            logger.error(f'{retro_alg} does not have a Diversity.csv file')
            continue
    # Plot results and save plot
    nrows, ncols = 2, 5
    size = (5*ncols, 5*nrows)
    fig, axs = plt.subplots(nrows, ncols, figsize=(size))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    axs = axs.reshape((nrows, ncols)) 
    rxn_classes = ["Heteroatom Alkylation", "Acylation", "C-C Formation", "Heterocycle Formation", "Protections", "Deprotections", "Reductions", "Oxidations", "FG Interconversion", "FG Addition"]
    
    for i, (_, div) in enumerate(alg_div_dist.items()):
        row_idx, col_idx = divmod(i, ncols)
        try: 
            ax = axs[row_idx, col_idx]
            bars = ax.bar(algorithms, div, edgecolor='black', align='center', width=1)
            ax.set_title(f"{rxn_classes[i]}", fontsize=14)
            # rotate x axis labels
            ax.set_ylabel("Frequency")
            new_bars = ax.bar("Ground-truth", rxn_freq[i], width=1, color='orange', edgecolor='black')
            ax.axhline(y=rxn_freq[i], color='r', linestyle='dashed', linewidth=1)
            new_list = algorithms + ["Ground-truth"]
            ax.set_xticklabels(new_list, rotation=90)
            [z.set_x(z.get_x() + 0.5) for z in new_bars]

        except Exception as e:
            print(e)
    plt.savefig(os.path.join(fig_dir, 'div_dist.png'), dpi=300, bbox_inches='tight')

    return alg_sim

def plot_dup(algorithms=algorithms, fig_dir=fig_dir, results_dir=results_dir):
    """ 
     Plot the duplicate reaction metrics for each algorithm 
    """
    cnt = 0
    alg_dup = {alg:[] for alg in algorithms}
    alg_dup_mean = {alg:[] for alg in algorithms}
    # Extract results from each algorithm
    for retro_alg in algorithms:
        try:
            dup_data = pd.read_csv(os.path.join(results_dir, retro_alg.lower(), "Duplicates.csv"))
            alg_dup_mean[f'{retro_alg}'] = dup_data['dup'].mean()
            bins = np.linspace(0, 1, 11)
            dup_data['bin'] = pd.cut(dup_data['dup'], bins=bins, include_lowest=False)
            rt_acc = dup_data.loc[:,['dup', 'bin']]
            counts = rt_acc.groupby('bin', observed=False).count()
            # Get extra column where the bin is set to min of bin range
            counts['bin'] = counts.index.map(lambda x: x.left)
            alg_dup[f'{retro_alg}'] = counts
            cnt +=1
        except:
            logger.error(f'{retro_alg} does not have a Duplicates.csv file')
            continue
    
    # Plot results and save plot
    nrows, ncols, fig, axs = plot_layout(cnt)  

    for i, (name, acc) in enumerate(alg_dup.items()):
        row_idx, col_idx = divmod(i, ncols)
        try:
            ax = axs[row_idx, col_idx]
            ax.bar(x=acc.bin, height=acc.dup, edgecolor='black', align='edge', width=0.1)
            ax.set_title(name, fontsize=16)
            ax.set_xlabel("Duplicate Index",fontsize=14)
            ax.set_ylabel("Frequency", fontsize=14)
            # Add horizontal line to plot
            x = alg_dup_mean[f'{name}']
            # Add mean to legend
            ax.axvline(x=x, color='r', linestyle='dashed', linewidth=1)
            ax.legend([f'Mean: {x:.2f}'])
            
        except Exception as e:
           pass

    plt.savefig(os.path.join(fig_dir, 'dup.png'))
    return alg_dup_mean

def table_round_trip(algorithms=algorithms, results_dir=results_dir):
    """
    Make a summary table for top-k round-trip metrics
    """
    list_rt = []
    for retro_alg in algorithms:
        try:
            rt_data = pd.read_csv(os.path.join(results_dir, retro_alg.lower(), "Round-trip.csv"))
            # compute mean for each column
            # remove the first column
            rt_data = rt_data.iloc[:,1:]
            rt_data_mean = rt_data.mean(axis=0)
            # round values to 3 decimal places
            rt_data_mean = rt_data_mean.round(3)
            top_k = [1, 3, 5, 10]
            metrics = ["acc","cov"]
            # get mean for each top-k
            rt_acc_mean = [rt_data_mean[f'{met}_top_{k}'] for met in metrics for k in top_k]

        except:
            logger.error(f'{retro_alg} does not have a Round-trip.csv file')
            continue
        list_rt.append(rt_acc_mean)
        
    # Make dataframe from dictionaries
    iterrables = [["Acc", "Cov"], [1,3,5,10]]
    mlt_index = pd.MultiIndex.from_product(iterrables, names=["Metric", "Top-k"])
    df = pd.DataFrame(list_rt, columns=mlt_index, index=algorithms)
    df.to_csv(os.path.join(fig_dir, 'rt_table.csv'))
    print("\nRound-trip table:")
    display(df)


def table_invalid_smi(algorithms=algorithms, results_dir=results_dir):
    """ 
    Make a summary table for invalid smiles metrics
    """
    dict_inv = {alg:[] for alg in algorithms}
    alg_inv = {alg:[] for alg in algorithms}
    for retro_alg in algorithms:
        k = np.array([1,3,5,10,20])-1
        try:
            inv_data = pd.read_csv(os.path.join(results_dir, retro_alg.lower(), "InvSmiles.csv"), names=['Top-k', 'value'], header=0, index_col=False)
            inv_data = inv_data.T
            inv_data.columns = inv_data.iloc[0]
            values = inv_data.iloc[1:2, k].values
            values *= 100
            alg_inv[f'{retro_alg}'] = pickle.load(open(os.path.join(results_dir, retro_alg.lower(), 'Inv_smi.pickle'), 'rb'))
        except Exception as e:
            print(e)
            logger.error(f'{retro_alg} does not have InvSmiles.csv file or Inv_smi.pickle file')
            continue
        dict_inv[f'{retro_alg}'] = values[0]
    
    # Make dataframe from dictionaries
    df = pd.DataFrame(dict_inv)
    df = df.T
    df = df.astype(float).round(2)
    df.columns = ['Top-1', 'Top-3', 'Top-5', 'Top-10', 'Top-20']
    df["Total %"] = [np.round(alg_inv[f'{alg}']*100,2) for alg in algorithms]
    print("\nInvalid Smiles Table:")
    display(df)
    df.to_csv(os.path.join(fig_dir, 'inv_smi_table.csv'))

    return 1 - df["Top-20"]/100

def table_topk(algorithms=algorithms, results_dir=results_dir):
    """
    The standard retrosynthesis top-k accuracy 
    """
    alg_topk = {alg:[] for alg in algorithms}
    for retro_alg in algorithms:
        k = ['Top_1','Top_3','Top_5','Top_10']
        try:
            topk_data = pd.read_csv(os.path.join(results_dir, retro_alg.lower(), "Top-k.csv"))
            topk_data = topk_data.map(lambda x: x*100)
            alg_topk[f'{retro_alg}'] = np.round(topk_data.iloc[0,1:].to_list(),1)
        except Exception as e:
            print(e)
            logger.error(f'{retro_alg} does not have InvSmiles.csv file or Inv_smi.pickle file')
            continue
    df = pd.DataFrame(alg_topk)
    df = df.T
    df.columns = k
    print("\nTop-k Table:")
    display(df)
    df.to_csv(os.path.join(fig_dir, 'topk_table.csv'))

# Make summary table
def make_sum_table(alg_acc_mean, alg_div_mean, alg_val_mean, alg_dup_mean, alg_sc_mean, fig_dir=fig_dir):
    """ 
    Make summary table of results 
    """
    df = pd.DataFrame([alg_acc_mean, alg_div_mean, alg_val_mean, alg_dup_mean, alg_sc_mean])
    df = df.T
    df.columns = ['Rt Accuracy', 'Diversity', 'Validity', 'Duplicity', 'SCScore']
    df = df.map(my_round)
    df.to_csv(os.path.join(fig_dir, 'summary_table.csv'))
    print("\nSummary Table:")
    display(df)
    return df

if __name__ == '__main__':
    # Plot results
    alg_cov, alg_acc_mean = plot_rt()
    alg_div_mean = plot_div()
    alg_sc_mean = plot_sc()
    alg_dup_mean = plot_dup()
    alg_sim = plot_div_dist()
    alg_val_mean = table_invalid_smi()
    # Make summary tables
    table_topk()
    table_round_trip()
    df = make_sum_table(alg_acc_mean, alg_div_mean, alg_val_mean, alg_dup_mean, alg_sc_mean)