import logging
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
from IPython.display import display
from ast import literal_eval

from scipy.stats import gaussian_kde, norm
from scipy.optimize import curve_fit

# Create logger for error handling
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


project_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(project_dir, 'results')
fig_dir = os.path.join(project_dir, "figs")

algorithms = [f for f in os.listdir(results_dir)]

def plot_layout(cnt):
    """
    Returns the layout for the plots
    """
    if cnt % 2 == 0:
        nrows, ncols = cnt // 2, 2
    elif cnt % 3 == 0:
        nrows, ncols = cnt // 3, 3
    else:
        nrows, ncols = cnt, 1

    size = (5*ncols, 5*nrows)
    fig, axs = plt.subplots(nrows, ncols, figsize=(size)) 
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
            rt_data = pd.read_csv(os.path.join(results_dir, retro_alg, "Round-trip.csv"))
            rt_data = rt_data.iloc[:]
            rt_cov = rt_data.groupby('cov_total').size().div(len(rt_data))
            rt_cov = rt_cov.to_dict()
            rt_cov = rt_cov[1]
            alg_cov[f'{retro_alg}'] = rt_cov
            bins = np.linspace(0, 1, 11)
            rt_data['bin'] = pd.cut(rt_data['acc_mean'], bins=bins, include_lowest=False)
            rt_acc = rt_data.loc[:,['acc_mean', 'bin']]
            counts = rt_acc.groupby('bin').count()
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
            ax.set_title(name)
            ax.set_xlabel("Accuracy")
            ax.set_ylabel("Frequency")
            # Add horizontal line to plot
            x = alg_acc_mean[f'{name}']
            # Add mean to legend
            ax.axvline(x=x, color='r', linestyle='dashed', linewidth=1)
            ax.legend([f'Mean: {x:.2f}'])
            
        except Exception as e:
           pass
    
    fig.suptitle("Round-trip Accuracy", fontsize=16)
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
            sc_data = pd.read_csv(os.path.join(results_dir, retro_alg, "SCScore.csv"))
            alg_sc[f'{retro_alg}'] = sc_data['SCScore']
            alg_sc_mean[f'{retro_alg}'] = sc_data['SCScore'].mean()
            cnt +=1
        except:
            logger.error(f'{retro_alg} does not have a ScScore.csv file')
            continue
    
    # Plot results and save plot
    nrows, ncols, fig, axs = plot_layout(cnt)  

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
            ax.set_title(name)
            ax.set_xlabel("ScScore Difference")
            ax.set_ylabel("Density")
            # Add horizontal line to plot
            x = alg_sc_mean[f'{name}']
            # Add mean to legend
            ax.axvline(x=x, color='r', linestyle='dashed', linewidth=1, label=f'Mean={x:.2f}')
            popt, _ = curve_fit(skewed_normal_pdf,xs,density, maxfev=5000)
            y_fit = skewed_normal_pdf(xs, *popt)
            ax.plot(xs,y_fit,'k',label='Fit')
            ax.set_xlim(-1.5,2)
            ax.legend()
        except Exception as e:
            pass
    
    fig.suptitle("ScScore Difference",fontsize=16)
    plt.savefig(os.path.join(fig_dir, 'scscore.png'), dpi=300, bbox_inches='tight')

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
            div_data = pd.read_csv(os.path.join(results_dir, retro_alg, "Diversity.csv"))
            alg_div_mean[f'{retro_alg}'] = div_data['No_classes'].mean()
            bins = list(range(0, 9))
            div_data['bin'] = pd.cut(div_data['No_classes'], bins=bins, include_lowest=True)
            div = div_data.loc[:,['No_classes', 'bin']]
            counts = div.groupby('bin').count()
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
            ax.set_title(name)
            ax.set_xlabel("No. of Reaction Classes per Target")
            ax.set_ylabel("Frequency")
            # Add horizontal line to plot
            x = alg_div_mean[f'{name}']
            # Modify the x-axis labels for 1-9 number of classes 
            ax.set_xticklabels([f'{i}' for i in range(1, 9)])
            # Add mean to legend
            ax.axvline(x=x, color='r', linestyle='dashed', linewidth=1)
            ax.legend([f'Mean: {x:.2f}'])
        except Exception as e:
            pass
    
    fig.suptitle("Diversity of reaction prediction", fontsize=16)
    plt.savefig(os.path.join(fig_dir, 'div.png'), dpi=300, bbox_inches='tight')

    return alg_div_mean

def plot_div_dist(algorithms=algorithms, fig_dir=fig_dir, results_dir=results_dir):
    
    cnt = 0
    alg_div_dist = {alg:{} for alg in algorithms}

    for retro_alg in algorithms:
        class_dist = {str(i):0 for i in range(1,11)}
        try:
            div_data = pd.read_csv(os.path.join(results_dir, retro_alg, "Diversity.csv"))
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
            alg_div_dist[f'{retro_alg}'] = class_dist
        except:
            logger.error(f'{retro_alg} does not have a Diversity.csv file')
            continue
    # Plot results and save plot
    nrows, ncols = 2, 5 
    size = (5*ncols, 5*nrows)
    fig, axs = plt.subplots(nrows, ncols, figsize=(size)) 
    axs = axs.reshape((nrows, ncols)) 

    # Group data by number of classes 
    for i, (name, div) in enumerate(alg_div_dist.items()):
        row_idx, col_idx = divmod(i, ncols)
        try: 
            keys, value = zip(*div.items())
            ax = axs[row_idx, col_idx]
            ax.bar(keys, value, edgecolor='black', align='center', width=1)
            ax.set_title(name)
            ax.set_xlabel("Reaction Class")
            ax.set_ylabel("Frequency")
            # Turn x axis into log
        except Exception as e:
            print(e)
    
    fig.suptitle("Diversity distribution", fontsize=16)
    plt.savefig(os.path.join(fig_dir, 'div_dist.png'), dpi=300, bbox_inches='tight')

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
            dup_data = pd.read_csv(os.path.join(results_dir, retro_alg, "Duplicates.csv"))
            alg_dup_mean[f'{retro_alg}'] = dup_data['dup'].mean()
            bins = np.linspace(0, 1, 11)
            dup_data['bin'] = pd.cut(dup_data['dup'], bins=bins, include_lowest=False)
            rt_acc = dup_data.loc[:,['dup', 'bin']]
            counts = rt_acc.groupby('bin').count()
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
            ax.set_title(name)
            ax.set_xlabel("Duplicate Index")
            ax.set_ylabel("Frequency")
            # Add horizontal line to plot
            x = alg_dup_mean[f'{name}']
            # Add mean to legend
            ax.axvline(x=x, color='r', linestyle='dashed', linewidth=1)
            ax.legend([f'Mean: {x:.2f}'])
            
        except Exception as e:
           pass

    fig.suptitle("Duplicate Reaction Index", fontsize=16)
    plt.savefig(os.path.join(fig_dir, 'dup.png'))
    return alg_dup_mean

def table_round_trip(algorithms=algorithms, results_dir=results_dir):
    """
    Make a summary table for top-k round-trip metrics
    """
    list_rt = []
    for retro_alg in algorithms:
        try:
            rt_data = pd.read_csv(os.path.join(results_dir, retro_alg, "Round-trip.csv"))
            # compute mean for each column
            rt_data_mean = rt_data.mean(axis=0)
            # round values to 3 decimal places
            rt_data_mean = rt_data_mean.round(2)
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
            inv_data = pd.read_csv(os.path.join(results_dir, retro_alg, "InvSmiles.csv"), names=['Top-k', 'value'], header=0, index_col=False)
            inv_data = inv_data.T
            inv_data.columns = inv_data.iloc[0]
            values = inv_data.iloc[1:2, k].values
            alg_inv[f'{retro_alg}'] = pickle.load(open(os.path.join(results_dir, retro_alg, 'Inv_smi.pickle'), 'rb'))
        except Exception as e:
            print(e)
            logger.error(f'{retro_alg} does not have InvSmiles.csv file or Inv_smi.pickle file')
            continue
        dict_inv[f'{retro_alg}'] = values[0]
    
    # Make dataframe from dictionaries
    df = pd.DataFrame(dict_inv)
    df = df.T
    df.columns = ['Top-1', 'Top-3', 'Top-5', 'Top-10', 'Top-20']
    df["Total %"] = [np.round(alg_inv[f'{alg}'],2) for alg in algorithms]
    print("\nInvalid Smiles Table:")
    display(df)
    df.to_csv(os.path.join(fig_dir, 'inv_smi_table.csv'))

def table_topk(algorithms=algorithms, results_dir=results_dir):
    """
    The standard retrosynthesis top-k accuracy 
    """
    alg_topk = {alg:[] for alg in algorithms}
    for retro_alg in algorithms:
        k = ['Top_1','Top_3','Top_5','Top_10','Top_20']
        try:
            topk_data = pd.read_csv(os.path.join(results_dir, retro_alg, "Top-k.csv"))
            alg_topk[f'{retro_alg}'] = np.round(topk_data.iloc[0,1:].to_list(),2)
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

def make_sum_table(alg_acc_mean, alg_cov, alg_sc_mean, alg_div_mean, alg_dup_mean, fig_dir=fig_dir):
    """ 
    Make summary table of results 
    """
    # Make dataframe from dictionaries
    df = pd.DataFrame([alg_acc_mean, alg_cov, alg_sc_mean, alg_div_mean, alg_dup_mean])
    # Transpose dataframe
    df = df.T
    # Rename columns
    df.columns = ['Rt Accuracy', 'Rt Cov', 'ScScore', 'Diversity', 'Duplicate Index']
    # Round values
    df = df.round(2)
    # Save table
    df.to_csv(os.path.join(fig_dir, 'summary_table.csv'))
    print("\nSummary Table:")
    display(df)
    return df

if __name__ == '__main__':
    # Plot results
    alg_div_mean = plot_div()
    alg_cov, alg_acc_mean = plot_rt()
    alg_sc_mean = plot_sc()
    alg_dup_mean = plot_dup()
    # Make summary table
    table_topk()
    table_invalid_smi()
    table_round_trip()
    df = make_sum_table(alg_acc_mean, alg_cov, alg_sc_mean, alg_div_mean, alg_dup_mean)