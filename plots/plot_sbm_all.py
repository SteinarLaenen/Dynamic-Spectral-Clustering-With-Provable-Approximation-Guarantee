import ast
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import glob


def plot_spectral_clustering_results_all(scale=1.5):
    # Define common plotting parameters
    fig_width = 7  # Adjust as needed
    fig_height = 2.5*2  # Adjust as needed
    base_fontsize = 9.5 * scale
    base_linewidth = 1 * scale
    base_markersize = 1.5 * scale

    # Enable LaTeX-style font rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=base_fontsize)
    plt.rc('axes', titlesize=base_fontsize)
    plt.rc('axes', labelsize=base_fontsize)
    plt.rc('xtick', labelsize=base_fontsize * 0.9)
    plt.rc('ytick', labelsize=base_fontsize * 0.9)
    plt.rc('legend', fontsize=base_fontsize * 0.9)

    # Function to format the y-axis labels with 2 significant digits
    def format_func(value, tick_number):
        return f'{value:.2f}'


    # Setup figure and axes for a 2x2 plot
    fig, axs = plt.subplots(2, 2, figsize=(fig_width, fig_height))

    # Iterate over the two datasets
    for col, dataset in enumerate(['increase_k', 'decrease_k']):
        # Load data based on dataset
        if dataset == "increase_k":
            title = "Increase $k$"
            # ENTER DATA #
            filename_pattern = f"../results/experiment_sbm_new_cluster_experiment_*.txt"
            filenames = glob.glob(filename_pattern)
        elif dataset == "decrease_k":
            title = "Decrease $k$"
            filename_pattern = f"../results/experiment_sbm_merge_experiment_*.txt"
            filenames = glob.glob(filename_pattern)

        experiment_data = load_experiment_data(filenames)

        # Extract data for plotting
        ari_g_avg = experiment_data["FULL_ARI"]['avg']
        ari_h_avg = experiment_data["SPARSE_ARI"]['avg']
        ari_tilde_g_avg = experiment_data["CONTRACT_ARI"]['avg']

        ari_g_std = experiment_data["FULL_ARI"]['std']
        ari_h_std = experiment_data["SPARSE_ARI"]['std']
        ari_tilde_g_std = experiment_data["CONTRACT_ARI"]['std']

        times_full_avg = experiment_data["runtime_full_sc"]['avg']
        times_sparse_avg = experiment_data["runtime_sparse_sc"]['avg']
        times_contract_avg = experiment_data["runtime_contract_sc"]['avg']

        times_full_std = experiment_data["runtime_full_sc"]['std']
        times_sparse_std = experiment_data["runtime_sparse_sc"]['std']
        times_contract_std = experiment_data["runtime_contract_sc"]['std']

        total_timesteps = len(ari_g_avg)
        x_labels = np.arange(1, total_timesteps + 1)

        # Adjust x-axis ticks for both plots

        # Plot for the current dataset
        ax1, ax2 = axs[0, col], axs[1, col]

        ax1.set_xticks([i for i in range(11)])
        ax2.set_xticks([i for i in range(11)])
        ax1.set_xticklabels([str(i + 1) for i in range(11)])
        ax2.set_xticklabels([str(i + 1) for i in range(11)])


        # Plot ARI scores
        ax1.plot(x_labels, ari_g_avg, marker='o', linestyle='-', linewidth=base_linewidth, markersize=base_markersize, label=r'\textsf{SC} on $G_T$')
        ax1.plot(x_labels, ari_h_avg, marker='s', linestyle='--', linewidth=base_linewidth, markersize=base_markersize, label=r'\textsf{SC} on $H_T$')
        ax1.plot(x_labels, ari_tilde_g_avg, marker='^', linestyle='-.', linewidth=base_linewidth, markersize=base_markersize, label=r'\textsf{SC} on $\widetilde{G}_T$')
        if col == 0:
            ax1.set_ylabel('ARI')
        ax1.yaxis.set_major_formatter(FuncFormatter(format_func))
        ax1.set_title(title)
        ax1.set_ylim(0.9, 1.02)

        # Plot Running Times
        ax2.plot(x_labels, times_full_avg, marker='o', linestyle='-', color='tab:blue', linewidth=base_linewidth, markersize=base_markersize, label=r'Full $G$')
        ax2.plot(x_labels, times_sparse_avg, marker='s', linestyle='--', color='tab:orange', linewidth=base_linewidth, markersize=base_markersize, label=r'Sparse $H$')
        ax2.plot(x_labels, times_contract_avg, marker='^', linestyle='-.', color='tab:green', linewidth=base_linewidth, markersize=base_markersize, label=r'Contracted Graph')
        if col == 0:
            ax2.set_ylabel('Time (s)')

        ax2.set_xlabel('$T$')

        # Fill between for error representation
        ax1.fill_between(x_labels, ari_g_avg - ari_g_std, ari_g_avg + ari_g_std, color='blue', alpha=0.2)
        ax1.fill_between(x_labels, ari_h_avg - ari_h_std, ari_h_avg + ari_h_std, color='orange', alpha=0.2)
        ax1.fill_between(x_labels, ari_tilde_g_avg - ari_tilde_g_std, ari_tilde_g_avg + ari_tilde_g_std, color='green', alpha=0.2)
        ax2.fill_between(x_labels, times_full_avg - times_full_std, times_full_avg + times_full_std, color='blue', alpha=0.2)
        ax2.fill_between(x_labels, times_sparse_avg - times_sparse_std, times_sparse_avg + times_sparse_std, color='orange', alpha=0.2)
        ax2.fill_between(x_labels, times_contract_avg - times_contract_std, times_contract_avg + times_contract_std, color='green', alpha=0.2)


    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))

    # handles, labels = axs[0, 0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout(rect=[0, 0, 0.98, 0.98])

    # Adjust layout
    plt.subplots_adjust(hspace=0.2, wspace=0.2)

    # Save the figure
    plt.savefig('all_sbm_results.pdf', format='pdf', dpi=300)

    plt.show()

def load_experiment_data(filenames):
    combined_data = {}
    count = 0

    # Initialize data accumulation
    for filename in filenames:
        with open(filename, 'r') as file:
            for line in file:
                key, values_str = line.split(':')
                values = np.array(ast.literal_eval(values_str.strip()))

                if key not in combined_data:
                    combined_data[key] = {'sum': np.zeros_like(values), 'sq_sum': np.zeros_like(values), 'count': 0}

                combined_data[key]['sum'] += values
                combined_data[key]['sq_sum'] += np.square(values)
                combined_data[key]['count'] += 1

    # Calculate average and standard deviation
    for key in combined_data.keys():
        n = combined_data[key]['count']
        combined_data[key]['avg'] = combined_data[key]['sum'] / n
        combined_data[key]['std'] = np.sqrt(combined_data[key]['sq_sum'] / n - np.square(combined_data[key]['avg']))

    return combined_data


if __name__ == "__main__":
    plot_spectral_clustering_results_all(scale=1.65)
