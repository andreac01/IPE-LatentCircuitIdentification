import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_search_strategy_results(
    df: pd.DataFrame,
    x_axis_col: str = 'execution_time_s',
    y_axis_col: str = 'relevant_paths_found',
    show_labels: bool = True,
    save_plot: bool = False,
    save_path: str = "search_strategy_plot.png"
):
    """
    Generates a scatter/line plot comparing relevant paths found vs. a chosen metric
    (e.g., execution time or memory), grouped by search strategy and type.

    Args:
        df: The input DataFrame containing the experiment results.
        x_axis_col: The column name to use for the x-axis.
        y_axis_col: The column name to use for the y-axis.
        show_labels: If True, adds text annotations (th=, n=, w=) to data points.
        save_plot: If True, saves the plot to a file.
        save_path: The file path where the plot image will be saved.
    """

    # Data Preprocessing
    # Remove from df the rows where relevant_paths_found is 0 or 1
    df_plot = df[df[y_axis_col] > 1].copy()
    df_plot = df_plot.sort_values(by=["type", "search_strategy", y_axis_col])

    # Define a name map for the 'type' and 'search_strategy' columns
    type_name_map = {
        "PathMessagePatchingBatchedHeadsPos": "PMP Batched Heads & Pos",
        "PathMessagePatchingBatchedPos": "PMP Batched Pos",
        "PathMessagePatching": "Path Message Patching",
        "PathAttributionPatching": "Path Attribution Patching"
    }
    search_strategy_name_map = {
        "BestFirstSearch": "Best First Search",
        "Threshold": "Threshold",
        "LimitedLevelWidth": "Limited Width"
    }
    
    # Map the names in the dataframe for legend clarity
    df_plot['type_legend'] = df_plot['type'].map(type_name_map).fillna(df_plot['type'])
    df_plot['search_strategy_legend'] = df_plot['search_strategy'].map(search_strategy_name_map).fillna(df_plot['search_strategy'])

    # Plotting
    plt.figure(figsize=(18, 7))
    ax = plt.gca()
    sns.set_theme(style="whitegrid", palette="colorblind")

    # Add line plot to connect points with thicker lines
    sns.lineplot(
        data=df_plot,
        x=x_axis_col,
        y=y_axis_col,
        hue='type_legend',
        style='search_strategy_legend',
        legend=False,  # Legend will be created by scatterplot
        alpha=0.25,
        linewidth=3,
        ax=ax
    )

    # Add scatter plot
    scatter = sns.scatterplot(
        data=df_plot,
        x=x_axis_col,
        y=y_axis_col,
        hue='type_legend',
        style='search_strategy_legend',
        s=150,
        edgecolor='none',
        alpha=0.8,
        ax=ax
    )

    # Text Labels
    # Function to create parameter labels (e.g., "th=0.5", "n=10", "w=200")
    def get_param_label(row):
        if pd.notna(row['threshold']):
            return f"th={row['threshold']}"
        elif pd.notna(row['top_n']):
            return f"n={int(row['top_n'])}"
        elif pd.notna(row['max_width']):
            return f"w={int(row['max_width'])}"
        else:
            return ""

    if show_labels:
        # Add text labels to each point with a basic overlap avoidance strategy
        for i, row in df_plot.iterrows():
            label = get_param_label(row)
            
            # Find entries with similar x and y values
            nearby_y = df_plot[(df_plot[y_axis_col] > row[y_axis_col] * 0.8) & (df_plot[y_axis_col] < row[y_axis_col] * 1.2)]
            nearby_x = df_plot[(df_plot[x_axis_col] > row[x_axis_col] * 0.25) & (df_plot[x_axis_col] < row[x_axis_col] * 3.5)]

            nearby_y_bottom = nearby_x[(nearby_x[y_axis_col] > row[y_axis_col] * 0.8) & (nearby_x[y_axis_col] <= row[y_axis_col])]
            nearby_y_top = nearby_x[(nearby_x[y_axis_col] >= row[y_axis_col]) & (nearby_x[y_axis_col] < row[y_axis_col] * 1.2)]

            nearby_x_left = nearby_y[(nearby_y[x_axis_col] > row[x_axis_col] * 0.35) & (nearby_y[x_axis_col] <= row[x_axis_col])]
            nearby_x_right = nearby_y[(nearby_y[x_axis_col] >= row[x_axis_col]) & (nearby_y[x_axis_col] < row[x_axis_col] * 2)]

            vertical_alignment = 'center'
            horizontal_alignment = 'left'
            offsetx_factor = 1.1

            if (len(nearby_y_bottom) + len(nearby_y_top)) > 1 and (len(nearby_x_right) + len(nearby_x_left)) > 1:
                if len(nearby_x_left) == 1:
                    horizontal_alignment = 'right'
                    offsetx_factor = 0.9
                else:
                    if len(nearby_y_bottom) > len(nearby_y_top):
                        vertical_alignment = 'bottom'
                    elif len(nearby_y_top) > len(nearby_y_bottom):
                        vertical_alignment = 'top'
                    elif len(nearby_x_right) % 2:
                        vertical_alignment = 'bottom'
                    else:
                        vertical_alignment = 'top'

            plt.text(
                x=row[x_axis_col] * offsetx_factor,  # Horizontal offset from point
                y=row[y_axis_col],                  # Vertical offset from point
                s=label,
                fontdict={'size': 12, 'color': 'black', 'alpha': 0.7},
                ha=horizontal_alignment,
                va=vertical_alignment
            )

    # Legend Formatting
    handles, labels = ax.get_legend_handles_labels()

    search_strategy_handles = []
    search_strategy_labels = []
    type_handles = []
    type_labels = []

    if 'search_strategy_legend' in labels and 'type_legend' in labels:
        style_title_index = labels.index('search_strategy_legend')
        hue_title_index = labels.index('type_legend')

        for i in range(hue_title_index + 1, style_title_index):
            type_handles.append(handles[i])
            type_labels.append(labels[i])

        for i in range(style_title_index + 1, len(labels)):
            search_strategy_handles.append(handles[i])
            search_strategy_labels.append(labels[i])

        ax.get_legend().remove()

        legend1 = ax.legend(
            search_strategy_handles, search_strategy_labels,
            title="Search Strategy", title_fontsize=16, fontsize=13, loc='upper left',
            bbox_to_anchor=(1.07, 0.55), borderaxespad=0., frameon=True
        )
        plt.setp(legend1.get_title(), fontweight='bold')
        ax.add_artist(legend1)

        legend2 = ax.legend(
            type_handles, type_labels,
            title="Type", title_fontsize=16, fontsize=13, loc='upper left',
            bbox_to_anchor=(1.025, 0.9), borderaxespad=0., frameon=True
        )
        plt.setp(legend2.get_title(), fontweight='bold')

    # Axis and Title Formatting
    # Define map for x-axis labels
    x_label_map = {
        'execution_time_s': 'Execution Time (s) [log scale]',
        'peak_process_memory_mb': 'Peak Process Memory (MB) [log scale]'
    }
    y_label_map = {
        'relevant_paths_found': 'Number of Relevant Paths [log scale]',
        'paths_found': 'Number of Paths Found [log scale]'
    }

    x_label = x_label_map.get(x_axis_col, f"{x_axis_col} [log scale]")
    y_label = y_label_map.get(y_axis_col, f"{y_axis_col} [log scale]")
    title_x = x_label.split('[')[0].strip()
    title_y = y_label.split('[')[0].strip()

    # Set logarithmic scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Set dynamic axis limits
    if not df_plot.empty:
        ax.set_ylim(df_plot[y_axis_col].min() * 0.8, df_plot[y_axis_col].max() * 1.5)
        ax.set_xlim(df_plot[x_axis_col].min() * 0.8, df_plot[x_axis_col].max() * 1.5)
    else:
        ax.set_ylim(1, 1000)
        ax.set_xlim(1, 1000)


    # Add informative and well-styled labels and title
    ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
    ax.set_title(f'{title_y} vs {title_x} by Search Strategy', fontsize=18, fontweight='bold')

    # Add a grid for easier reading of values
    ax.grid(True, which='both', linestyle='--', alpha=0.6)

    # Adjust the plot layout to make space for the external legend
    plt.tight_layout(rect=[0, 0, 0.8, 1])

    # # Save the final plot to a high-resolution file
    if save_plot:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()
