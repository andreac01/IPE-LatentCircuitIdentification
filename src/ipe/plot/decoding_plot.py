import plotly.graph_objects as go
import ipywidgets as widgets
from ipywidgets import interactive_output, VBox, HBox, Output
from IPython.display import display
import torch
import torch.nn.functional as F
import numpy as np
from plotly.subplots import make_subplots
from ipe.paths import get_path_msgs
from ipe.miscellanea import get_topk


# Helper function from the user's request
def plot_probability_distribution_plotly(logits, title, model, top_n=10):
    """Plots the probability distribution for the top N tokens using Plotly."""
    probs = F.softmax(logits.to(torch.float32), dim=-1).detach().cpu().numpy()
    idxs = np.argsort(probs)[-top_n:][::-1]
    probs_top = probs[idxs]
    tokens_top = [model.to_string([int(i)]) for i in idxs]

    fig = make_subplots(rows=top_n, cols=1, shared_xaxes=True)
    
    for i in range(top_n):
        fig.add_trace(go.Bar(
            y=[tokens_top[i]],
            x=[probs_top[i]],
            orientation='h',
            marker_color='#52b788',
            hovertemplate="<b>%{y}</b>: %{x:.2f}<extra></extra>",
            showlegend=False,
        ), row=i+1, col=1)
        fig.update_xaxes(range=[0, 1], row=i+1, col=1)
    
    fig.update_layout(
        title=title,
        xaxis_title="Probability",
        yaxis_title="Token",
        xaxis=dict(range=[0, 1]),
        height=20 * top_n + 50,  # Adjust height based on top_n
        margin=dict(l=100, r=20, t=40, b=20),
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12,
        template="plotly_white",
        showlegend=False,
    )
    return fig


def create_interactive_decoding_plot(logits, model):
    """
    Creates an interactive plot to visualize the top N token predictions.
    
    Args:
        logits (torch.Tensor): 
            A tensor of shape (batch_size, d_model) or (d_model,) representing the logits.
        model (HookedTransformer): 
            The model object with to_string() and other necessary methods.
    """
    batch_size = logits.shape[0] if len(logits.shape) > 1 else 1
    style = {'description_width': '120px'}
    layout = widgets.Layout(width='600px', margin='10px 20px')

    widgets_list = []
    if batch_size > 1:
        batch_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=batch_size - 1,
            step=1,
            description="Batch Index:",
            style=style,
            layout=layout,
            continuous_update=False
        )
        widgets_list.append(batch_slider)
    
    top_n_slider = widgets.IntSlider(
        value=10,
        min=1,
        max=min(25, logits.shape[-1]),  # Limit max to a reasonable number
        step=1,
        description="Top N Tokens:",
        style=style,
        layout=layout,
        continuous_update=False
    )
    widgets_list.append(top_n_slider)
    
    plot_output = Output()
    print(logits.shape)
    def update_plot(batch_index=0, top_n=10):
        if batch_size == 1:
            logits_selected = logits[0]
        else:
            logits_selected = logits[batch_index]
        
        with plot_output:
            plot_output.clear_output(wait=True)
            fig = plot_probability_distribution_plotly(
                logits_selected,
                f"Top {top_n} Predicted Tokens (Batch {batch_index})",
                model,
                top_n=top_n
            )
            display(fig)

    if batch_size > 1:
        interactive_output(update_plot, {'batch_index': batch_slider, 'top_n': top_n_slider})
    else:
        interactive_output(update_plot, {'top_n': top_n_slider})
    
    display(VBox(widgets_list + [plot_output], layout=widgets.Layout(margin="20px")))

# Helper function to generate a readable label for a node
def get_node_label(node):
    """Creates a descriptive string label for a path node."""
    h = f" H{node.head}" if hasattr(node, 'head') and node.head is not None else ""
    return f"{node.__class__.__name__.replace('_Node', '')} L{node.layer}{h}  P{node.position}"

def create_interactive_path_visualization(paths, model, cache):
    """
    Creates an interactive Plotly and IPyWidgets visualization for model paths,
    optimized for Google Colab using interactive_output.

    Args:
        paths (list): A list where each element is a path (a list of node objects).
        model: A HookedTransformer model instance.
        cache: The activation cache from a model run.
    """
    batch_size = cache['hook_embed'].shape[0]
    fixed_path_plot_height = 400  # Fixed height for the path plot

    # --- 1. WIDGETS ---
    path_selector = widgets.Dropdown(
        options=[(f"Path {i}", i) for i in range(len(paths))],
        value=0,
        description='Select Path:',
        style={'description_width': 'initial'}
    )
    
    batch_selector = widgets.IntSlider(
        value=0, min=0, max=max(0, batch_size - 1), step=1,
        description='Select Sample:',
        style={'description_width': 'initial'},
        continuous_update=False
    )
    node_slider = widgets.IntSlider(
        value=0, min=0, max=max(0, len(paths[0]) - 1), step=1,
        orientation='vertical', description='Node', readout=False,
        layout=widgets.Layout(height=f'{int(fixed_path_plot_height * 0.8)}px', width='auto'),
        continuous_update=False
    )

    # --- 2. OUTPUT WIDGETS ---
    path_output = Output()
    token_output = Output()

    # --- 3. UPDATE LOGIC ---
    def update_visualization(path_idx, batch_idx, node_idx):
        """
        Draws both the path plot and the top-k token plot.
        This function is executed when any linked widget changes.
        """
        path_data = paths[path_idx]
        path_len = len(path_data)
        
        # --- PATH FIGURE (LEFT) ---
        path_fig = go.Figure()
        
        # Distribute nodes evenly over the fixed height
        y_coords = np.linspace(0, fixed_path_plot_height, path_len) if path_len > 1 else [fixed_path_plot_height / 2]
        
        # Add line segments
        for i in range(path_len - 1):
            node_to = path_data[i+1]
            is_dashed = hasattr(node_to, 'patch_query') and node_to.patch_query
            path_fig.add_trace(go.Scatter(
                x=[0, 0], y=[y_coords[i], y_coords[i+1]], mode='lines',
                line=dict(color='lightgrey', width=4, dash='dash' if is_dashed else 'solid'),
                hoverinfo='none'
            ))

        # Add markers
        node_labels = [get_node_label(node) for node in path_data]
        colors = ['#adb5bd'] * path_len
        if node_idx < path_len:
            colors[node_idx] = '#0d6efd'
        
        path_fig.add_trace(go.Scatter(
            x=[0] * path_len, y=y_coords,
            mode='markers+text',
            marker=dict(
                size=[24 if i == node_idx else 16 for i in range(path_len)],
                color=colors,
                symbol='circle'
            ),
            text=node_labels,
            textfont=dict(size=10),
            textposition='middle right',
            hovertext=node_labels,
            hoverinfo='text'
        ))

        path_fig.update_layout(
            title=f"Path {path_idx} Components",
            template="plotly_white",
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            margin=dict(l=40, r=40, t=40, b=40),
            height=fixed_path_plot_height,
            showlegend=False
        )

        # --- TOKEN FIGURE (RIGHT) ---
        current_path_messages = get_path_msgs(path_data, messages=[], model=model, msg_cache=cache)
        
        selected_node = path_data[node_idx]
        residual = current_path_messages[node_idx][batch_idx, path_data[node_idx].position, :]
        topk_dict = get_topk(model, residual, topk=10)
        probs = topk_dict['topk_probs']
        tokens = topk_dict['topk_strtokens']

        token_fig = go.Figure()
        token_fig.add_trace(go.Bar(
            y=tokens, x=probs, orientation='h', marker_color='#52b788',
            hovertemplate="<b>%{y}</b>: %{x:.2f}<extra></extra>",
            showlegend=False,
        ))
        
        token_fig.update_layout(
            title=f"Top Tokens at {get_node_label(selected_node)} (Sample {batch_idx})",
            template="plotly_white", xaxis_title="Probability",
            yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, 1]),
            margin=dict(l=100, r=20, t=40, b=40), bargap=0.1,
            height=400
        )
        
        # --- DISPLAY OUTPUTS ---
        with path_output:
            path_output.clear_output(wait=True)
            display(path_fig)
        
        with token_output:
            token_output.clear_output(wait=True)
            display(token_fig)

    # --- 4. INTERACTIVE LINKING ---
    # Link widgets to the main update function
    out = interactive_output(
        update_visualization, 
        {'path_idx': path_selector, 'batch_idx': batch_selector, 'node_idx': node_slider}
    )
    
    # Create a separate observer to update the node slider's range when the path changes
    def on_path_change(change):
        new_path_idx = change['new']
        new_path_len = len(paths[new_path_idx])
        node_slider.max = max(0, new_path_len - 1)
        # Reset slider to the top to avoid index out of bounds errors
        node_slider.value = 0

    path_selector.observe(on_path_change, names='value')
    
    # --- 5. LAYOUT & DISPLAY ---
    controls = HBox([path_selector, batch_selector])
    path_viz_with_slider = HBox([path_output, node_slider], layout=widgets.Layout(align_items='center'))
    main_layout = HBox([path_viz_with_slider, token_output])
    
    display(VBox([controls, main_layout]))

