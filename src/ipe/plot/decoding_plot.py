import plotly.graph_objects as go
import ipywidgets as widgets
from ipywidgets import interactive_output, VBox, HBox, Output
from IPython.display import display
import torch
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer
from plotly.subplots import make_subplots


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
