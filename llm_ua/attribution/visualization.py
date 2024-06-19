from typing import Dict, List

import numpy as np
import plotly.express as px
from tabulate import tabulate

from .post_processing import normalize_attr_map


def visualize(
        attribution_dict: Dict[str, np.ndarray],
        input_prompt: str,
        decoded_input_tokens: List[str],
        uncertainty: float,
        logits: np.ndarray,
        variance: np.ndarray,
        gt_class: int,
        save_path: str,
        absolute: bool) -> None:
    assert variance.ndim == 2, f'variance should be a 2-D ndarray. But got {variance.ndim}-D.'
    layer_attr_maps: List[np.ndarray] = []
    layer_names: List[str] = []
    for layer_name, attr_map in attribution_dict.items():
        layer_names.append(layer_name)
        # Note: attribution map is normalized per-layer. I.e., using the min-max values within each layer.
        layer_attr_maps.append(normalize_attr_map(attr_map, absolute=absolute, min_max=True, to_image=False))

    # shape: (num_layers, num_tokens). dtype: float64. value range: [0, 1]
    stacked_attr_map = np.stack(layer_attr_maps)

    num_layers, num_tokens = stacked_attr_map.shape
    assert len(decoded_input_tokens) == num_tokens

    x_ticks = np.arange(num_tokens)
    y_ticks = np.arange(num_layers)
    fig = px.imshow(stacked_attr_map, x=x_ticks, y=y_ticks, color_continuous_scale='Viridis', aspect='auto')
    fig.update_traces(
        text=[decoded_input_tokens.copy() for _ in range(num_layers)],
        texttemplate='%{text}',
        hovertemplate='%{x} - [%{text}] - attr: %{customdata:.3f}',
        customdata=stacked_attr_map)
    fig.update_xaxes(side='top', title_text='Tokens')
    fig.update_yaxes(tickmode='array', tickvals=y_ticks, ticktext=layer_names)

    showed_text = f'<b>[Input]</b>:\n{input_prompt}\n'
    showed_text += f"<b>[GT Class]</b>: {chr(ord('A')+ gt_class)}\n"
    showed_text += f'<b>[Uncertainty Score]</b>: {uncertainty:.4f}\n'
    showed_text += f'<b>[Logits]</b>: [{", ".join(str(x) for x in logits.round(4))}]\n'
    showed_text += f"<b>[Covariance]</b>:\n{tabulate(variance.round(2), tablefmt='grid', numalign='right')}"

    # input_prompt also contains '\n'. However, plotly add_annotation does not render '\n'. So use html '<br>'.
    showed_text = showed_text.replace('\n', '<br>')

    # Set the layout to have more space at the bottom
    fig.update_layout(margin=dict(t=50, b=450))

    fig.add_annotation(
        x=0.01,
        y=-0.01,
        xref='paper',
        yref='paper',
        text=showed_text,
        showarrow=False,
        font=dict(size=12, color='black'),
        align='left',
        xanchor='left',
        yanchor='top')
    fig.write_html(save_path)
