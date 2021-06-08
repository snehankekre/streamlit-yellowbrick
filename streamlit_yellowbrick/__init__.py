import streamlit as st
import streamlit.components.v1 as components

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def st_yellowbrick(visualizer, scrolling=False):

    if isinstance(visualizer.fig, Figure):
        height = visualizer.size[1]
        width = visualizer.size[0]
    else:
        height = 500
        width = 500

    fig = visualizer.fig
    ax = visualizer.show()
    fig.axes.append(ax)

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    html_str = f"<img src='data:image/png;base64,{data}'/>"

    return components.html(
        html_str,
        height=height,
        width=width,
        scrolling=scrolling,
    )
