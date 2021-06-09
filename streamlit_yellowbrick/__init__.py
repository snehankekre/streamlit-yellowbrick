import streamlit as st
import streamlit.components.v1 as components

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def st_yellowbrick(visualizer, scrolling=False):
    """Embed a Yellowbrick visualizer within a Streamlit app
    Parameters
    ----------
    visualizer: 
        Yellowbrick visualizer that has been fit to data.
    scrolling: bool
        If True, show a scrollbar when the content is larger than the iframe. 
        Otherwise, do not show a scrollbar. Defaults to False.
    
    Example
    -------
    >>> import streamlit as st
    >>> from streamlit_yellowbrick import st_yellowbrick

    >>> from yellowbrick.datasets import load_credit
    >>> from yellowbrick.features import PCA

    >>> X, y = load_credit()
    >>> classes = ['account in default', 'current with bills']

    >>> visualizer = PCA(scale=True, classes=classes)
    >>> visualizer.fit_transform(X, y)  # Fit the data to the visualizer

    >>> st_yellowbrick(visualizer) # Use in place of visualizer.show()
    """

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

    plt.cla()
    plt.close(fig)

    return components.html(
        html_str,
        height=height,
        width=width,
        scrolling=scrolling,
    )
