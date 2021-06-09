# streamlit-yellowbrick

This component provides a convenience function to display [Yellowbrick](https://www.scikit-yb.org/en/latest/index.html) [visualizers](https://www.scikit-yb.org/en/latest/api/index.html) in Streamlit.

## Installation

`pip install streamlit-yellowbrick`

## Example usage

```python
import streamlit as st
from streamlit_yellowbrick import st_yellowbrick

from yellowbrick.datasets import load_credit
from yellowbrick.features import PCA

# Specify the features of interest and the target
X, y = load_credit()
classes = ['account in default', 'current with bills']

visualizer = PCA(scale=True, classes=classes)
visualizer.fit_transform(X, y)  # Fit the data to the visualizer
st_yellowbrick(visualizer)      # Finalize and render the figure
```

![st_yellowbrick](https://github.com/snehankekre/streamlit-yellowbrick/blob/master/_static/example.png)

## Live demo

View a [live demo](https://share.streamlit.io/snehankekre/streamlit-yellowbrick/examples/app.py) on Streamlit sharing!

![st_example](https://i.imgur.com/OPZ5ulZ.mp4)
