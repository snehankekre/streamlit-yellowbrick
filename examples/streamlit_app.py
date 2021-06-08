import streamlit as st
from streamlit_yellowbrick import st_yellowbrick

from yellowbrick.datasets import load_concrete
from yellowbrick.features import JointPlotVisualizer

X, y = load_concrete()

visualizer = JointPlotVisualizer(columns="cement")
visualizer.fit_transform(X, y)

st_yellowbrick(visualizer)