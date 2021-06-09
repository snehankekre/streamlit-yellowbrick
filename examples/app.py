import streamlit as st
import streamlit.components.v1 as components
from streamlit_yellowbrick import st_yellowbrick
from modules.feature_analysis import run_feature_analysis
from modules.regression import run_regression
from modules.classification import run_classification
from modules.clustering import run_clustering
from modules.model_selection import run_model_selection
from modules.text_modeling import run_text_modeling
from modules.target_visualization import run_target_visualization

st.set_page_config(
    page_title="Streamlit ❤️ Yellowbrick",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="https://media.tenor.com/images/f9a29e37ac3fe60465477478ab40e673/tenor.gif",
)

page = st.sidebar.selectbox("Navigation", ["About", "Live Demos"], index=0)

if page == "About":

    st.title("Streamlit Yellowbrick Demo")

    col1, col2 = st.beta_columns(2)

    with col1:
        st.markdown(
            """
            The [Yellowbrick](https://www.scikit-yb.org/en/latest/index.html) library is a diagnostic visualization platform
            for machine learning that allows data scientists to steer the model selection process. It extends the scikit-learn
            API with a new core object: the Visualizer. Visualizers allow visual models to be fit and transformed as part of
            the scikit-learn pipeline process, providing visual diagnostics throughout the transformation of high-dimensional
            data. Vist the Yellowbrick [documentation](https://www.scikit-yb.org/en/latest/about.html) and
            [GitHub repository](https://github.com/DistrictDataLabs/yellowbrick/) to learn more about this awesome open source,
            pure Python project!

            [streamlit-yellowbrick](https://github.com/snehankekre/streamlit-yellowbrick/) is a Streamlit component that provides
            a convenience function to display [Yellowbrick](https://www.scikit-yb.org/en/latest/index.html) visualizers in your
            Streamlit apps.

            Click on **Live Demos** from the **Navigation** dropdown on the left to view live demos of all Yellowbrick visualizers
            included in their Gallery on the right. Learn how to display visualizers in your Streamlit apps with 
            `streamlit-yellowbrick`!
            """
        )
        st.image(
            "https://github.com/snehankekre/streamlit-yellowbrick/raw/master/_static/navigation.png"
        )
    with col2:
        components.iframe(
            "https://www.scikit-yb.org/en/latest/gallery.html",
            height=800,
            scrolling=True,
        )

if page == "Live Demos":
    visualizers_list = [
        "Classification",
        "Feature Analysis",
        "Regression",
        # "Clustering",
        # "Model Selection",
        # "Text Modeling",
        # "Target Visualization",
    ]

    st.sidebar.header("Yellowbrick Visualizers Gallery")

    st.sidebar.markdown(
        "Visualizers are grouped according to the type of analysis they are well suited for."
    )
    visualizers_type = st.sidebar.radio(
        "What type of analysis do you want to explore?", visualizers_list
    )

    if visualizers_type == "Feature Analysis":
        run_feature_analysis()

    if visualizers_type == "Regression":
        run_regression()

    if visualizers_type == "Classification":
        run_classification()
