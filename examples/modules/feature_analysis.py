import streamlit as st
from streamlit_yellowbrick import st_yellowbrick


def run_feature_analysis():

    with st.sidebar.form(key="feat_analysis_form"):
        feature_analysis_visualizers = st.multiselect(
            "Choose Feature Analysis Visualizers",
            [
                "RadViz Visualizer",
                "Parallel Coordinates",
                "Joint Plot Visualization",
                "PCA",
            ],
        )
        submit_button = st.form_submit_button(label="Show")

    if "RadViz Visualizer" in feature_analysis_visualizers:

        with st.beta_expander("Collapse", expanded=True):

            agree = st.checkbox("What is RadViz?", value=False)

            if agree:
                st.markdown(
                    """
                `RadViz` is a multivariate data visualization algorithm that plots 
                each feature dimension uniformly around the circumference of a circle 
                then plots points on the interior of the circle such that the point 
                normalizes its values on the axes from the center to each arc. This 
                mechanism allows as many dimensions as will easily fit on a circle, 
                greatly expanding the dimensionality of the visualization.
                """
                )
            col1, col2 = st.beta_columns(2)

            with col1:
                radviz()

            col2.code(
"""
import streamlit as st
from streamlit_yellowbrick import st_yellowbrick

from yellowbrick.datasets import load_occupancy
from yellowbrick.features import RadViz

# Load the classification dataset
X, y = load_occupancy()

# Specify the target classes
classes = ["unoccupied", "occupied"]

# Instantiate the visualizer
visualizer = RadViz(classes=classes)

visualizer.fit(X, y)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data

st_yellowbrick(visualizer)     # Finalize and render the figure
"""
            , language="python")

    if "Parallel Coordinates" in feature_analysis_visualizers:

        with st.beta_expander("Collapse", expanded=True):

            agree = st.checkbox("What is Parallel Coordinates?", value=False)

            if agree:
                st.markdown(
                    """
                    Parallel coordinates is multi-dimensional feature visualization technique
                    where the vertical axis is duplicated horizontally for each feature. 
                    Instances are displayed as a single line segment drawn from each vertical
                    axes to the location representing their value for that feature. This
                    allows many dimensions to be visualized at once; in fact given infinite
                    horizontal space (e.g. a scrolling window), technically an infinite number
                    of dimensions can be displayed!
                    """
                )
            col1, col2 = st.beta_columns(2)

            with col1:
                parallel_coordinates()

            col2.code(
"""
import streamlit as st
from streamlit_yellowbrick import st_yellowbrick

from yellowbrick.features import ParallelCoordinates
from yellowbrick.datasets import load_occupancy

# Load the classification data set
X, y = load_occupancy()

# Specify the features of interest and the classes of the target
features = [
    "temperature", "relative humidity", "light", "CO2", "humidity"
]
classes = ["unoccupied", "occupied"]

# Instantiate the visualizer
visualizer = ParallelCoordinates(
    classes=classes, features=features, sample=0.05, shuffle=True
)

# Fit and transform the data to the visualizer
visualizer.fit_transform(X, y)

# Finalize the title and axes then display the visualization
st_yellowbrick(visualizer)
"""
            , language="python")

    if "Joint Plot Visualization" in feature_analysis_visualizers:

        with st.beta_expander("Collapse", expanded=True):

            agree = st.checkbox("What is a Joint Plot?", value=False)

            if agree:
                st.markdown(
                    """
                    The `JointPlotVisualizer` plots a feature against the target and shows the
                    distribution of each via a histogram on each axis.
                    """
                )
            col1, col2 = st.beta_columns(2)

            with col1:
                joint_plot()

            col2.code(
"""
import streamlit as st
from streamlit_yellowbrick import st_yellowbrick

from yellowbrick.datasets import load_concrete
from yellowbrick.features import JointPlotVisualizer

# Load the dataset
X, y = load_concrete()

# Instantiate the visualizer
visualizer = JointPlotVisualizer(columns="cement")

visualizer.fit_transform(X, y)        # Fit and transform the data
st_yellowbrick(visualizer)            # Finalize and render the figure
"""
            , language="python")

    if "PCA" in feature_analysis_visualizers:

        with st.beta_expander("Collapse", expanded=True):

            agree = st.checkbox("What is PCA?", value=False)

            if agree:
                st.markdown(
                    """
                    The PCA Decomposition visualizer utilizes principal component analysis
                    to decompose high dimensional data into two or three dimensions so that
                    each instance can be plotted in a scatter plot. The use of PCA means that
                    the projected dataset can be analyzed along axes of principal variation
                    and can be interpreted to determine if spherical distance metrics can be utilized.

                    The PCA projection can be enhanced to a biplot whose points are the projected
                    instances and whose vectors represent the structure of the data in high
                    dimensional space. By using `proj_features=True`, vectors for each feature
                    in the dataset are drawn on the scatter plot in the direction of the maximum
                    variance for that feature. These structures can be used to analyze the
                    importance of a feature to the decomposition or to find features of related
                    variance for further analysis.
                    """
                )
            col1, col2 = st.beta_columns(2)

            with col1:
                PCA()

            col2.code(
"""
import streamlit as st
from streamlit_yellowbrick import st_yellowbrick

from yellowbrick.datasets import load_concrete
from yellowbrick.features import PCA

X, y = load_concrete()

visualizer = PCA(scale=True, proj_features=True, projection=3)
visualizer.fit_transform(X, y)
st_yellowbrick(visualizer)
"""
            , language="python")

    return None


def radviz():
    from yellowbrick.datasets import load_occupancy
    from yellowbrick.features import RadViz

    # Load the classification dataset
    X, y = load_occupancy()

    # Specify the target classes
    classes = ["unoccupied", "occupied"]

    # Instantiate the visualizer
    radviz_visualizer = RadViz(classes=classes)

    radviz_visualizer.fit(X, y)  # Fit the data to the visualizer
    radviz_visualizer.transform(X)  # Transform the data

    return st_yellowbrick(radviz_visualizer)  # Finalize and render the figure


def parallel_coordinates():
    from yellowbrick.features import ParallelCoordinates
    from yellowbrick.datasets import load_occupancy

    # Load the classification data set
    X, y = load_occupancy()

    # Specify the features of interest and the classes of the target
    features = ["temperature", "relative humidity", "light", "CO2", "humidity"]
    classes = ["unoccupied", "occupied"]

    # Instantiate the visualizer
    visualizer = ParallelCoordinates(
        classes=classes, features=features, sample=0.05, shuffle=True
    )

    # Fit and transform the data to the visualizer
    visualizer.fit_transform(X, y)

    # Finalize the title and axes then display the visualization
    return st_yellowbrick(visualizer)


def joint_plot():
    from yellowbrick.datasets import load_concrete
    from yellowbrick.features import JointPlotVisualizer

    # Load the dataset
    X, y = load_concrete()

    # Instantiate the visualizer
    visualizer = JointPlotVisualizer(columns="cement")

    visualizer.fit_transform(X, y)  # Fit and transform the data
    return st_yellowbrick(visualizer)  # Finalize and render the figure


def PCA():
    from yellowbrick.datasets import load_concrete
    from yellowbrick.features import PCA

    X, y = load_concrete()

    visualizer = PCA(scale=True, proj_features=True, projection=3)
    visualizer.fit_transform(X, y)
    return st_yellowbrick(visualizer)
