import streamlit as st
from streamlit_yellowbrick import st_yellowbrick


def run_regression():

    with st.sidebar.form(key="regression_form"):
        regression_visualizers = st.multiselect(
            "Choose Regression Visualizers",
            [
                "Residuals Plot",
                "Prediction Error Plot",
                "Alpha Section",
            ],
        )
        submit_button = st.form_submit_button(label="Show")

    if "Residuals Plot" in regression_visualizers:

        with st.beta_expander("Collapse", expanded=True):

            agree = st.checkbox("What is a Residuals Plot?", value=False)

            if agree:
                st.markdown(
                    """
                    Residuals, in the context of regression models, are the difference between
                    the observed value of the target variable (y) and the predicted value (ŷ),
                    i.e. the error of the prediction. The residuals plot shows the difference
                    between residuals on the vertical axis and the dependent variable on the
                    horizontal axis, allowing you to detect regions within the target that may
                    be susceptible to more or less error.
                    """
                )
            col1, col2 = st.beta_columns(2)

            with col1:
                residuals_plot()

            col2.code(
                """
import streamlit as st
from streamlit_yellowbrick import st_yellowbrick

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import ResidualsPlot

# Load a regression dataset
X, y = load_concrete()

# Create the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the linear model and visualizer
model = Ridge()
visualizer = ResidualsPlot(model)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
st_yellowbrick(visualizer)        # Finalize and render the figure
""",
                language="python",
            )

    if "Prediction Error Plot" in regression_visualizers:

        with st.beta_expander("Collapse", expanded=True):

            agree = st.checkbox("What is a Prediction Error Plot?", value=False)

            if agree:
                st.markdown(
                    """
                    A prediction error plot shows the actual targets from the dataset against
                    the predicted values generated by our model. This allows us to see how
                    much variance is in the model. Data scientists can diagnose regression
                    models using this plot by comparing against the 45 degree line, where
                    the prediction exactly matches the model.
                    """
                )
            col1, col2 = st.beta_columns(2)

            with col1:
                prediction_error()

            col2.code(
                """
import streamlit as st
from streamlit_yellowbrick import st_yellowbrick

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import PredictionError

# Load a regression dataset
X, y = load_concrete()

# Create the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the linear model and visualizer
model = Lasso()
visualizer = PredictionError(model)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
st_yellowbrick(visualizer)        # Finalize and render the figure
""",
                language="python",
            )

    if "Alpha Section" in regression_visualizers:

        with st.beta_expander("Collapse", expanded=True):

            agree = st.checkbox("What is Alpha Section?", value=False)

            if agree:
                st.markdown(
                    """
                    Regularization is designed to penalize model complexity, therefore the higher
                    the alpha, the less complex the model, decreasing the error due to variance
                    (overfit). Alphas that are too high on the other hand increase the error due
                    to bias (underfit). It is important, therefore to choose an optimal alpha
                    such that the error is minimized in both directions.

                    The `AlphaSelection` Visualizer demonstrates how different values of alpha
                    influence model selection during the regularization of linear models.
                    Generally speaking, alpha increases the affect of regularization, e.g.
                    if alpha is zero there is no regularization and the higher the alpha,
                    the more the regularization parameter influences the final model.
                    """
                )
            col1, col2 = st.beta_columns(2)

            with col1:
                alpha_selection()

            col2.code(
                """
import numpy as np
from sklearn.linear_model import LassoCV
from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import AlphaSelection

# Load the regression dataset
X, y = load_concrete()

# Create a list of alphas to cross-validate against
alphas = np.logspace(-10, 1, 400)

# Instantiate the linear model and visualizer
model = LassoCV(alphas=alphas)
visualizer = AlphaSelection(model)
visualizer.fit(X, y)
st_yellowbrick(visualizer)
""",
                language="python",
            )

    return None


def residuals_plot():
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    from yellowbrick.datasets import load_concrete
    from yellowbrick.regressor import ResidualsPlot

    # Load a regression dataset
    X, y = load_concrete()

    # Create the train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Instantiate the linear model and visualizer
    model = Ridge()
    visualizer = ResidualsPlot(model)

    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    return st_yellowbrick(visualizer)  # Finalize and render the figure


def prediction_error():
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import train_test_split

    from yellowbrick.datasets import load_concrete
    from yellowbrick.regressor import PredictionError

    # Load a regression dataset
    X, y = load_concrete()

    # Create the train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Instantiate the linear model and visualizer
    model = Lasso()
    visualizer = PredictionError(model)

    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    return st_yellowbrick(visualizer)  # Finalize and render the figure


def alpha_selection():
    import numpy as np
    from sklearn.linear_model import LassoCV
    from yellowbrick.datasets import load_concrete
    from yellowbrick.regressor import AlphaSelection

    # Load the regression dataset
    X, y = load_concrete()

    # Create a list of alphas to cross-validate against
    alphas = np.logspace(-10, 1, 400)

    # Instantiate the linear model and visualizer
    model = LassoCV(alphas=alphas)
    visualizer = AlphaSelection(model)
    visualizer.fit(X, y)
    return st_yellowbrick(visualizer)
