import streamlit as st
from streamlit_yellowbrick import st_yellowbrick


def run_classification():

    with st.sidebar.form(key="classification_form"):
        classification_visualizers = st.multiselect(
            "Choose Classification Visualizers",
            [
                "Classification Report",
                "ROCAUC",
                "Multi-class ROCAUC Curves",
                "Class Prediction Error",
                "Confusion Matrix",
                "Discrimination Threshold",
            ],
        )
        submit_button = st.form_submit_button(label="Show")

    if "Classification Report" in classification_visualizers:

        with st.beta_expander("Collapse", expanded=True):

            agree = st.checkbox("What is a Classification Report?", value=False)

            if agree:
                st.markdown(
                    """
                    The classification report visualizer displays the precision, recall, F1,
                    and support scores for the model. In order to support easier interpretation
                    and problem detection, the report integrates numerical scores with a
                    color-coded heatmap. All heatmaps are in the range `(0.0, 1.0)` to facilitate
                    easy comparison of classification models across different classification reports.
                    """
                )
            col1, col2 = st.beta_columns(2)

            with col1:
                classification_report()

            with col2:

                st.code(
                    """
                    import streamlit as st
                    from streamlit_yellowbrick import st_yellowbrick

                    from sklearn.model_selection import TimeSeriesSplit
                    from sklearn.naive_bayes import GaussianNB

                    from yellowbrick.classifier import ClassificationReport
                    from yellowbrick.datasets import load_occupancy

                    # Load the classification dataset
                    X, y = load_occupancy()

                    # Specify the target classes
                    classes = ["unoccupied", "occupied"]

                    # Create the training and test data
                    tscv = TimeSeriesSplit()
                    for train_index, test_index in tscv.split(X):
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    # Instantiate the classification model and visualizer
                    model = GaussianNB()
                    visualizer = ClassificationReport(model, classes=classes, support=True)

                    visualizer.fit(X_train, y_train)        # Fit the visualizer and the model
                    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
                    st_yellowbrick(visualizer)              # Finalize and show the figure
                    """
                )

    if "ROCAUC" in classification_visualizers:

        with st.beta_expander("Collapse", expanded=True):

            agree = st.checkbox("What is ROCAUC?", value=False)

            if agree:
                st.markdown(
                    """
                   A `ROCAUC` (Receiver Operating Characteristic/Area Under the Curve) plot allows 
                   the user to visualize the tradeoff between the classifier’s sensitivity and 
                   specificity.
                    """
                )
            col1, col2 = st.beta_columns(2)

            with col1:
                rocauc()

            with col2:

                st.code(
                    """
                    import streamlit as st
                    from streamlit_yellowbrick import st_yellowbrick

                    from sklearn.linear_model import LogisticRegression
                    from sklearn.model_selection import train_test_split

                    from yellowbrick.classifier import ROCAUC
                    from yellowbrick.datasets import load_spam

                    # Load the classification dataset
                    X, y = load_spam()

                    # Create the training and test data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

                    # Instantiate the visualizer with the classification model
                    model = LogisticRegression(multi_class="auto", solver="liblinear")
                    visualizer = ROCAUC(model, classes=["not_spam", "is_spam"])

                    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
                    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
                    st_yellowbrick(visualizer)              # Finalize and show the figure
                    """
                )

    if "Multi-class ROCAUC Curves" in classification_visualizers:

        with st.beta_expander("Collapse", expanded=True):

            agree = st.checkbox("What are Multi-class ROCAUC Curves?", value=False)

            if agree:
                st.markdown(
                    """
                   Yellowbrick’s `ROCAUC` Visualizer does allow for plotting multiclass
                   classification curves. ROC curves are typically used in binary classification,
                   and in fact the Scikit-Learn `roc_curve` metric is only able to perform
                   metrics for binary classifiers. Yellowbrick addresses this by binarizing
                   the output (per-class) or to use one-vs-rest (micro score) or one-vs-all
                   (macro score) strategies of classification.
                    """
                )
            col1, col2 = st.beta_columns(2)

            with col1:
                multi_class_rocauc()

            with col2:

                st.code(
                    """
                    import streamlit as st
                    from streamlit_yellowbrick import st_yellowbrick

                    from sklearn.linear_model import RidgeClassifier
                    from sklearn.model_selection import train_test_split
                    from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

                    from yellowbrick.classifier import ROCAUC
                    from yellowbrick.datasets import load_game

                    # Load multi-class classification dataset
                    X, y = load_game()

                    # Encode the non-numeric columns
                    X = OrdinalEncoder().fit_transform(X)
                    y = LabelEncoder().fit_transform(y)

                    # Create the train and test data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

                    # Instaniate the classification model and visualizer
                    model = RidgeClassifier()
                    visualizer = ROCAUC(model, classes=["win", "loss", "draw"])

                    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
                    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
                    st_yellowbrick(visualizer)              # Finalize and render the figure
                    """
                )

    if "Class Prediction Error" in classification_visualizers:

        with st.beta_expander("Collapse", expanded=True):

            agree = st.checkbox("What is a Class Prediction Error?", value=False)

            if agree:
                st.markdown(
                    """
                   The Yellowbrick `ClassPredictionError` plot is a twist on other and sometimes more
                   familiar classification model diagnostic tools like the [Confusion Matrix](https://www.scikit-yb.org/en/latest/api/classifier/confusion_matrix.html)
                   and [Classification Report](https://www.scikit-yb.org/en/latest/api/classifier/classification_report.html).
                   Like the [Classification Report](https://www.scikit-yb.org/en/latest/api/classifier/classification_report.html),
                   this plot shows the support (number of training samples) for
                   each class in the fitted classification model as a stacked bar chart. Each bar is segmented
                   to show the proportion of predictions (including false negatives and false positives, like a
                   [Confusion Matrix](https://www.scikit-yb.org/en/latest/api/classifier/confusion_matrix.html))
                   for each class. You can use a `ClassPredictionError` to visualize which classes
                   your classifier is having a particularly difficult time with, and more importantly, what incorrect
                   answers it is giving on a per-class basis. This can often enable you to better understand strengths
                   and weaknesses of different models and particular challenges unique to your dataset.
                    """
                )
            col1, col2 = st.beta_columns(2)

            with col1:
                class_prediction_error()

            with col2:

                st.code(
                    """
                    import streamlit as st
                    from streamlit_yellowbrick import st_yellowbrick

                    from sklearn.datasets import make_classification
                    from sklearn.model_selection import train_test_split
                    from sklearn.ensemble import RandomForestClassifier
                    from yellowbrick.classifier import ClassPredictionError

                    # Create classification dataset
                    X, y = make_classification(
                        n_samples=1000,
                        n_classes=5,
                        n_informative=3,
                        n_clusters_per_class=1,
                        random_state=36,
                    )

                    classes = ["apple", "kiwi", "pear", "banana", "orange"]

                    # Perform 80/20 training/test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.20, random_state=42
                    )
                    # Instantiate the classification model and visualizer
                    visualizer = ClassPredictionError(
                        RandomForestClassifier(random_state=42, n_estimators=10), classes=classes
                    )

                    # Fit the training data to the visualizer
                    visualizer.fit(X_train, y_train)

                    # Evaluate the model on the test data
                    visualizer.score(X_test, y_test)

                    # Draw visualization
                    st_yellowbrick(visualizer)
                    """
                )

    if "Confusion Matrix" in classification_visualizers:

        with st.beta_expander("Collapse", expanded=True):

            agree = st.checkbox("What is a Confusion Matrix?", value=False)

            if agree:
                st.markdown(
                    """
                   The `ConfusionMatrix` visualizer is a `ScoreVisualizer` that takes a fitted scikit-learn classifier
                   and a set of test `X` and `y` values and returns a report showing how each of the test values
                   predicted classes compare to their actual classes. Data scientists use confusion matrices to
                   understand which classes are most easily confused. These provide similar information as what
                   is available in a `ClassificationReport`, but rather than top-level scores, they provide deeper
                   insight into the classification of individual data points.
                    """
                )
            col1, col2 = st.beta_columns(2)

            with col1:
                confusion_matrix()

            with col2:

                st.code(
                    """
                    import streamlit as st
                    from streamlit_yellowbrick import st_yellowbrick

                    from sklearn.datasets import load_iris
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.model_selection import train_test_split as tts

                    from yellowbrick.classifier import ConfusionMatrix

                    iris = load_iris()
                    X = iris.data
                    y = iris.target
                    classes = iris.target_names

                    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)

                    model = LogisticRegression(multi_class="auto", solver="liblinear")

                    iris_cm = ConfusionMatrix(
                        model,
                        classes=classes,
                        label_encoder={0: "setosa", 1: "versicolor", 2: "virginica"},
                    )

                    iris_cm.fit(X_train, y_train)
                    iris_cm.score(X_test, y_test)
                    st_yellowbrick(iris_cm)
                    """
                )

    if "Discrimination Threshold" in classification_visualizers:

        with st.beta_expander("Collapse", expanded=True):

            agree = st.checkbox("What is a Discrimination Threshold?", value=False)

            if agree:
                st.markdown(
                    """
                   A visualization of precision, recall, f1 score, and queue rate with respect to the
                   discrimination threshold of a binary classifier. The *discrimination threshold* is the
                   probability or score at which the positive class is chosen over the negative class.
                   Generally, this is set to 50% but the threshold can be adjusted to increase or decrease
                   the sensitivity to false positives or to other application factors.
                    """
                )
            col1, col2 = st.beta_columns(2)

            with col1:
                discrimination_threshold()

            with col2:

                st.code(
                    """
                    import streamlit as st
                    from streamlit_yellowbrick import st_yellowbrick

                    from sklearn.linear_model import LogisticRegression

                    from yellowbrick.classifier import DiscriminationThreshold
                    from yellowbrick.datasets import load_spam

                    # Load a binary classification dataset
                    X, y = load_spam()

                    # Instantiate the classification model and visualizer
                    model = LogisticRegression(multi_class="auto", solver="liblinear")
                    visualizer = DiscriminationThreshold(model)

                    visualizer.fit(X, y)        # Fit the data to the visualizer
                    st_yellowbrick(visualizer)  # Finalize and render the figure
                    """
                )

    return None


def classification_report():
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.naive_bayes import GaussianNB

    from yellowbrick.classifier import ClassificationReport
    from yellowbrick.datasets import load_occupancy

    # Load the classification dataset
    X, y = load_occupancy()

    # Specify the target classes
    classes = ["unoccupied", "occupied"]

    # Create the training and test data
    tscv = TimeSeriesSplit()
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Instantiate the classification model and visualizer
    model = GaussianNB()
    visualizer = ClassificationReport(model, classes=classes, support=True)

    visualizer.fit(X_train, y_train)  # Fit the visualizer and the model
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    return st_yellowbrick(visualizer)  # Finalize and show the figure


def rocauc():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from yellowbrick.classifier import ROCAUC
    from yellowbrick.datasets import load_spam

    # Load the classification dataset
    X, y = load_spam()

    # Create the training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Instantiate the visualizer with the classification model
    model = LogisticRegression(multi_class="auto", solver="liblinear")
    visualizer = ROCAUC(model, classes=["not_spam", "is_spam"])

    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    return st_yellowbrick(visualizer)  # Finalize and show the figure


def multi_class_rocauc():
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

    from yellowbrick.classifier import ROCAUC
    from yellowbrick.datasets import load_game

    # Load multi-class classification dataset
    X, y = load_game()

    # Encode the non-numeric columns
    X = OrdinalEncoder().fit_transform(X)
    y = LabelEncoder().fit_transform(y)

    # Create the train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Instaniate the classification model and visualizer
    model = RidgeClassifier()
    visualizer = ROCAUC(model, classes=["win", "loss", "draw"])

    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    return st_yellowbrick(visualizer)  # Finalize and render the figure


def class_prediction_error():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from yellowbrick.classifier import ClassPredictionError

    # Create classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_classes=5,
        n_informative=3,
        n_clusters_per_class=1,
        random_state=36,
    )

    classes = ["apple", "kiwi", "pear", "banana", "orange"]

    # Perform 80/20 training/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    # Instantiate the classification model and visualizer
    visualizer = ClassPredictionError(
        RandomForestClassifier(random_state=42, n_estimators=10), classes=classes
    )

    # Fit the training data to the visualizer
    visualizer.fit(X_train, y_train)

    # Evaluate the model on the test data
    visualizer.score(X_test, y_test)

    # Draw visualization
    return st_yellowbrick(visualizer)


def confusion_matrix():
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split as tts

    from yellowbrick.classifier import ConfusionMatrix

    iris = load_iris()
    X = iris.data
    y = iris.target
    classes = iris.target_names

    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)

    model = LogisticRegression(multi_class="auto", solver="liblinear")

    iris_cm = ConfusionMatrix(
        model,
        classes=classes,
        label_encoder={0: "setosa", 1: "versicolor", 2: "virginica"},
    )

    iris_cm.fit(X_train, y_train)
    iris_cm.score(X_test, y_test)
    return st_yellowbrick(iris_cm)


def discrimination_threshold():
    from sklearn.linear_model import LogisticRegression

    from yellowbrick.classifier import DiscriminationThreshold
    from yellowbrick.datasets import load_spam

    # Load a binary classification dataset
    X, y = load_spam()

    # Instantiate the classification model and visualizer
    model = LogisticRegression(multi_class="auto", solver="liblinear")
    visualizer = DiscriminationThreshold(model)

    visualizer.fit(X, y)  # Fit the data to the visualizer
    return st_yellowbrick(visualizer)  # Finalize and render the figure
