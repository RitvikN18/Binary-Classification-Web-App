import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score


def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Only Binary Classification datasets can be trained 🍄")
    st.sidebar.markdown("Binary Classification 🍄")

    def NoDataError():
        st.write("Please upload a dataset with valid contents")

    # @st.cache(persist=True)
    def load_data():
        # pd.read_csv('./mushrooms.csv')
        data = st.file_uploader("Upload Dataset", type=["csv"])
        if data is not None:
            df = pd.read_csv(data)
            label = LabelEncoder()
            for col in df.columns:
                df[col] = label.fit_transform(df[col].astype(str))
        else:
            NoDataError()
            return None
        return df

    @st.cache(persist=True)
    def split(df, label_column):
        y = df[label_column]
        x = df.drop(columns=[label_column])
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test,
                                  display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Cuve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

    label_column = st.text_input(
        "Enter the name of the label column in your dataset", "type")
    df = load_data()
    if df is not None:
        x_train, x_test, y_train, y_test = split(df, label_column)
        class_names = st.text_input(
            "Enter two class names with space in between them", "edible poisonous").split()
        st.sidebar.subheader("Choose Classifier")
        classifier = st.sidebar.selectbox(
            "Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

        if classifier == 'Support Vector Machine (SVM)':
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input(
                "C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
            kernel = st.sidebar.radio(
                "Kernel", ("rbf", "linear"), key='kernel')
            gamma = st.sidebar.radio(
                "Gamma (Kernal Coefficient", ("scale", "auto"), key="gamma")

            metrics = st.sidebar.multiselect(
                "What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Support Vector Machine (SVM) Results")
                model = SVC(C=C, kernel=kernel, gamma=gamma)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(
                    y_test, y_pred, labels=class_names).round(2))
                st.write("Recall: ", recall_score(
                    y_test, y_pred, labels=class_names).round(2))
                plot_metrics(metrics)

        if classifier == 'Logistic Regression':
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input(
                "C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
            max_iter = st.sidebar.slider(
                "Maximum number of iterations", 100, 500, key='max_iter')

            metrics = st.sidebar.multiselect(
                "What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Logistic Regression Results")
                model = LogisticRegression(C=C, max_iter=max_iter)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(
                    y_test, y_pred, labels=class_names).round(2))
                st.write("Recall: ", recall_score(
                    y_test, y_pred, labels=class_names).round(2))
                plot_metrics(metrics)

        if classifier == 'Random Forest':
            st.sidebar.subheader("Model Hyperparameters")
            n_estimators = st.sidebar.number_input(
                "The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
            max_depth = st.sidebar.number_input(
                "Max depth of the tree", 1, 20, step=1, key='max_depth')
            bootstrap = st.sidebar.radio(
                "Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')

            metrics = st.sidebar.multiselect(
                "What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Random Forest Results")
                model = RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(
                    y_test, y_pred, labels=class_names).round(2))
                st.write("Recall: ", recall_score(
                    y_test, y_pred, labels=class_names).round(2))
                plot_metrics(metrics)

        if st.sidebar.checkbox("Show raw data", False):
            st.subheader("Dataset")
            st.write(df)

    st.write("By Ritvik Nimmagadda©")


if __name__ == '__main__':
    main()
