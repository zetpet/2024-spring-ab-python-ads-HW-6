import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklift.models import SoloModel, TwoModels
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklift.viz import plot_uplift_curve


class DataAndModelHandler:
    def __init__(self, data_directory="./data"):
        self.data_dir = data_directory
        self.features_path = os.path.join(data_directory, "features.parquet")
        self.training_data_path = os.path.join(data_directory, "training_data.parquet")
        self.model = None
        self.X_train, self.y_train, self.treat_train = None, None, None
        self.X_val, self.y_val, self.treat_val = None, None, None

    @property
    def is_data_loaded(self):
        return os.path.exists(self.features_path) and os.path.exists(
            self.training_data_path
        )

    def load_data(self):
        if not self.is_data_loaded:
            st.error("Data files not found")
            return
        self.df_features = pd.read_parquet(self.features_path)
        self.df_train = pd.read_parquet(self.training_data_path)
        st.success("Data success loaded")

    def split_data(self, test_size=0.2):
        if self.df_features is None or self.df_train is None:
            st.error("Data not loaded")
            return
        X = self.df_features
        y = self.df_train["target"]
        treatment = self.df_train["treatment"]
        (
            self.X_train,
            self.X_val,
            self.y_train,
            self.y_val,
            self.treat_train,
            self.treat_val,
        ) = train_test_split(X, y, treatment, test_size=test_size, random_state=42)
        st.success(f"Data split into train and test sets with test size = {test_size}")

    def train_model(self, model_choice, classifier_choice, params):
        if model_choice == "Solo Model":
            model = SoloModel(estimator=classifier_choice(**params))
        else:
            model = TwoModels(
                estimator_trmnt=classifier_choice(**params),
                estimator_ctrl=classifier_choice(**params),
                method="vanilla",
            )
        model.fit(X=self.X_train, treatment=self.treat_train, y=self.y_train)
        self.model = model
        st.success("Model trained success")

    def evaluate_model(self):
        if self.model is None:
            st.error("No model is trained")
            return
        uplift_score = self.model.predict(self.X_val)
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_uplift_curve(
            y_true=self.y_val, uplift=uplift_score, treatment=self.treat_val, ax=ax
        )
        st.pyplot(fig)


def show_eda(handler):
    if not handler.is_data_loaded:
        st.warning("Data not loaded. Please load the data.")
        if st.button("Load Data"):
            handler.load_data()
    else:
        st.subheader("Data Overview")
        st.write("Features DataFrame:")
        st.write(handler.df_features.head())
        st.write("Training DataFrame:")
        st.write(handler.df_train.head())


def train_model_ui(handler):
    st.subheader("Train Uplift Model")
    test_size = st.slider(
        "Test Set Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05
    )
    if st.button("Split Data"):
        handler.split_data(test_size=test_size)

    model_choice = st.selectbox("Choose a model type:", ["Solo Model", "Two Models"])
    classifier_choice = st.selectbox(
        "Choose a classifier:", ["CatBoostClassifier", "RandomForestClassifier"]
    )

    if st.button("Train Model"):
        params = {
            "iterations": 100,
            "thread_count": 2,
            "random_state": 777,
            "silent": True,
        }
        if classifier_choice == "CatBoostClassifier":
            handler.train_model(model_choice, CatBoostClassifier, params)
        else:
            handler.train_model(model_choice, RandomForestClassifier, params)

    if st.button("Evaluate Model"):
        handler.evaluate_model()


def main():
    st.title("Uplift Modeling Dashboard")
    handler = DataAndModelHandler()

    app_mode = st.sidebar.selectbox("Choose Mode", ["EDA", "Train Model"])

    if app_mode == "EDA":
        show_eda(handler)
    elif app_mode == "Train Model":
        train_model_ui(handler)


if __name__ == "__main__":
    main()
