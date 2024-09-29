import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

class ExploratoryDataAnalysis:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def run(self):
        st.sidebar.title("Data Exploration and EDA")
        st.sidebar.image("data1.png", use_column_width=True)

        explore_option = st.sidebar.selectbox(
            "Explore Model",
            ["View Data", "View Info", "View Description", "View Missing Values"]
        )

        eda_option = st.sidebar.selectbox(
            "Exploratory Data Analysis",
            ["Select Column", "Univariate Graphs", "Bivariate Graphs"]

        )
        st.sidebar.image("data2.png", use_column_width=True)

        if explore_option == "View Data":
            self.view_data()
        elif explore_option == "View Info":
            self.view_info()
        elif explore_option == "View Description":
            self.view_description()
        elif explore_option == "View Missing Values":
            self.view_missing_values()

        if eda_option == "Univariate Graphs":
            self.univariate_graphs()
        elif eda_option == "Bivariate Graphs":
            self.bivariate_graphs()
        elif eda_option == "Multivariate Graphs":
            self.multivariate_graphs()

    def view_data(self):
        st.title("Data Preview")
        st.dataframe(self.data.head())

    def view_info(self):
        buffer = io.StringIO()
        self.data.info(buf=buffer)
        info_str = buffer.getvalue()
        st.write("Data Information")
        st.text(info_str)

    def view_description(self):
        st.write("Data Description")
        st.dataframe(self.data.describe())

    def view_missing_values(self):
        st.write("Missing Values")
        st.dataframe(self.data.isnull().sum())

    def univariate_graphs(self):
        column = st.selectbox("Select Column for Univariate Analysis", self.data.columns)
        plot_type = st.selectbox("Select Plot Type", ["Histogram", "Boxplot", "Countplot"])

        if pd.api.types.is_numeric_dtype(self.data[column]):
            if plot_type == "Histogram":
                plt.figure(figsize=(10, 6))
                sns.histplot(self.data[column].dropna(), kde=True)
                st.pyplot(plt)
            elif plot_type == "Boxplot":
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=self.data[column].dropna())
                st.pyplot(plt)
        elif pd.api.types.is_categorical_dtype(self.data[column]) or self.data[column].dtype == 'object':
            if plot_type == "Countplot":
                plt.figure(figsize=(10, 6))
                sns.countplot(y=self.data[column].dropna())
                st.pyplot(plt)
        else:
            st.write(f"Cannot plot {plot_type} for column: {column}")

    def bivariate_graphs(self):
        x_column = st.selectbox("Select X-axis Column", self.data.columns)
        y_column = st.selectbox("Select Y-axis Column", self.data.columns)
        plot_type = st.selectbox("Select Plot Type", ["Scatterplot", "Lineplot", "Barplot"])

        if pd.api.types.is_numeric_dtype(self.data[x_column]) and pd.api.types.is_numeric_dtype(self.data[y_column]):
            if plot_type == "Scatterplot":
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=self.data[x_column], y=self.data[y_column])
                st.pyplot(plt)
            elif plot_type == "Lineplot":
                plt.figure(figsize=(10, 6))
                sns.lineplot(x=self.data[x_column], y=self.data[y_column])
                st.pyplot(plt)
        elif pd.api.types.is_categorical_dtype(self.data[x_column]) or self.data[x_column].dtype == 'object':
            if plot_type == "Barplot":
                plt.figure(figsize=(10, 6))
                sns.barplot(x=self.data[x_column], y=self.data[y_column])
                st.pyplot(plt)
        else:
            st.write(f"Cannot plot {plot_type} for columns: {x_column} or {y_column}")

    def multivariate_graphs(self):
        columns = st.multiselect("Select Columns for Multivariate Analysis", self.data.columns)
        plot_type = st.selectbox("Select Plot Type", ["Pairplot", "Heatmap"])

        if plot_type == "Pairplot":
            if all(pd.api.types.is_numeric_dtype(self.data[col]) for col in columns):
                plt.figure(figsize=(10, 6))
                sns.pairplot(self.data[columns])
                st.pyplot(plt)
            else:
                st.write("All selected columns must be numeric for Pairplot.")
        elif plot_type == "Heatmap":
            if len(columns) > 1 and all(pd.api.types.is_numeric_dtype(self.data[col]) for col in columns):
                plt.figure(figsize=(10, 6))
                sns.heatmap(self.data[columns].corr(), annot=True, cmap='coolwarm')
                st.pyplot(plt)
            else:
                st.write("Select at least two numeric columns for a Heatmap.")



# if __name__ == "__main__":
#     eda = ExploratoryDataAnalysis("clothes_price_prediction_data.csv")
#     eda.run()
