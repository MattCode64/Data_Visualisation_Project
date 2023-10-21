import calendar
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import streamlit as st


def select_year(df, key):
    years = df['date_heure'].dt.year.unique().tolist()  # Créez une liste d'années uniques
    year = st.slider('Select year :', min_value=min(years), max_value=max(years), value=max(years), key=key)
    return year


def select_numeric_column(df):
    numeric_columns = df.select_dtypes(['float64', 'int64']).columns.tolist()
    selected_column = st.selectbox('Select column :', numeric_columns)
    return selected_column


def select_multiple_columns(df):
    columns = df.columns.tolist()
    selected_columns = st.multiselect('Select columns :', columns)
    return selected_columns


def numeric_column(df):
    # convert df to numeric columns only
    num_df = df.select_dtypes(['float64', 'int64'])
    return num_df


def datavisualisation():
    st.markdown("## Data Visualisation :")
    # Variables
    year_selected = select_year(dataframe, "1")
    select_column = select_numeric_column(dataframe)

    # Plot Histogram
    plot_histogram(dataframe, select_column, year_selected)

    # Plot Boxplot
    plot_boxplot(dataframe, select_column, year_selected)

    # Plot Line Chart
    plot_line_chart(dataframe, select_column, year_selected)

    # Plot Scatter Plot
    plot_scatter_plot(dataframe, select_column, year_selected)

    # Filter new dataframe with only numeric columns
    num_df = numeric_column(dataframe)

    # Plot Matrix Correlation
    plot_matrix_correlation(num_df)


def advanced_datavisualisation():
    st.markdown("## Advanced Data Visualisation :")

    # Filter new dataframe with only numeric columns
    num_df = numeric_column(dataframe)

    # Variables
    selected_year = select_year(dataframe, "2")
    selected_multiple_columns = select_multiple_columns(num_df)

    # Plot Pie Chart
    plot_pie_chart(dataframe, selected_multiple_columns, selected_year)

    # Plot Line Chart
    plot_linechart(dataframe, selected_multiple_columns)

    # Plot Heatmap
    line_function(dataframe, selected_multiple_columns, selected_year)


def begin_function():
    banner_image(img_path)
    introduction()
    description()
    st.divider()
    display_data(dataframe)
    expander(dataframe)


@st.cache_data(persist="disk")
def banner_image(image):
    try:
        st.image(image, use_column_width=True, caption="Source : DALL-E 3", width=1000)
    except Exception as e:
        st.write(f"Une erreur s'est produite lors de l'affichage de l'image : {e}")


@st.cache_data(persist="disk")
def introduction():
    st.markdown("# Data Visualisation Project")
    st.markdown("## Using Streamlit")
    st.markdown("By : **FREIRE Matthieu DE1**")
    st.markdown("## ***Electricity and Gas consumption in Metropolitan France***")
    st.markdown("Link : [Government Data](https://www.data.gouv.fr/fr/datasets/consommation-quotidienne-brute/)")
    st.markdown("You can download it using the link above.")
    st.divider()


@st.cache_data(persist="disk")
def description():
    st.markdown("## ***Description of the data :*** ")
    st.markdown("This dataset presents the consumption curves for electricity (per hour in MW) "
                "and gas (per hour in MW PCS 0°C).")
    st.markdown("The data is provided by GRTgaz and Teréga, the two gas transport system operators in France.")
    st.markdown("The data is provided by RTE, the French electricity transmission system operator.")

    st.markdown("**EDF is the intermediary between RTE and consumers (households)**")
    st.markdown("**Téréga manages the gas transmission network in the southwest quarter of France, while GRTgaz "
                "manages the network in the rest of the country.**")
    st.markdown("**Their energy is then transported to homes by GRDF.**")

    st.markdown("### ***Problematic :***")
    st.markdown("The objective of this project is to visualize the data in order to understand the "
                "consumption of electricity and gas in France through many variables and to be able to "
                "make statements about it.")


@st.cache_data(persist="disk")
def load_data(data):
    data = pd.read_csv(data, sep=",")
    return data


@st.cache_data(persist="disk")
def display_data(data):
    st.markdown("**Dataframe :**")
    st.dataframe(data)


@st.cache_data(persist="disk")
def expander(data):
    with st.expander("Click here to see the explanation of the columns"):
        st.markdown("- date_time: the combined date and time, probably in a specific time zone (+01:00 indicates the "
                    "time zone).\n- date: the specific date of data recording.\n- heure: the specific time the data was "
                    "recorded.\n- consommation_brute_gaz_grtgaz: gas consumption by GRTgaz.\n- statut_grtgaz: the status "
                    "of the data provided by GRTgaz.\n- consommation_brute_gaz_terega: gas consumption by Teréga.\n- "
                    "statut_terega: the status of the data provided by Teréga.\n- consommation_brute_gaz_totale: total gas "
                    "consumption.\n- consommation_brute_electricite_rte: electricity consumption.\n- statut_rte: the status"
                    "of the electricity data provided.\n- consommation_brute_totale: total consumption of gas and "
                    "electricity.")

    with st.expander("EDA :"):
        st.markdown("**Shape of the dataframe :**")
        st.code(data.shape, language="python")
        st.markdown("**Columns of the dataframe :**")
        st.code(data.columns, language="python")
        st.markdown("**Data types of the dataframe :**")
        st.code(data.dtypes, language="python")
        st.markdown("**Descriptive statistics :**")
        st.code(data.describe(), language="python")


@st.cache_data(persist="disk")
def plot_histogram(df, column, year):
    with st.expander("Histogram :"):
        st.markdown(f"**Histogram of {column} in {year} :**")
        fig, ax = plt.subplots()
        sns.histplot(df[df['date_heure'].dt.year == year], x=column, ax=ax)
        st.pyplot(fig)


# Function to plot boxplot with selected column and year with plotly
@st.cache_data(persist="disk")
def plot_boxplot(df, column, year):
    with st.expander("Boxplot :"):
        st.markdown(f"**Boxplot of {column} in {year} :**")
        fig = px.box(df[df['date_heure'].dt.year == year], y=column)
        st.plotly_chart(fig)


# Function to plot line chart with selected column and year with st.line_chart
@st.cache_data(persist="disk")
def plot_line_chart(df, column, year):
    with st.expander("Line Chart :"):
        st.markdown(f"**Line Chart of {column} in {year} :**")
        fig = px.line(df[df['date_heure'].dt.year == year], x='date_heure', y=column)
        st.plotly_chart(fig)


# Function to plot scatter plot with selected column and year with st.scatter_chart
def plot_scatter_plot(df, column, year):
    with st.expander("Scatter Plot :"):
        st.markdown(f"**Scatter Plot of {column} in {year} :**")
        st.scatter_chart(df[df['date_heure'].dt.year == year], x='date_heure', y=column)


# Function to plot matrix correlation with all numeric columns with plotly
@st.cache_data(persist="disk")
def plot_matrix_correlation(df):
    with st.expander("Matrix Correlation :"):
        st.markdown("**Matrix Correlation :**")
        fig = go.Figure(data=go.Heatmap(z=df.corr(), x=df.columns, y=df.columns, colorscale='Viridis'))
        st.plotly_chart(fig)


# Function to plot every month of the year with a pie chart of selected column and year with plotly
@st.cache_data(persist="disk")
def plot_pie_chart(df, columns, year):
    if not columns:
        st.warning('No columns selected for plotting.')
        return

    df_year = df[df['date_heure'].dt.year == year].copy()
    df_year['month'] = df_year['date_heure'].dt.month
    df_year['month'] = df_year['month'].apply(lambda x: calendar.month_abbr[x])

    with st.expander("Pie Charts :", expanded=True):
        if len(columns) == 1:
            column = columns[0]
            fig = px.pie(df_year, values=column, names='month')
            fig.update_layout(showlegend=False, height=400, width=400)  # Hide legend and adjust margins
            fig.update_traces(textinfo='percent+label')  # Adjust the displayed text in the chart
            st.plotly_chart(fig, use_container_width=True)  # Use the full width of the container
        else:
            col1, col2 = st.columns(2)
            for i, column in enumerate(columns):
                fig = px.pie(df_year, values=column, names='month')
                fig.update_layout(showlegend=False, height=400, width=400)  # Cacher la légende et ajuster les marges
                fig.update_traces(textinfo='percent+label')  # Ajuster le texte affiché dans le graphique

                if i % 2 == 0:
                    with col1:
                        st.plotly_chart(fig, use_container_width=True)  # Utiliser la largeur complète du conteneur
                else:
                    with col2:
                        st.plotly_chart(fig, use_container_width=True)  # Utiliser la largeur complète du conteneur


@st.cache_data(persist="disk")
def plot_linechart(df, columns):
    with st.expander("Line Chart :"):
        if not columns:
            st.warning('No columns selected for plotting.')
            return

        # Extraire l'heure de la colonne datetime
        df['heure'] = df['date_heure'].dt.hour.copy()

        # Grouper par heure et calculer la moyenne des colonnes numériques
        grouped = df.groupby('heure')[columns].mean()

        traces = []
        for column in columns:
            trace = go.Scatter(
                x=grouped.index,
                y=grouped[column],
                mode='lines+markers',
                name=column
            )
            traces.append(trace)

        # Définition de la mise en page du graphique
        layout = go.Layout(
            title='Consommation Moyenne d\'Énergie par Heure de la Journée',
            xaxis=dict(title='Heure de la Journée'),
            yaxis=dict(title='Consommation Moyenne d\'Énergie'),
            hovermode='closest'
        )

        # Création du graphique
        fig = go.Figure(data=traces, layout=layout)

        # Afficher le graphique dans Streamlit
        st.plotly_chart(fig)


@st.cache_data(persist="disk")
def line_function(data, columns, year):
    with st.expander("Heatmap :"):
        try:
            # Filter data by the selected year
            data = data[data['date_heure'].dt.year == year]

            # Extract the hour from the 'date_heure' column
            data['hour'] = data['date_heure'].dt.hour

            # Group by hour and calculate the mean of the selected columns
            hourly_avg = data.groupby('hour')[columns].mean()

            # Create a heatmap with the selected columns
            trace = go.Heatmap(z=[hourly_avg[col] for col in columns],
                               x=hourly_avg.index,
                               y=columns)

            # Set layout options
            layout = go.Layout(
                title=f'Heatmap de la Consommation Moyenne par Heure en {year}',
                xaxis=dict(title='Heure de la Journée'),
                yaxis=dict(title='Consommation Type', tickangle=-60),
            )

            # Create the figure
            fig = go.Figure(data=[trace], layout=layout)

            # Show the plot in Streamlit
            st.plotly_chart(fig)

        except Exception as e:
            st.write(f"Une erreur s'est produite : {e}")


if __name__ == '__main__':
    print("Hello World!")
    # Every Path
    img_path = "../../data/imagedataviz.png"
    data_path = (r"C:\Data\Projet CODE\Code Python\Data Visualisation\DataViz "
                 r"S7\Project\data\gouv_data\consommation-quotidienne-brute_clean.csv")

    # Load data with streamlit function cache
    dataframe = load_data(data_path)
    dataframe['date_heure'] = pd.to_datetime(dataframe['date_heure'], utc=True)

    # Beginning Functions
    begin_function()
    st.divider()
    datavisualisation()
    st.divider()

    advanced_datavisualisation()
    st.divider()

    st.markdown("## ***Thank you for your attention !***")
    print("Goodbye World!")
