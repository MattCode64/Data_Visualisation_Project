import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import streamlit as st


def introduction():
    st.markdown("# Data Visualisation of Government Data")
    st.write("Electricity and Gas consumption in France")
    st.write("Link : [Government Data](https://www.data.gouv.fr/fr/datasets/consommation-quotidienne-brute/)")
    st.write("You can download it using the link above.")
    st.divider()


path = (r"C:\Data\Projet CODE\Code Python\Data Visualisation\DataViz "
        r"S7\Project\data\gouv_data\consommation-quotidienne-brute.csv")

introduction()


# %%
@st.cache_data(persist="disk")
def load_data(path_data):
    data = pd.read_csv(path_data, delimiter=';')
    return data


dataframe = load_data(path)


def display_data(data):
    st.markdown("# Data Importation and Quick View")
    st.write("Dataframe (head):")

    code_1 = """
        dataframe = pd.read_csv(path, delimiter=';')
        dataframe.head()
        """
    st.code(code_1, language="python")

    st.dataframe(data.head())  # Print the data of the dataset

    list_col = data.columns.to_list()

    st.write("This is our columns:")
    st.code(list_col)

    st.write("This is our shape:")
    st.code(data.shape)

    st.write("This is our describe (for `date`, `hour`):")
    st.code(data[['date', 'heure']].describe())

    st.write("This is our describe (for `consommation_brute_gaz_grtgaz`, "
             "`consommation_brute_gaz_terega`, `consommation_brute_gaz_totale`):")
    st.code(data[['consommation_brute_gaz_grtgaz', 'consommation_brute_gaz_terega',
                  'consommation_brute_gaz_totale']].describe())

    st.write("Explanation of the columns:")
    st.markdown("- date_time: the combined date and time, probably in a specific time zone (+01:00 indicates the "
                "time zone).\n- date: the specific date of data recording.\n- heure: the specific time the data was "
                "recorded.\n- consommation_brute_gaz_grtgaz: gas consumption by GRTgaz.\n- statut_grtgaz: the status "
                "of the data provided by GRTgaz.\n- consommation_brute_gaz_terega: gas consumption by Teréga.\n- "
                "statut_terega: the status of the data provided by Teréga.\n- consommation_brute_gaz_totale: total gas "
                "consumption.\n- consommation_brute_electricite_rte: electricity consumption.\n- statut_rte: the status"
                "of the electricity data provided.\n- consommation_brute_totale: total consumption of gas and "
                "electricity.")

    st.divider()


display_data(dataframe)

# %%
st.markdown("# Data Cleaning :")
st.write("Lot of data cleaning is already done")
st.write("We will just display and change the type of `date` and `heure` to datetime")


def convert_to_datetime(data):
    dataframe['date_heure'] = pd.to_datetime(dataframe['date_heure'], utc=True)
    dataframe['date'] = pd.to_datetime(dataframe['date'], dayfirst=True)
    dataframe['heure'] = pd.to_datetime(dataframe['heure']).dt.time
    code_2 = """
        dataframe['date_heure'] = pd.to_datetime(dataframe['date_heure'])
        dataframe['date'] = pd.to_datetime(dataframe['date']).dt.date
        dataframe['heure'] = pd.to_datetime(dataframe['heure']).dt.time
    """
    st.code(code_2, language="python")

    return data


dataframe = convert_to_datetime(dataframe)


def drop_pair_rows_and_sort(data):
    # Remove every pair row
    data = data.drop(data.index[::2])

    # Sort by date and time
    data.sort_values(by=['date_heure'])

    # Drop NaN values
    data = dataframe.dropna()

    st.write("Dropping pair row:")
    code_3 = """
    # Remove every pair row
    dataframe = dataframe.drop(dataframe.index[::2])
    
    # Sort by date and time
    dataframe = dataframe.sort_values(by=['date_heure'])
    """
    st.code(code_3, language="python")

    return data


# Call the function to drop pair rows and sort the DataFrame
dataframe = drop_pair_rows_and_sort(dataframe)
dataframe.reset_index(inplace=True, drop=True)

st.markdown("### Dataframe that we will use :")


def display_dataframe(data):
    st.dataframe(data.head(50))


display_dataframe(dataframe)
st.divider()
# %%
st.markdown("# EDA :")


def display_types(data):
    st.markdown("Types :")
    st.code(data.dtypes)


def display_missing_values(data):
    st.markdown("Missing values :")
    st.code(data.isna().sum())


def display_num_columns(data):
    st.markdown("Numeric columns :")
    st.code(data.select_dtypes(include=['int64', 'float64']).columns.to_list())


def display_shape(data):
    st.markdown("Shape :")
    st.code(data.shape)


display_types(dataframe)
display_missing_values(dataframe)
display_num_columns(dataframe)
display_shape(dataframe)

st.markdown("**Head**")
st.dataframe(dataframe.select_dtypes(include=['int64', 'float64']).head())

# Insert in a list the numeric columns
numeric_columns = dataframe.select_dtypes(['int', 'float']).columns

selected_column = st.selectbox('Select a column :', numeric_columns)


def dist_plot(data, column):
    st.markdown("Distribution chart of numeric columns :")
    fig, ax = plt.subplots()
    sns.histplot(data[column], ax=ax)
    st.pyplot(fig)


def box_plot(data, column):
    st.markdown("Boxplot of numeric columns :")
    fig, ax = plt.subplots()
    sns.boxplot(data=data[column], ax=ax, orient='h')
    st.pyplot(fig)


def line_plot(data, column):
    st.markdown("Line chart of numeric columns :")
    st.line_chart(data[column])


def scatter_plot(data):
    st.header('Scatter chart of the variables')
    st.markdown("### Variables for the scatter plot")
    var1 = st.selectbox('Variable 1', ['consommation_brute_gaz_grtgaz', 'consommation_brute_gaz_terega',
                                       'consommation_brute_gaz_totale',
                                       'consommation_brute_electricite_rte', 'consommation_brute_totale'])

    var2 = st.selectbox('Variable 2', ['consommation_brute_gaz_grtgaz', 'consommation_brute_gaz_terega',
                                       'consommation_brute_gaz_totale',
                                       'consommation_brute_electricite_rte', 'consommation_brute_totale'])

    st.markdown(f"Scatter plot **{var1}** / **{var2}**")
    st.markdown("Scatter plot of numeric columns :")
    st.scatter_chart(data[[var1, var2]])


def correlation(data):
    st.markdown("Correlation between numeric columns :")
    st.header('Correlation between the variables')
    st.write("Correlation between numeric columns :")
    st.code(data.corr())


dist_plot(dataframe, selected_column)
box_plot(dataframe, selected_column)
line_plot(dataframe, selected_column)
scatter_plot(dataframe)

st.divider()

# %%

st.markdown("# Advance Data Visualisation :")
st.markdown("## Pie chart :")


def pie_chart_by_year(data):
    list_year = data['date_heure'].dt.year.unique()
    selected_years = st.multiselect('Select year', list_year, key='year')

    if not selected_years:
        st.warning("Please select at least one year.")
        return

    columns = data.select_dtypes(['int', 'float']).columns
    selected_features = st.multiselect('Select feature :', columns, key='feature')

    if not selected_features:
        st.warning("Please select at least one feature.")
        return

    # Boucle pour chaque combinaison d'année et de caractéristique
    for year in selected_years:
        for feature in selected_features:
            df_year = data[data['date_heure'].dt.year == year]
            fig = px.pie(df_year,
                         values=feature,
                         names=df_year['date_heure'].dt.month_name(),
                         title=f'({feature}) in {year} by Month',
                         template='plotly_dark')

            st.plotly_chart(fig)


pie_chart_by_year(dataframe)


def plot_selected_data(data):
    st.markdown("## Advanced Line chart :")
    # Sélection de l'année par l'utilisateur
    list_year = data['date_heure'].dt.year.unique()
    selected_years = st.selectbox('Select year', list_year, key='yeader')
    if not selected_years:
        st.warning("Please select at least one year.")
        return

    # Filtrer les données par année
    df = data[data['date_heure'].dt.year == selected_years]

    # Sélection des variables numériques par l'utilisateur
    columns = data.select_dtypes(['int', 'float']).columns
    selected_features = st.multiselect('Select feature :', columns, key='feadedeture')
    if not selected_features:
        st.warning("Please select at least one feature.")
        return

    # Regrouper le dataframe par heure et calculer la moyenne des colonnes numériques sélectionnées
    grouped = df.groupby('heure')[selected_features].mean()

    # Créer des traces pour chaque variable numérique sélectionnée
    traces = [
        go.Scatter(
            x=grouped.index,
            y=grouped[var],
            mode='lines+markers',
            name=var
        ) for var in selected_features
    ]

    # Définition de la mise en page du graphique
    layout = go.Layout(
        title=f"Consommation Moyenne d'Énergie par Heure de la Journée en {selected_years}",
        xaxis=dict(title='Heure de la Journée'),
        yaxis=dict(title='Consommation Moyenne d\'Énergie'),
        showlegend=True,
        hovermode='closest'
    )

    # Création du graphique
    fig = go.Figure(data=traces, layout=layout)

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)


# Appeler la fonction avec le dataframe en paramètre
plot_selected_data(dataframe)
