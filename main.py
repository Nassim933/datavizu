import pandas as pd
import streamlit as st
import plotly.express as px
import altair as alt
import functools
import time
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import Spacer
import io



@st.cache_data()
def load_data():
    df = pd.read_csv('data.csv', encoding='ISO-8859-1', sep=';')
    return df[df['Statistique'] == 'Nombre']
def preprocess_data(df):
    df['Valeurs'] = pd.to_numeric(df['Valeurs'], errors='coerce').fillna(0).astype(int)
    return df
def aggregate_data(df):
    return df.groupby(df['Unite temps'].str[:4])['Valeurs'].sum()
def aggregate_data_2(df):
    return df.groupby(['Unite temps', 'Zone_geographique'])['Valeurs'].sum().reset_index()
def create_subindicator_mapping(df):
    mapping = {}
    for indicateur in df['Indicateur'].unique():
        subindicateurs = df[df['Indicateur'] == indicateur]['Sous_indicateur'].dropna().unique()
        subindicateurs = [s for s in subindicateurs if s != "Non Renseigné"]
        mapping[indicateur] = subindicateurs
    return mapping
def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

        with open('execution_log.txt', 'a') as log_file:
            log_file.write(f'{timestamp} - {func.__name__} executed in {execution_time:.2f} seconds\n')

        return result

    return wrapper
@log_execution_time
def show_data(df):
    st.write("Download the complete dataset :")
    csv_data = df.to_csv(index=False, encoding='utf-8')
    st.download_button(
        label="Download the complete dataset",
        data=csv_data,
        file_name="dataset_complet.csv",
        key="download_button_dataset_complet"
    )
@log_execution_time
def graphique1(df):

    st.title("Number of crimes per year")

    indicateurs = df['Indicateur'].unique()
    indicateur_choisi = st.selectbox("Choose an indicator for the ordinate:", sorted(indicateurs), key="select_indicateur")

    st.write(f"You have selected the {indicateur_choisi}")

    df_indicateur = df[df['Indicateur'] == indicateur_choisi]

    subindicator_mapping = create_subindicator_mapping(df)

    sous_indicateur_choisi = None

    if st.checkbox("Filter by sub-indicator"):
        if indicateur_choisi in subindicator_mapping:
            sous_indicateurs = subindicator_mapping[indicateur_choisi]
            sous_indicateur_choisi = st.selectbox("Sous-indicateur :", sous_indicateurs)

            filtered_data = df_indicateur[(df_indicateur['Sous_indicateur'] == sous_indicateur_choisi)]

            data_aggregated = aggregate_data(filtered_data)

            st.subheader(f"{indicateur_choisi} ({sous_indicateur_choisi})")

            if len(data_aggregated) > 1:
                st.line_chart(data_aggregated)
            else:
                st.warning("Not enough data to generate the curve graph.")
        else:
            st.warning(f"No sub-indicator available for {indicateur_choisi}.")
    else:
        data_aggregated = aggregate_data(df_indicateur)

        st.subheader(f"{indicateur_choisi}")

        if len(data_aggregated) > 1:
            st.line_chart(data_aggregated)
        else:
            st.warning("Not enough data to generate the curve graph.")

    if st.button(f"Download filtered dataset for {indicateur_choisi} ({sous_indicateur_choisi if sous_indicateur_choisi else 'Tous'})"):
        df_filtre = df_indicateur[(df_indicateur['Sous_indicateur'] == sous_indicateur_choisi) if sous_indicateur_choisi else True]
        columns_to_export = ['Valeurs', 'Indicateur', 'Sous_indicateur']
        df_filtre = df_filtre[columns_to_export]

        csv_data = df_filtre.to_csv(index=False, encoding='utf-8')
        sous_indicateur_label = sous_indicateur_choisi if sous_indicateur_choisi else 'Tous'
        st.download_button(
            label=f"Download filtered dataset for {indicateur_choisi} ({sous_indicateur_label})",
            data=csv_data,
            file_name=f"dataset_filtre_{indicateur_choisi}_{sous_indicateur_label}.csv",
            key=f"download_button_{indicateur_choisi}"
        )
    data_aggregated = aggregate_data(df_indicateur)
    if len(data_aggregated) > 1:
        fig = go.Figure(data=go.Scatter(x=data_aggregated.index, y=data_aggregated.values, mode='lines'))
        pio.write_image(fig, "img1.png")
    else:
        st.warning("Not enough data to generate the curve graph.")
@log_execution_time
def graphique2(df):
    st.title("Number of crimes for each year by department")

    indicateurs_2 = [
        "Destructions et dégradations volontaires",
        "Infractions à la législation sur les stupéfiants",
        "Vols et tentatives de vols liés aux véhicules",
        "Cambriolages et tentatives",
        "Violences physiques",
        "Violences sexuelles",
        "Vols et tentatives de vols avec violence",
        "Vols et tentatives de vols sans violence"
    ]

    indicateur_choisi_2 = st.selectbox("Choose an ordinate indicator for the second graph:",
                                       sorted(indicateurs_2))
    st.write(f"You have selected the {indicateur_choisi_2}")

    zones_geographiques = df['Zone_geographique'].dropna().unique()
    zones_geographiques = [z for z in zones_geographiques if z != "Non Renseigné"]
    zone_geographique_choisie = st.selectbox("Zone géographique :", zones_geographiques)

    df_filtered = df[(df['Indicateur'] == indicateur_choisi_2) & (df['Zone_geographique'] == zone_geographique_choisie)]

    data_aggregated = aggregate_data_2(df_filtered)

    st.subheader(f"Number of crimes for indicator {indicateur_choisi_2} ({zone_geographique_choisie})")

    if len(data_aggregated) > 0:
        fig = px.line(data_aggregated, x='Unite temps', y='Valeurs',
                      labels={'Unite temps': 'Année', 'Valeurs': 'Nombre de crimes'}, markers=True, line_shape='linear')

        fig.update_traces(hoverinfo='x+y', mode='lines+markers', marker=dict(size=8, color='blue'))

        st.plotly_chart(fig)
    else:
        st.warning("No data available to generate the curve graph.")

    if st.button(f"Download filtered dataset for {indicateur_choisi_2} ({zone_geographique_choisie})"):
        df_filtre_2 = df[
            (df['Indicateur'] == indicateur_choisi_2) & (df['Zone_geographique'] == zone_geographique_choisie)]
        columns_to_export = ['Valeurs', 'Indicateur', 'Sous_indicateur', 'Zone_geographique']
        df_filtre_2 = df_filtre_2[columns_to_export]

        csv_data_2 = df_filtre_2.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label=f"Download filtered dataset for {indicateur_choisi_2} ({zone_geographique_choisie})",
            data=csv_data_2,
            file_name=f"dataset_filtre_{indicateur_choisi_2}_{zone_geographique_choisie}.csv",
            key=f"download_button_{indicateur_choisi_2}"
        )

    if len(data_aggregated) > 0:
        fig = px.line(data_aggregated, x='Unite temps', y='Valeurs',
                      labels={'Unite temps': 'Année', 'Valeurs': 'Nombre de crimes'},
                      markers=True, line_shape='linear')

        fig.update_traces(hoverinfo='x+y', mode='lines+markers', marker=dict(size=8, color='blue'))

        fig.write_image("img2.png")
@log_execution_time
def graphique3(df):
    st.title("Histogram of sub-indicators by year")

    indicateurs = df['Indicateur'].unique()
    indicateur_choisi = st.selectbox("Choose an indicator :", sorted(indicateurs))

    st.write(f"You have selected the {indicateur_choisi}")

    df_indicateur = df[df['Indicateur'] == indicateur_choisi]

    sous_indicateurs = df_indicateur['Sous_indicateur'].dropna().unique()
    sous_indicateurs = [s for s in sous_indicateurs if s != "Non Renseigné"]
    sous_indicateur_choisi = st.selectbox("Sous-indicateur :", sous_indicateurs)
    st.subheader(f"Histogram of sub-indicators for the indicator {indicateur_choisi} ({sous_indicateur_choisi})")

    data_filtered = df[(df['Indicateur'] == indicateur_choisi) & (df['Sous_indicateur'] == sous_indicateur_choisi)]

    if len(data_filtered) > 0:
        data_aggregated = aggregate_data(data_filtered)
        chart = alt.Chart(data_aggregated.reset_index()).mark_bar().encode(
            x=alt.X('Unite temps:N', title='Année'),
            y=alt.Y('Valeurs:Q', title='Nombre de crimes'),
            color=alt.Color('Unite temps:N', title='Année')
        ).properties(
            width=600,
            height=400
        )

        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No data available to generate histogram.")

    if st.button(f"Download filtered dataset for {indicateur_choisi} ({sous_indicateur_choisi})"):
        df_filtre = df[(df['Indicateur'] == indicateur_choisi) & (df['Sous_indicateur'] == sous_indicateur_choisi)]
        columns_to_export = ['Valeurs', 'Indicateur', 'Sous_indicateur', 'Zone_geographique']
        df_filtre = df_filtre[columns_to_export]

        csv_data = df_filtre.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label=f"Download filtered dataset for {indicateur_choisi} ({sous_indicateur_choisi})",
            data=csv_data,
            file_name=f"dataset_filtre_{indicateur_choisi}_{sous_indicateur_choisi}.csv",
            key=f"download_button_{indicateur_choisi}"
        )

    if len(data_filtered) > 0:
        # Grouper les données par année
        data_aggregated = data_filtered.groupby('Unite temps')['Valeurs'].sum()

        # Créer l'histogramme en utilisant Matplotlib
        plt.figure(figsize=(10, 6))
        data_aggregated.plot(kind='bar', color='blue')
        plt.xlabel('Année')
        plt.ylabel('Nombre de crimes')
        plt.title(f"Histogram of sub-indicators for the indicator {indicateur_choisi} ({sous_indicateur_choisi})")
        plt.savefig('img3.png', dpi=300, bbox_inches='tight')
    else:
        st.warning("No data available to generate histogram.")
@log_execution_time
def graphique4(df):
    st.title("Breakdown of crimes by year")

    years = sorted(map(str, df['Unite temps'].str[:4].unique()))
    annee_choisie = st.selectbox("Choose a year :", years)

    st.write(f"You have chosen the year {annee_choisie}")

    df_annee = df[df['Unite temps'].str[:4] == annee_choisie]
    data = df_annee.groupby('Indicateur')['Valeurs'].sum().reset_index()

    if not data.empty:
        data['Pourcentage'] = (data['Valeurs'] / data['Valeurs'].sum()) * 100

        fig = px.pie(data, names='Indicateur', values='Valeurs',
                     title=f"Breakdown of indicators for the year {annee_choisie}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available to generate the pie chart.")

    if st.button(f"Download the filtered dataset for the year {annee_choisie}"):
        df_filtre = df[df['Unite temps'].str[:4] == annee_choisie]
        columns_to_export = ['Unite temps', 'Valeurs', 'Indicateur']
        df_filtre = df_filtre[columns_to_export]

        csv_data = df_filtre.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label=f"Download the filtered dataset for the year {annee_choisie}",
            data=csv_data,
            file_name=f"dataset_filtre_annee_{annee_choisie}.csv",
            key=f"download_button_{annee_choisie}"
        )

    if not data.empty:
        data['Pourcentage'] = (data['Valeurs'] / data['Valeurs'].sum()) * 100

        fig = px.pie(data, names='Indicateur', values='Valeurs',
                     title=f"Breakdown of indicators for the year {annee_choisie}")


        # Sauvegarde du graphique en PNG
        pio.write_image(fig, "img4.png")
    else:
        st.warning("No data available to generate the pie chart.")
@log_execution_time
def graphique5(df):
    st.title("Histogram of the distribution of the number of crimes by department and by year")

    departements = df['Zone_geographique'].dropna().unique()
    departement_choisi = st.selectbox("Choose a department :", sorted(departements))

    annees = sorted(map(str, df['Unite temps'].str[:4].unique()))
    annees = [annee for annee in annees if annee.isnumeric() and int(annee) >= 2016]
    annee_choisie = st.selectbox("Choose a year :", sorted(annees))

    st.write(f"You have chosen the department {departement_choisi} and the year {annee_choisie}")

    df_departement_annee = df[
        (df['Zone_geographique'] == departement_choisi) & (df['Unite temps'].str[:4] == annee_choisie)]

    if not df_departement_annee.empty:
        chart = alt.Chart(df_departement_annee).mark_bar().encode(
            x=alt.X('sum(Valeurs):Q', title='Nombre de crime'),
            y=alt.Y('Indicateur:N', title='Type de crime'),
            color='Indicateur:N'
        ).properties(
            width=600,
            height=400
        )

        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No data available to generate histogram.")

    if st.button(f"Download filtered dataset for department {departement_choisi} in {annee_choisie}"):
        df_filtre = df[
            (df['Zone_geographique'] == departement_choisi) & (df['Unite temps'].str[:4] == annee_choisie)]
        columns_to_export = ['Valeurs', 'Indicateur', 'Zone_geographique']
        df_filtre = df_filtre[columns_to_export]

        csv_data = df_filtre.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label=f"TDownload filtered dataset for department {departement_choisi} in {annee_choisie}",
            data=csv_data,
            file_name=f"dataset_filtre_departement_annee_{departement_choisi}_{annee_choisie}.csv",
            key=f"download_button_{departement_choisi}_{annee_choisie}"
        )

    if not df_departement_annee.empty:
        # Générer un diagramme en barres avec Matplotlib
        plt.figure(figsize=(8, 6))
        plt.barh(df_departement_annee['Indicateur'], df_departement_annee['Valeurs'])
        plt.xlabel('Nombre de crimes')
        plt.ylabel('Type de crime')
        plt.title(f"Breakdown of crimes for {departement_choisi} en {annee_choisie}")
        plt.tight_layout()
        plt.savefig('img5.png', dpi=300, bbox_inches='tight')
@log_execution_time
def graphique6(df):
    indicateurs_autorises = [
        "Vols et tentatives de vols liés aux véhicules",
        "Cambriolages et tentatives",
        "Destructions et dégradations volontaires",
        "Escroqueries et autres infractions assimilées",
        "Infractions à la législation sur les stupéfiants",
        "Violences physiques",
        "Violences sexuelles",
        "Vols et tentatives de vols avec violence",
        "Vols et tentatives de vols liés aux véhicules",
        "Vols et tentatives de vols sans violence"
    ]
    st.title("Number of crimes per month for a selected crime and year")

    indicateur_choisi = st.selectbox("Choose an indicator :", sorted(indicateurs_autorises))

    st.write(f"You have selected the {indicateur_choisi}")

    df_filtre = df[df['Unite temps'].str.match(r'\d{4}M\d{2}') & df['Unite temps'].notna()]

    annees = df_filtre['Unite temps'].str[:4].unique()
    annee_choisie = st.selectbox("You have selected the", sorted(annees))

    st.write(f"You have chosen the year {annee_choisie}")

    st.title(f"Number of crimes per month for the indicator {indicateur_choisi} in {annee_choisie}")

    df_filtre = df[df['Unite temps'].str.match(r'\d{4}M\d{2}') & df['Unite temps'].notna()]

    df_annee = df_filtre[df_filtre['Unite temps'].str.startswith(annee_choisie)]
    df_indicateur_annee = df_annee[
        (df_annee['Indicateur'] == indicateur_choisi) & (df_annee['Correction'] == 'Série brute')]

    if not df_indicateur_annee.empty:
        st.line_chart(df_indicateur_annee.set_index('Unite temps')['Valeurs'], use_container_width=True)
    else:
        st.warning("No data available to generate the graph.")

    if st.button(f"Download filtered dataset for indicator {indicateur_choisi} in {annee_choisie}"):
        df_filtre = df[
            (df['Indicateur'] == indicateur_choisi) & (df['Unite temps'].str[:4] == annee_choisie) & (
                    df['Correction'] == 'Série brute')]
        columns_to_export = ['Valeurs', 'Indicateur', 'Unite temps']
        df_filtre = df_filtre[columns_to_export]

        csv_data = df_filtre.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label=f"Download filtered dataset for indicator {indicateur_choisi} in {annee_choisie}",
            data=csv_data,
            file_name=f"dataset_filtre_indicateur_annee_{indicateur_choisi}_{annee_choisie}.csv",
            key=f"download_button_{indicateur_choisi}_{annee_choisie}"
        )

    if not df_indicateur_annee.empty:
        plt.bar(df_indicateur_annee['Unite temps'], df_indicateur_annee['Valeurs'])
        plt.xlabel('Unite temps')
        plt.ylabel('Valeurs')
        plt.title(f"Number of crimes per month for the indicator {indicateur_choisi} in {annee_choisie}")

        plt.savefig("img6.png")
    else:
        st.warning("No data available to generate the graph.")
def generate_pdf_with_images_and_titles():
    # Créez un document PDF en mémoire
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)

    # Styles pour les titres et les paragraphes
    styles = getSampleStyleSheet()

    # Titre de la page
    st.text("PDF generation with Streamlit")

    # Liste pour stocker les éléments du PDF
    elements = []
    elements.append(Paragraph("TIME SERIES REPORT ON DELINQUENCY AND INSECURITY ", styles["Title"]))
    elements.append(Paragraph("Number of crimes per year", styles["Title"]))
    img1 = Image("img1.png", width=7 * inch, height=6 * inch)
    elements.append(img1)
    elements.append(Spacer(1, 1.5 * inch))

    # Affichez l'image 2 avec un titre et ajoutez-la au PDF
    elements.append(Paragraph("Number of crimes for each year by department", styles["Title"]))
    img2 = Image("img2.png", width=8 * inch, height=6 * inch)
    elements.append(img2)
    elements.append(Spacer(1, 2 * inch))

    # Affichez l'image 3 avec un titre et ajoutez-la au PDF
    elements.append(Paragraph("Histogram of sub-indicators by year", styles["Title"]))
    img3 = Image("img3.png", width=8 * inch, height=4 * inch)
    elements.append(img3)
    elements.append(Spacer(1, 4.2 * inch))

    elements.append(Paragraph("Breakdown of crimes by year", styles["Title"]))
    img4 = Image("img4.png", width=8 * inch, height=5.5 * inch)
    elements.append(img4)
    elements.append(Spacer(1, 2.8 * inch))

    elements.append(Paragraph("Histogram of the distribution of the number of crimes by department and by year", styles["Title"]))
    img5 = Image("img5.png", width=8 * inch, height=5.5 * inch)
    elements.append(img5)
    elements.append(Spacer(1, 2.2 * inch))

    elements.append(Paragraph("Number of crimes per month for a selected crime and year", styles["Title"]))
    img6 = Image("img6.png", width=8 * inch, height=5.5 * inch)
    elements.append(img6)

    doc.build(elements)

    pdf_data = pdf_buffer.getvalue()

    return pdf_data
@log_execution_time
def main():
    df = load_data()
    df = preprocess_data(df)
    st.title("TIME SERIES ON DELINQUENCY AND INSECURITY")
    show_data(df)
    graphique1(df)
    graphique2(df)
    graphique3(df)
    graphique4(df)
    graphique5(df)
    graphique6(df)
    st.write("Click on the button below to generate and download the PDF.")

    if st.button("Download PDF"):
        pdf_data = generate_pdf_with_images_and_titles()
        st.download_button(
            label="Download PDF",
            data=pdf_data,
            file_name="rapport.pdf",
            key="download_button_pdf"
        )
    st.sidebar.header("Personal information :")
    st.sidebar.write("NADY")
    st.sidebar.write("Nassim")
    st.sidebar.write("2025")
    st.sidebar.write("BIA-2")
    st.sidebar.write("#datavz2023efrei")

if __name__ == "__main__":
    main()

