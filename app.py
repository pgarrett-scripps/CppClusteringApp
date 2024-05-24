import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px

# set random seed
random_seed = 42

with st.sidebar:
    csv_file = st.file_uploader("Upload a CSV file", type=['csv'])

    if csv_file is None:
        st.warning("No file uploaded")
        st.stop()

    df = pd.read_csv(csv_file)

    # drop rows with a missing value
    df.dropna(inplace=True)
    columns = df.columns.tolist()
    label = st.selectbox("Select the column to use as label", columns)
    remaining_columns = [col for col in columns if col != label]
    data_cols = st.multiselect("Select the columns to use for clustering", remaining_columns, remaining_columns)
    kmeans = st.slider("Select the number of clusters", min_value=2, max_value=10, value=2)
    n_components = 2

data = df[data_cols].values

kmeans = KMeans(n_clusters=kmeans, random_state=random_seed)
clusters = kmeans.fit_predict(data)
df['cluster'] = clusters


st.title("KMeans Clustering Analysis")

st.subheader("Clusters in PCA space")
st.caption("Use the 'Pan' Option to select single data points. Use the 'Box Select' and 'Lasso Select' options to "
           "select multiple data points")
if n_components == 2:
    pcas = PCA(n_components=n_components).fit_transform(data)
    df['pca1'] = pcas[:, 0]
    df['pca2'] = pcas[:, 1]

    df['cluster'] = df['cluster'].astype(str)
    fig = px.scatter(df, x='pca1',
                     y='pca2',
                     color='cluster',
                     hover_data=[label] + data_cols,
                     color_discrete_sequence=px.colors.qualitative.Alphabet_r)
    event = st.plotly_chart(fig, on_select="rerun")

elif n_components == 3:
    pcas = PCA(n_components=n_components).fit_transform(data)
    df['pca1'] = pcas[:, 0]
    df['pca2'] = pcas[:, 1]
    df['pca3'] = pcas[:, 2]

    # 3D plot
    fig = px.scatter_3d(df, x='pca1', y='pca2', z='pca3', color='cluster', hover_data=[label] + data_cols)
    event = st.plotly_chart(fig, on_select="rerun")

#st.dataframe(df)
#event
st.subheader("Selected data points")
if event["selection"]["point_indices"]:
    selected_data = df.iloc[event["selection"]["point_indices"]]
    st.write(selected_data)
else:
    st.dataframe(df)





