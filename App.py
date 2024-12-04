import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.metrics import r2_score

# Título del dashboard
st.title("Sistema de Recomendación de Películas")

# Cargar datasets desde archivos locales (ya presentes)
ratings_file = "ratings_small.csv"  # Ruta al archivo de ratings
movies_file = "movies_metadata.csv"  # Ruta al archivo de metadata de películas

# Cargar los archivos CSV
users = pd.read_csv(ratings_file)
movies = pd.read_csv(movies_file)

# Preprocesamiento
indices_a_eliminar = [19729, 29502, 35586, 19730, 29503, 35587]
movies = movies.drop(indices_a_eliminar)

# Asegurarse de que ambas columnas son cadenas
users['movieId'] = users['movieId'].astype(str)
movies['id'] = movies['id'].astype(str)

# Crear diccionario de usuarios y ratings
data = {}
for _, row in users.iterrows():
    user = row['userId']
    movie = row['movieId']
    rating = row['rating']

    if user not in data:
        data[user] = {}
    data[user][movie] = rating

# Funciones de similitud
def euclidean_similarity(person1, person2):
    common_ranked_items = [itm for itm in data[person1] if itm in data[person2]]
    rankings = [(data[person1][itm], data[person2][itm]) for itm in common_ranked_items]
    distance = [pow(rank[0] - rank[1], 2) for rank in rankings]
    return 1 / (1 + sum(distance))

def pearson_similarity(person1, person2):
    common_ranked_items = [itm for itm in data[person1] if itm in data[person2]]
    n = len(common_ranked_items)
    if n == 0:
        return 0

    s1 = sum([data[person1][item] for item in common_ranked_items])
    s2 = sum([data[person2][item] for item in common_ranked_items])

    ss1 = sum([pow(data[person1][item], 2) for item in common_ranked_items])
    ss2 = sum([pow(data[person2][item], 2) for item in common_ranked_items])

    ps = sum([data[person1][item] * data[person2][item] for item in common_ranked_items])

    num = n * ps - (s1 * s2)
    den = math.sqrt((n * ss1 - math.pow(s1, 2)) * (n * ss2 - math.pow(s2, 2)))
    return (num / den) if den != 0 else 0

# Función para recomendar películas
def user_recommend(person, bound, data, similarity=pearson_similarity):
    scores = [(similarity(person, other), other) for other in data if other != person]
    scores.sort(reverse=True)
    scores = scores[:bound]

    recomms = {}
    for sim, other in scores:
        ranked = data[other]
        for itm in ranked:
            if itm not in data[person]:
                weight = sim * ranked[itm]
                if itm in recomms:
                    s, weights = recomms[itm]
                    recomms[itm] = (s + sim, weights + [weight])
                else:
                    recomms[itm] = (sim, [weight])

    for r in recomms:
        sim, item = recomms[r]
        recomms[r] = sum(item) / sim
    return recomms

# Interfaz interactiva
st.sidebar.header("Opciones de Recomendación")
user_id = st.sidebar.number_input("ID del Usuario", min_value=1, value=1, step=1)
num_recommendations = st.sidebar.slider("Número de Recomendaciones", min_value=1, max_value=10, value=5, step=1)

# Fijar cantidad de vecinos a 3 y métrica de similitud a Pearson (sin mostrar en la interfaz)
bound = 3
similarity_func = pearson_similarity  # Fijado a Pearson

if st.sidebar.button("Generar Recomendaciones"):
    rec = user_recommend(user_id, bound, data, similarity=similarity_func)
    top_recommendations = sorted(rec.items(), key=lambda item: item[1], reverse=True)

    # Filtrar las recomendaciones para que solo se muestren las que tienen título en el dataset movies
    valid_recommendations = []
    for movie_id, score in top_recommendations:
        movie_title = movies[movies['id'] == movie_id]['title'].values
        if movie_title.size > 0:
            movie_title = movie_title[0]
            valid_recommendations.append((movie_title, score))
        
        # Limitar las recomendaciones a la cantidad seleccionada por el usuario
        if len(valid_recommendations) == num_recommendations:
            break

    st.write("### Recomendaciones")
    for movie_title, score in valid_recommendations:
        st.write(f"- **{movie_title}**: Puntuación {score:.2f}")