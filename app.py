
import streamlit as st
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity


# Загрузка данных
movies = pd.read_pickle("movies_app.pkl")
movies = movies.reset_index(drop=True)

# Название фильма + год для выпадающего списка
movies["display_title"] = (
    movies["title_movies"]
    + " ("
    + movies["year"].astype(int).astype(str)
    + ")"
)


# Создание матрицы жанров
mlb = MultiLabelBinarizer()
genres_matrix = mlb.fit_transform(movies["genres_list"])


# Функция рекомендаций
def recommend_by_genres(movie_idx, top_k=10):
    movie_vector = genres_matrix[movie_idx].reshape(1, -1)

    similarities = cosine_similarity(
        movie_vector,
        genres_matrix
    )[0]

    # Сортируем фильмы по сходству
    similar_indices = similarities.argsort()[::-1]

    # Исключаем выбранный фильм
    similar_indices = similar_indices[similar_indices != movie_idx]

    # Берём top-k
    top_indices = similar_indices[:top_k]

    result = movies.iloc[top_indices][
        ["title_movies", "year", "genres_list"]
    ].copy()

    result["similarity"] = similarities[top_indices]

    return result


# Интерфейс
st.title("Поиск похожих фильмов")

st.write(
    "Рекомендации строятся на основе сходства жанров фильмов."
)

selected_movie = st.selectbox(
    "Выберите фильм:",
    movies["display_title"].sort_values()
)

# Получаем индекс выбранного фильма
movie_idx = movies[
    movies["display_title"] == selected_movie
].index[0]

top_k = st.slider(
    "Количество рекомендаций:",
    min_value=5,
    max_value=20,
    value=10
)


if st.button("Найти похожие фильмы"):
    recommendations = recommend_by_genres(
        movie_idx,
        top_k
    )

    st.subheader("Рекомендации")

    for _, row in recommendations.iterrows():
        genres = ", ".join(row["genres_list"])

        st.write(
            f"**{row['title_movies']} ({int(row['year'])})**  \n"
            f"{genres}"
        )
