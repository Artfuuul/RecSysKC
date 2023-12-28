"""
Модуль рекомендательной системы.

Этот модуль содержит реализацию API для рекомендательной системы, использующей
CatBoostClassifier для предсказания интересов пользователя и выдачи рекомендаций.
Он включает функции для загрузки модели, обработки данных, взаимодействия с базой данных
и формирования рекомендательного ленты постов.

Функции:
- get_model_path: Определяет путь к модели.
- load_models: Загружает модель CatBoost.
- batch_load_sql: Выполняет загрузку данных из SQL базы частями.
- load_features: Загружает обработанные данные из базы.
- get_db: Создает сессию базы данных.
- get_recommended_feed: Формирует ленту рекомендаций постов.
- recommended_posts: API-эндпоинт для получения рекомендаций.

Классы:
- Post: Определяет структуру поста в базе данных.

Использование модуля требует наличия доступа к базе данных PostgreSQL и предобученной модели
CatBoostClassifier.

"""

import os
from typing import List
import pandas as pd
from catboost import CatBoostClassifier
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from fastapi import FastAPI, Depends
from schema import PostGet


app_1 = FastAPI()


def get_model_path(path: str) -> str:
    """
    Определяет путь к модели в зависимости от окружения.
    
    Args:
        path (str): Путь по умолчанию к модели, если код не выполняется в инфраструктуре курса.
    
    Returns:
        str: Корректный путь к модели.
    """

    if os.environ.get("IS_LMS") == "1":
        model_path = '/workdir/user_input/model'
    else:
        model_path = path
    return model_path

def load_models():
    """
    Загружает модель CatBoostClassifier.
    
    Returns:
        CatBoostClassifier: Объект загруженной модели.
    """
    model_path = get_model_path("/Users/your_model_name")
    from_file = CatBoostClassifier()
    catboost_ = from_file.load_model(model_path)

    return catboost_

def batch_load_sql(query: str) -> pd.DataFrame:
    """
    Выполняет SQL-запрос и загружает данные частями.
    
    Args:
        query (str): SQL-запрос для выполнения.
    
    Returns:
        pd.DataFrame: Датафрейм, содержащий результаты запроса.
    """
    chunksize = 200000
    engine_ = create_engine(
        "postgresql://login:password@"
        "path"
    )
    conn = engine_.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def load_features() -> pd.DataFrame:
    """
    Загружает обработанные данные из таблицы Postgres.
    
    Returns:
        pd.DataFrame: Датафрейм с загруженными данными.
    """
    return batch_load_sql('SELECT * FROM danissimos_features')


SQLALCHEMY_DATABASE_URL = "postgresql://login:password@path"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """
    Создает и возвращает сессию базы данных для последующих операций.
    
    Returns:
        Session: Объект сессии базы данных.
    """
    with SessionLocal() as db:
        return db


features = load_features()
model = load_models()

class Post(Base):
    " Класс для определния постов "
    __tablename__ = 'post'
    id = Column(Integer, primary_key=True)
    text = Column(String)
    topic = Column(String)

app = FastAPI()

def get_recommended_feed(
		user_id: int,
		limit: int = 5,
        db: Session = Depends(get_db)
) -> List[PostGet]:
    """
    Генерирует рекомендованный ленту постов для пользователя.
    
    Args:
        id (int): Идентификатор пользователя.
        limit (int): Количество рекомендуемых постов.
        db (Session): Сессия базы данных.
    
    Returns:
        List[PostGet]: Список рекомендованных постов.
    """

    # выбарем фичи юзера
    user_data = features[features['user_id'] == user_id]

    # опередялем колонки для выбора
    columns = ['business', 'covid', 'entertainment', 'movie', 'politics', 'sport', 'tech']

    # топ 3 темы по лайкам
    top_columns = user_data[columns].sum().nlargest(3).index.tolist()

    top_columns_df = features[features['topic'].isin(top_columns)]
    top_columns_df_100 = top_columns_df.head(100)
    top_columns_df_100 = top_columns_df_100[['post_id', 'pca1', 'pca2', 'topic']]

    # Создаем копию таблицы user_data и дублируем ее
    user_data_100 = user_data.drop(['post_id', 'pca1', 'pca2', 'topic'], axis=1)
    user_data_100 = pd.concat([user_data_100] * 100, ignore_index=True)

    # Проверяем количество строк в новой таблице
    new = pd.concat([user_data_100, top_columns_df_100], axis=1, join='inner')
    col_to_move = new.pop('post_id')  # Извлекаем столбец 'post_id' из датафрейма
    new.insert(2, 'post_id', col_to_move)  # Вставляем 'post_id' позицию 2

    new['proba'] = model.predict_proba(new)[:, 1]

    preds_id = new.sort_values(by='proba', ascending=False)['post_id'][:limit].values.tolist()

    preds = []

    for i in preds_id:
        preds.append(db.query(Post).filter(Post.id == i).one_or_none())
    return preds

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(user_id: int, limit: int=10, db: Session = Depends(get_db)) -> List[PostGet]:
    """
    API-эндпоинт для получения рекомендаций постов.
    
    Args:
        id (int): Идентификатор пользователя.
        limit (int): Максимальное количество возвращаемых постов.
        db (Session): Сессия базы данных.
    
    Returns:
        List[PostGet]: Список рекомендованных постов.
    """
    return get_recommended_feed(user_id, limit, db)
