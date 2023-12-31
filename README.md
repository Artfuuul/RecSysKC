# Рекомендательная Система на Основе CatBoost

## Описание
Этот проект включает реализацию API для рекомендательной системы, использующей CatBoostClassifier для предсказания интересов пользователя. Он охватывает загрузку модели, обработку данных, взаимодействие с базой данных PostgreSQL и формирование рекомендательной ленты постов.

## Содержание
- `model_training.ipynb`: Jupyter Notebook с кодом для тренировки модели CatBoost.
- `main.py`: Основной файл Python с реализацией API и вспомогательными функциями.
- `requirements.txt`: Список зависимостей Python.
- `README.md`: Документация проекта (этот файл).

## Установка и Запуск
Для запуска проекта выполните следующие шаги:
1. Клонируйте репозиторий.
2. Установите необходимые зависимости, используя `pip install -r requirements.txt`.
3. Запустите Jupyter Notebook `model_training.ipynb` для тренировки модели.
4. Запустите `main.py` для запуска API.

## Использование API
Для получения рекомендаций используйте эндпоинт `/post/recommendations/` с параметрами `id` (идентификатор пользователя) и `limit` (количество постов).

## Требования
- Python 3.x
- CatBoost
- Pandas
- SQLAlchemy
- FastAPI
