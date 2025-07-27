# src/predictor/xgb.py

import pandas as pd
import numpy as np
import joblib
import optuna
from pathlib import Path
from typing import List

# Scikit-learn & XGBoost
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Отключаем лишние логи Optuna, оставляя только самое важное
optuna.logging.set_verbosity(optuna.logging.WARNING)

def create_pipeline(random_state: int) -> Pipeline:
    """
    Создает и возвращает пайплайн для предварительной обработки данных и обучения модели.
    Пайплайн включает в себя:
    1. StandardScaler: Стандартизация признаков.
    2. PCA: Уменьшение размерности с сохранением 95% дисперсии.
    3. XGBRegressor: Модель градиентного бустинга.
    
    Args:
        random_state (int): Состояние для воспроизводимости PCA и регрессора.
        
    Returns:
        Pipeline: Готовый пайплайн scikit-learn.
    """

    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95, random_state=random_state)),
        ('regressor', XGBRegressor(seed=random_state, n_jobs=1))
    ])

def run_hyperparameter_optimization(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int, random_state: int) -> dict:
    """
    Запускает оптимизацию гиперпараметров для XGBoost регрессора в пайплайне с помощью Optuna.

    Args:
        pipeline (Pipeline): Сконфигурированный пайплайн.
        X_train (pd.DataFrame): Обучающие признаки.
        y_train (pd.Series): Целевая переменная для обучения.
        n_trials (int): Количество итераций для поиска Optuna.
        random_state (int): Состояние для воспроизводимости KFold.

    Returns:
        dict: Словарь с лучшими найденными гиперпараметрами.
    """

    print(f"Оптимизация гиперпараметров...")

    def objective(trial: optuna.Trial) -> float:
        """Целевая функция для Optuna."""
        # Определяем пространство поиска гиперпараметров
        params = {
            'regressor__n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=100),
            'regressor__learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'regressor__max_depth': trial.suggest_int('max_depth', 3, 12),
            'regressor__subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'regressor__colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'regressor__min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'regressor__reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'regressor__reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'regressor__gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True)
        }
        pipeline.set_params(**params)
        
        # Используем кросс-валидацию для оценки
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = cross_val_score(pipeline, X_train, y_train, scoring='neg_mean_squared_error', cv=kf, n_jobs=-1)
        
        rmse = np.sqrt(-scores.mean())
        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Лучшие параметры: {study.best_params}")
    print(f"Лучший RMSE: {study.best_value:.4f}")
    
    print("✓ Оптимизация завершена")
    
    return study.best_params

def train_model(
    data_path: Path, 
    model_output_path: Path, 
    smiles_col: str, 
    target_col: str, 
    test_size: float, 
    random_state: int, 
    n_trials: int
):
    """
    Полный цикл обучения модели: загрузка данных, разделение, оптимизация гиперпараметров,
    обучение финальной модели, оценка и сохранение.

    Args:
        data_path (Path): Путь к датасету с признаками.
        model_output_path (Path): Путь для сохранения обученной модели.
        smiles_col (str): Название колонки со SMILES.
        target_col (str): Название целевой колонки.
        test_size (float): Доля данных для тестового набора.
        random_state (int): Состояние для воспроизводимости.
        n_trials (int): Количество итераций для Optuna.
    """

    # Загрузка и подготовка данных
    df = pd.read_csv(data_path)
    X = df.drop(columns=[smiles_col, target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Создание пайплайна и оптимизация гиперпараметров
    pipeline = create_pipeline(random_state)
    best_params = run_hyperparameter_optimization(pipeline, X_train, y_train, n_trials, random_state)
    
    # Обучение финальной модели с лучшими параметрами
    print("Обучение финальной модели...")
    final_pipeline = create_pipeline(random_state)
    regressor_params = {'regressor__' + key: value for key, value in best_params.items()}
    final_pipeline.set_params(**regressor_params)
    final_pipeline.fit(X_train, y_train)

    # Оценка модели на тестовом наборе
    final_preds = final_pipeline.predict(X_test)
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, final_preds)):.4f}")
    print(f"R²:   {r2_score(y_test, final_preds):.4f}")
    
    # Сохранение модели
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_output_path)

def load_pipeline(model_path: str) -> Pipeline:
    """
    Загружает обученный пайплайн модели из файла.

    Args:
        model_path (str | Path): Путь к файлу модели (*.joblib).

    Returns:
        Pipeline: Загруженный пайплайн scikit-learn.
    
    Raises:
        FileNotFoundError: Если файл модели не найден.
        Exception: При других ошибках загрузки.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Файл модели не найден по пути {model_path}")
    try:
        model = joblib.load(model_path)
        print(f"✓ Модель успешно загружена из {model_path}")
        return model
    except Exception as e:
        print(f"X Произошла ошибка при загрузке модели: {e}")
        raise

def predict(model: Pipeline, data_to_predict: pd.DataFrame, training_columns: List[str], smiles_col: str) -> pd.DataFrame:
    """
    Выполняет предсказания pIC50 для новых данных, используя обученную модель.
    
    Эта функция критически важна, так как она синхронизирует колонки новых
    данных с теми, что использовались при обучении.

    Args:
        model (Pipeline): Обученный пайплайн.
        data_to_predict (pd.DataFrame): DataFrame с новыми данными (включая SMILES и признаки).
        training_columns (List[str]): Список названий колонок, на которых обучалась модель.
        smiles_col (str): Название колонки со SMILES.

    Returns:
        pd.DataFrame: DataFrame с колонками SMILES и predicted_pIC50.
    """
    if model is None:
        raise ValueError("Модель не была загружена. Предсказание невозможно.")
        
    print(f"Исходное количество молекул для предсказания: {len(data_to_predict)}")

    # 1. Получаем имена и порядок признаков непосредственно из объекта модели.
    # Это самый надежный источник информации о том, на каких данных обучалась модель.
    try:
        model_feature_names = model.steps[0][1].feature_names_in_
    except (AttributeError, IndexError):
        # Если структура пайплайна отличается, выводим ошибку
        raise ValueError("Не удалось извлечь имена признаков из модели. Убедитесь, что первым шагом в пайплайне является обученный трансформер (например, StandardScaler).")

    # 2. Убедимся, что все нужные колонки есть в данных для предсказания
    missing_cols = set(model_feature_names) - set(data_to_predict.columns)
    if missing_cols:
        raise ValueError(f"В данных для предсказания отсутствуют следующие колонки: {missing_cols}")
        
    # 3. Синхронизируем порядок и набор колонок, используя список из модели
    features_df = data_to_predict[model_feature_names].copy()
    
    # 4. Обработка типов и пропусков
    features_df = features_df.apply(pd.to_numeric, errors='coerce')
    original_index = features_df.index
    features_df.dropna(inplace=True)
    cleaned_index = features_df.index
    
    num_dropped = len(original_index) - len(cleaned_index)
    if num_dropped > 0:
        print(f"Удалено {num_dropped} молекул из-за пропусков в признаках.")
    
    if features_df.empty:
        print("После очистки не осталось данных для предсказания.")
        return pd.DataFrame(columns=[smiles_col, 'predicted_pIC50'])
        
    # 5. Предсказание
    predictions = model.predict(features_df)
    
    # 6. Формирование итогового DataFrame
    # Используем .loc с сохраненным индексом, чтобы сопоставить SMILES и предсказания
    result_df = pd.DataFrame({
        smiles_col: data_to_predict.loc[cleaned_index, smiles_col],
        'predicted_pIC50': predictions
    }).reset_index(drop=True)
    
    return result_df