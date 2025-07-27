# scripts/02_train_model.py

from pathlib import Path
import sys
import pandas as pd
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from utils import load_config
from predictor import train_model

def main():
    # ====== 0. КОНФИГУРАЦИЯ ======

    # Файл конфигурации
    config_path = PROJECT_ROOT / "configs" / "main_config.yaml"
    config = load_config(config_path)

    # Данные
    data_cfg = config["data"]
    smiles_col = data_cfg["smiles_col"]
    target_col = data_cfg["target_col"]
    target_id = data_cfg["chembl_target_id"]

    # Модель
    model_type = config["model_type"]

    if model_type == "xgb":
        xgb_train_cfg = config["training_xgb"]

        # Данные
        feature_type = xgb_train_cfg["features"]["feature_type"]
        data_path = PROJECT_ROOT / data_cfg["features_dir"] / f"{target_id}_{feature_type}.csv"

        # Пути сохранения
        model_dir = PROJECT_ROOT / xgb_train_cfg['model_output_dir']
        model_output_path = model_dir / "model.joblib"
        columns_output_path = model_dir / "columns.json"

        # Создание папок, если нет
        model_dir.mkdir(parents=True, exist_ok=True)

    elif model_type == "gnn":
        gnn_train_cfg = config['training_gnn']

        # Данные
        data_path = PROJECT_ROOT / data_cfg['processed_dir'] / f"all_processed.csv"
        graphs_dir = PROJECT_ROOT / data_cfg['graphs_dir_gnn']

        # Пути сохранения 
        model_dir = PROJECT_ROOT / gnn_train_cfg['model_output_dir']
        model_output_path = model_dir / "model.pth"
        graphs_dir = PROJECT_ROOT / data_cfg['graphs_dir_gnn']

        # Создание папок, если нет
        model_dir.mkdir(parents=True, exist_ok=True)
        graphs_dir.mkdir(parents=True, exist_ok=True)


    # ====== 1. ОБУЧЕНИЕ ======
    
    train_model(
        data_path=data_path,
        model_output_path=model_output_path,
        smiles_col=smiles_col,
        target_col=target_col,
        test_size=xgb_train_cfg['test_size'],
        random_state=config['random_state'],
        n_trials=xgb_train_cfg['n_trials_optuna']
    )
    print(f"✓ Модель обучена и сохранена в: {model_output_path}")

    # Сохранение списка колонок для будущих предсказаний
    xgb_columns = pd.read_csv(data_path) \
        .drop(columns=[smiles_col, target_col]) \
        .columns \
        .to_list()
    with open(columns_output_path, 'w') as f:
        json.dump(xgb_columns, f, indent=4)
    print(f"✓ Тренировочные колонки сохранены в: {columns_output_path}")

if __name__ == '__main__':
    main()