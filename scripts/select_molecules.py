# scripts/03_filter_and_predict.py

import sys
from pathlib import Path
import pandas as pd
import json
from torch_geometric.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from utils import load_config
from data.score import (
    calculate_scoring_features,
    apply_programmatic_filters,
    prepare_for_swissadme,
    apply_swissadme_filters
)
from data.descriptors import (
    calculate_morgan_fingerprints,
    calculate_rdkit_descriptors,
    calculate_mordred_descriptors
)
from predictor import (
    load_pipeline,
    predict
)

def main():
    # ====== 0. КОНФИГУРАЦИЯ ======

    config_path = PROJECT_ROOT / "configs" / "main_config.yaml"
    config = load_config(config_path)

    # Данные
    data_cfg = config["data"]
    smiles_col = data_cfg['smiles_col']
    target_col = data_cfg['target_col']
    generated_path = PROJECT_ROOT / data_cfg["generated_dir"] / "sampling.csv"
    swissadme_path = PROJECT_ROOT / data_cfg["selected_dir"] / "swissadme.csv"

    # Отбор
    selection_cfg = config["selection"]
    pic50_threshold = selection_cfg["predicted_pIC50"]
    
    predictor_cfg = config["training"]
    model_dir = PROJECT_ROOT / predictor_cfg['model_output_dir']
    feature_type = predictor_cfg['features']['feature_type']
 
    # Пути сохранения
    output_dir = PROJECT_ROOT / data_cfg["selected_dir"]

    # Создание папок, если нет
    output_dir.mkdir(parents=True, exist_ok=True)


    # ====== 1. ПРОГРАММНАЯ ФИЛЬТРАЦИЯ ======

    # Загрузка сгенерированных молекул  
    df_generated = pd.read_csv(generated_path)

    if smiles_col not in df_generated.columns and 'SMILES' in df_generated.columns:
        df_generated.rename(columns={'SMILES': smiles_col}, inplace=True)
    
    # Расчёт фильтруемых признаков
    df_with_props = calculate_scoring_features(df_generated, smiles_col=smiles_col)

    # Применение фильтра
    df_program_filtered = apply_programmatic_filters(
        df=df_with_props,
        filters=selection_cfg["programmatic"]
    )
    
    # Сохранение
    program_filtered_path = output_dir / "01_program_filtered.csv"
    df_program_filtered.to_csv(program_filtered_path, index=False)

    if df_program_filtered.empty:
        print("После программной фильтрации не осталось молекул. Завершение.")
        sys.exit(0)


    # ====== 2. ПРЕДСКАЗАНИЕ И ОТБОР ПО pIC50 ======

    # Загрузка модели
    model = load_pipeline(model_dir / "model.joblib")

    # Загрузка перечня признаков модели
    with open(model_dir / "columns.json", 'r') as f:
        xgb_columns = json.load(f)
    
    # Расчёт признаков под модель
    df_with_features = df_program_filtered[[smiles_col]].copy()
    if feature_type == 'rdkit':
        df_with_features = calculate_rdkit_descriptors(df_with_features, smiles_col=smiles_col)
    elif feature_type == 'mordred':
        df_with_features = calculate_mordred_descriptors(df_with_features, smiles_col=smiles_col)
    elif feature_type == 'morgan':
        morgan_cfg = predictor_cfg['features']['morgan']
        df_with_features = calculate_morgan_fingerprints(
            df_with_features,
            smiles_col,
            morgan_cfg['radius'],
            morgan_cfg['n_bits']
        )
    
    # Предсказание для тренировочных признаков
    predictions = predict(
        model=model,
        data_to_predict=df_with_features,
        training_columns=xgb_columns,
        smiles_col=smiles_col)
    
    # Объединение предсказаний с датасетом
    df_with_predictions = pd.merge(
        df_program_filtered,
        predictions, 
        on=smiles_col,
        how='inner'
    )

    # Фильтрация
    df_pic50_filtered = df_with_predictions.query(f"predicted_pIC50 {pic50_threshold}")
    df_pic50_filtered_sorted = df_pic50_filtered.sort_values(
        by='predicted_pIC50', 
        ascending=False
    ).reset_index(drop=True)
    
    pic50_filtered_path = output_dir / "02_pic50_filtered.csv"
    df_pic50_filtered_sorted.to_csv(pic50_filtered_path, index=False)


    # ====== 3. SWISSADME ======

    if df_pic50_filtered_sorted.empty:
        print("Не найдено молекул с высокой предсказанной активностью. Завершение.")
        sys.exit(0)

    # Подготовка входного файла для SwissADME    
    swissadme_input_path = output_dir / "03_for_swissadme.txt"
    prepare_for_swissadme(
        df_pic50_filtered_sorted,
        smiles_col=smiles_col,
        output_path=swissadme_input_path
    )

    print(f"""
        **********************************************************
        Необходимо выполнить ручное действие!
        1. Перейдите в сервис SwissADME: http://www.swissadme.ch/
        2. Вставьте в него содержимое файла {swissadme_input_path.resolve()}.
        3. Сохраните полученный csv в: {swissadme_path.resolve()}
        **********************************************************
    """)

    while not swissadme_path.is_file():
        input(">>> Нажмите Enter, когда файл будет сохранен, чтобы продолжить...")
        
        if not swissadme_path.is_file():
            print(f"\n❌ Файл не найден по пути '{swissadme_path.resolve()}'")
            print("Пожалуйста, выполните инструкцию и нажмите Enter.\n")
    
    final_hits_path = output_dir / "04_final_ranked_hits.csv"

    df_final_hits = apply_swissadme_filters(
        filtered_df=df_pic50_filtered_sorted,
        smiles_col=smiles_col,
        swiss_results_path=swissadme_path,
        output_path=final_hits_path
    )
    
    if not df_final_hits.empty:
        print(f"✓ Отобранные молекулы ({len(df_final_hits)}) сохранены в: {final_hits_path}")
    else:
        print("К сожалению, ни одна из активных молекул не прошла финальную проверку SwissADME.")

if __name__ == '__main__':
    main()