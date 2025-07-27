# scripts/01_prepare_dataset.py

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from utils import load_config
from data.download import (
    download_chembl_data,
    download_pubchem_data
)
from data.preprocess import (
    preprocess_chembl_data,
    preprocess_pubchem_data,
    combine_datasets
)
from data.descriptors import (
    calculate_rdkit_descriptors,
    calculate_mordred_descriptors,
    calculate_morgan_fingerprints,
    clean_features
)

def main():
    # ====== 0. КОНФИГУРАЦИЯ ======

    config_path = PROJECT_ROOT / "configs" / "main_config.yaml"
    config = load_config(config_path)

    # Данные
    data_config = config["data"]
    target_col = data_config["target_col"]
    smiles_col = data_config["smiles_col"]
    chembl_target_id = data_config["chembl_target_id"]
    pubchem_accession = data_config["pubchem_accession"]

    # Дескрипторы
    features_config = config["training"]["features"]
    feature_type = features_config["feature_type"]

    # Папки данных
    raw_dir = PROJECT_ROOT / data_config['raw_dir']
    processed_dir = PROJECT_ROOT / data_config['processed_dir']
    features_dir = PROJECT_ROOT / data_config['features_dir']

    # Создание папок, если нет
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)


    # ====== 1. СКАЧИВАНИЕ ======

    # Определение путей для сохранения
    chembl_raw_path = raw_dir / f"chembl_{chembl_target_id}_raw.csv"
    pubchem_raw_path = raw_dir / f"pubchem_{pubchem_accession}_raw.csv"

    # Скачивание
    print("Загрузка с ChEMBL...")
    download_chembl_data(
        target_chembl_id=data_config['chembl_target_id'],
        output_path=chembl_raw_path
    )
    print(f"✓ Сохранено в: {chembl_raw_path}")

    print("Загрузка с PubChem...")
    download_pubchem_data(
        target_accession=data_config['pubchem_accession'],
        output_path=pubchem_raw_path
    )
    print(f"✓ Сохранено в: {pubchem_raw_path}")


    # === 2. ПРЕДОБРАБОТКА ===

    # Определение путей для сохранения
    chembl_processed_path = processed_dir / f"chembl_{chembl_target_id}_processed.csv"
    pubchem_processed_path = processed_dir / f"pubchem_{pubchem_accession}_processed.csv"
    all_processed_path = processed_dir / f"all_processed.csv"

    # Предобработка
    print("Предобработка данных с ChEMBL...")
    preprocess_chembl_data(
        raw_path=chembl_raw_path,
        output_path=chembl_processed_path
    )
    print(f"✓ Сохранено в: {chembl_processed_path}")

    print("Предобработка данных с PubChem...")
    preprocess_pubchem_data(
        raw_path=pubchem_raw_path,
        output_path=pubchem_processed_path
    )
    print(f"✓ Сохранено в: {pubchem_processed_path}")

    print("Объединение датасетов...")
    combine_datasets(
        chembl_path=chembl_processed_path,
        pubchem_path=pubchem_processed_path,
        output_path=all_processed_path
    )
    print(f"✓ Сохранено в: {all_processed_path}")
    

    # ====== 3. ДЕСКРИПТОРЫ ======
        
    # Загрузка предобработанного файла
    df = pd.read_csv(all_processed_path)
    df_features = None
    
    # Генерация дескрипторов
    if feature_type == 'rdkit':
        df_features = calculate_rdkit_descriptors(df, smiles_col=smiles_col)
    elif feature_type == 'mordred':
        df_features = calculate_mordred_descriptors(df, smiles_col=smiles_col)
    elif feature_type == 'morgan':
        morgan_cfg = features_config['morgan']
        df_features = calculate_morgan_fingerprints(
            df=df, 
            smiles_col=smiles_col,
            radius=morgan_cfg['radius'],
            n_bits=morgan_cfg['n_bits']
        )

    # Очистка дескрипторов
    df_final = clean_features(
        df=df_features,
        smiles_col=smiles_col,
        target_col=target_col,
        corr_threshold=features_config['corr_threshold']
    )

    # Сохранение
    final_features_path = features_dir / f"{chembl_target_id}_{feature_type}.csv"
    df_final.to_csv(final_features_path, index=False)

if __name__ == '__main__':
    main()