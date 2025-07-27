# src/data/preprocess.py

from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

def is_valid_smiles(smiles: str) -> bool:
    """Проверяет, является ли строка SMILES валидной."""
    if not isinstance(smiles, str) or pd.isna(smiles):
        return False
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except (ValueError, Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException):
        return False

def preprocess_chembl_data(raw_path: str | Path, output_path: str | Path):
    """
    Очищает и преобразует сырые данные из ChEMBL.
    - Фильтрует по типу активности IC50.
    - Вычисляет pIC50.
    - Валидирует SMILES.
    - Удаляет дубликаты.
    """
    # Загрузка файла
    df = pd.read_csv(raw_path)

    # 1. Фильтрация IC50
    df = df[df['standard_type'] == 'IC50'].copy()
    df = df.dropna(subset=['standard_value', 'canonical_smiles'])

    # 2. Валидация SMILES
    df['is_valid'] = df['canonical_smiles'].apply(is_valid_smiles)
    df = df[df['is_valid']].drop(columns=['is_valid'])
    
    # 3. Преобразование в pIC50
    sv_numeric_nm = pd.to_numeric(df['standard_value'], errors='coerce')
    sv_molar = sv_numeric_nm * 1e-9
    safe_sv_molar = sv_molar.where(sv_molar > 0, 1e-12)
    pIC50_calculated = -np.log10(safe_sv_molar)
    df['pIC50'] = df['pchembl_value'].fillna(pIC50_calculated)

    # 4. Удаление дубликатов по SMILES, оставляем запись с наибольшей активностью
    df_clean = df[['canonical_smiles', 'pIC50']].copy()
    df_clean = df_clean.sort_values('pIC50', ascending=False).drop_duplicates('canonical_smiles')  

    # Сохранение
    df_clean.to_csv(output_path, index=False)


def preprocess_pubchem_data(raw_path: str | Path, output_path: str | Path):
    """
    Очищает и преобразует сырые данные из PubChem.
    - Фильтрует по типу активности IC50 и статусу 'Active'.
    - Вычисляет pIC50 из 'Activity Value [uM]'.
    - Валидирует SMILES.
    - Удаляет дубликаты.
    """
    # Загрузка файла
    df = pd.read_csv(raw_path, low_memory=False)

    # 1. Фильтрация IC50 и активности
    df = df[df['Activity Name'] == 'IC50'].copy()
    df = df[df['Activity Outcome'] == 'Active'].copy()
    
    # 2. Валидация SMILES
    df['is_valid'] = df['SMILES'].apply(is_valid_smiles)
    df = df[df['is_valid']].drop(columns=['is_valid'])

    # 3. Преобразование в pIC50
    df['Activity Value [uM]'] = pd.to_numeric(df['Activity Value [uM]'], errors='coerce')
    df = df.dropna(subset=['Activity Value [uM]', 'SMILES'])
    activity_uM = pd.to_numeric(df['Activity Value [uM]'], errors='coerce')
    safe_activity_uM = activity_uM.where(activity_uM > 0, 1e-6)
    df['pIC50'] = 6 - np.log10(safe_activity_uM)

    # 4. Удаление дубликатов
    df = df.rename(columns={'SMILES': 'canonical_smiles'})
    df_clean = df[['canonical_smiles', 'pIC50']].copy()
    df_clean = df_clean.sort_values('pIC50', ascending=False).drop_duplicates('canonical_smiles')

    # Сохранение
    df_clean.to_csv(output_path, index=False)


def combine_datasets(chembl_path: str | Path, pubchem_path: str | Path, output_path: str | Path):
    """
    Объединяет очищенные датасеты из ChEMBL и PubChem в один.
    """
    # Загрузка файлов PubChem и ChEMBL
    df_chembl = pd.read_csv(chembl_path)
    df_pubchem = pd.read_csv(pubchem_path)
    
    # Объединение
    df_combined = pd.concat([df_chembl, df_pubchem], ignore_index=True)
    
    # Удаление дубликатов
    df_combined = df_combined.sort_values('pIC50', ascending=False).drop_duplicates('canonical_smiles')
    
    # Сохранение
    df_combined.to_csv(output_path, index=False)