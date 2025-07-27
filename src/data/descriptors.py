# src/data/build_features.py

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from mordred import Calculator, descriptors
from tqdm import tqdm


def calculate_rdkit_descriptors(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """
    Вычисляет ~200 дескрипторов RDKit для каждой SMILES-строки в DataFrame.

    Args:
        df (pd.DataFrame): DataFrame, содержащий колонку со SMILES.
        smiles_col (str): Название колонки со SMILES.

    Returns:
        pd.DataFrame: Исходный DataFrame, объединенный с новыми дескрипторами.
    """
    mols = [Chem.MolFromSmiles(smi) for smi in tqdm(df[smiles_col], desc="Конвертация SMILES в Mol")]
    desc_list = [Descriptors.CalcMolDescriptors(mol) for mol in tqdm(mols, desc="Генерация дескрипторов RDKit")]
    df_desc = pd.DataFrame(desc_list, index=df.index)

    return pd.concat([df, df_desc], axis=1)


def calculate_mordred_descriptors(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """
    Вычисляет 1600+ дескрипторов Mordred для каждой SMILES-строки.

    Args:
        df (pd.DataFrame): DataFrame, содержащий колонку со SMILES.
        smiles_col (str): Название колонки со SMILES.

    Returns:
        pd.DataFrame: Исходный DataFrame, объединенный с дескрипторами Mordred.
    """

    # Игнорируем предупреждения, которые может выдавать Mordred
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered in reduce')
    
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(s) for s in df[smiles_col]]
    
    # calc.pandas() автоматически показывает прогресс
    df_mordred = calc.pandas(mols).set_index(df.index)

    return pd.concat([df, df_mordred], axis=1)


def calculate_morgan_fingerprints(df: pd.DataFrame, smiles_col: str, radius: int, n_bits: int) -> pd.DataFrame:
    """
    Генерирует фингерпринты Morgan для каждой SMILES-строки.

    Args:
        df (pd.DataFrame): DataFrame, содержащий колонку со SMILES.
        smiles_col (str): Название колонки со SMILES.
        radius (int): Радиус фингерпринта.
        n_bits (int): Размер битового вектора.

    Returns:
        pd.DataFrame: Исходный DataFrame, объединенный с битами фингерпринтов.
    """

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mols = [Chem.MolFromSmiles(s) for s in tqdm(df[smiles_col], desc="Конвертация SMILES в Mol")]
    
    fps = [list(mfpgen.GetFingerprint(mol).ToBitString()) for mol in tqdm(mols, desc="Генерация фингерпринтов")]
    
    # Преобразуем битовые векторы в DataFrame
    df_fp = pd.DataFrame(np.array(fps), index=df.index, columns=[f'bit_{i}' for i in range(n_bits)])

    return pd.concat([df, df_fp], axis=1)


def clean_features(df: pd.DataFrame, smiles_col: str, target_col: str, corr_threshold: float = 0.7) -> pd.DataFrame:
    """
    Производит очистку датасета с дескрипторами:
    1. Приводит все признаки к числовому типу, удаляет строки с NaN.
    2. Удаляет признаки с нулевой дисперсией.
    3. Удаляет высококоррелированные признаки.
    
    Args:
        df (pd.DataFrame): Входной DataFrame с признаками.
        smiles_col (str): Название колонки со SMILES.
        target_col (str): Название целевой колонки.
        corr_threshold (float): Порог для удаления коррелирующих признаков.
        
    Returns:
        pd.DataFrame: Очищенный DataFrame.
    """

    print(f"Очистка набора признаков ({df.shape[1]} признаков)...")
    
    feature_cols = [col for col in df.columns if col not in [smiles_col, target_col]]
    
    # 1. Приведение к числам и удаление NaN
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    
    # 2. Удаление колонок с нулевой дисперсией
    variances = df[feature_cols].var()
    zero_var_columns = variances[variances == 0].index
    df.drop(columns=zero_var_columns, inplace=True)
    
    # 3. Удаление высококоррелированных признаков
    feature_cols_after_var = [col for col in df.columns if col not in [smiles_col, target_col]]
    corr_matrix = df[feature_cols_after_var].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_columns = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    df.drop(columns=high_corr_columns, inplace=True)
    
    print(f"✓ Набор признаков очищен ({df.shape[1]} признаков)")
    return df