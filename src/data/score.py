# src/data/make_dataset.py

import pandas as pd
import numpy as np
from rdkit import Chem, rdBase
from pathlib import Path

from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.rdMolDescriptors import CalcNumHBD, CalcNumHBA, CalcTPSA
from tqdm import tqdm

# Импорт SA_Score
from rdkit.Chem import RDConfig
import os.path as op
import sys
sys.path.append(op.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer # type: ignore

# Подавление логов RDKit, чтобы не засорять вывод
rdBase.DisableLog('rdApp.error')

def calculate_scoring_features(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """
    Рассчитывает набор хемоинформатических свойств для DataFrame с молекулами.
    Использует tqdm для отображения прогресс-бара.
    
    Args:
        df (pd.DataFrame): Входной DataFrame.
        smiles_col (str): Название колонки со SMILES.

    Returns:
        pd.DataFrame: DataFrame с добавленными колонками свойств.
    """
    # Списки для хранения вычисленных свойств
    qeds, sa_scores, lipinski_violations_list = [], [], []
    has_pains_list, has_brenk_list, bbb_predictions = [], [], []
    
    # Оборачиваем итератор в tqdm для отображения прогресс-бара
    for smiles in tqdm(df[smiles_col], desc="Расчет физико-химических свойств"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Невалидный SMILES")

            # --- Базовые дескрипторы (вычисляются один раз) ---
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = CalcTPSA(mol)
            hbd = CalcNumHBD(mol)
            hba = CalcNumHBA(mol)

            # --- 1. QED (Количественная оценка схожести с лекарством) ---
            qeds.append(Descriptors.qed(mol))

            # --- 2. Нарушения правила "пяти" Липински ---
            violations = 0
            if mw > 500: violations += 1
            if logp > 5: violations += 1
            if hbd > 5: violations += 1
            if hba > 10: violations += 1
            lipinski_violations_list.append(violations)
            
            # --- 3. Прогноз проходимости через ГЭБ ---
            bbb_ok = (mw < 500) and (1 < logp < 3) and (tpsa < 90) and (hbd <= 5) and (hba <= 10)
            bbb_predictions.append(bbb_ok)

            # --- 4. Проверка на наличие PAINS и Brenk структур ---
            has_pains_list.append(has_pains_alert(mol))
            has_brenk_list.append(has_brenk_alert(mol))

            # --- 5. Оценка синтетической доступности (SA Score) ---
            try:
                complexity = (mw / 100 +
                             Descriptors.NumRotatableBonds(mol) / 5 +
                             len(mol.GetSubstructMatches(Chem.MolFromSmarts('[!$(*#*)&!D1]-!@[!$(*#*)&!D1]'))) / 10)
                rings = Descriptors.RingCount(mol)
                sa_score = max(1.0, min(6.0, 1.0 + complexity + rings * 0.5))
                sa_scores.append(sa_score)
            except:
                sa_scores.append(4.0) # Среднее значение в случае ошибки расчета

        except Exception:
            # В случае ошибки (невалидный SMILES) добавляем "плохие" значения
            qeds.append(np.nan)
            sa_scores.append(np.nan)
            lipinski_violations_list.append(5) # Максимальное значение нарушений
            has_pains_list.append(True)      # Считаем, что проблемы есть
            has_brenk_list.append(True)      # Считаем, что проблемы есть
            bbb_predictions.append(False)    # Считаем, что не проходит

    # Присваиваем новые колонки DataFrame'у
    df['qed'] = qeds
    df['sa_score'] = sa_scores
    df['lipinski_violations'] = lipinski_violations_list
    df['has_pains'] = has_pains_list
    df['has_brenk'] = has_brenk_list
    df['predict_bbb'] = bbb_predictions
    
    # Удаляем строки, где не удалось рассчитать ключевые метрики (qed или sa_score)
    df.dropna(subset=['qed', 'sa_score'], inplace=True)
    
    return df

def has_pains_alert(mol: Chem.Mol) -> bool:
    """Проверяет наличие PAINS-структур (токсикофоры)."""
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)
    return catalog.HasMatch(mol)

def has_brenk_alert(mol: Chem.Mol) -> bool:
    """Проверяет наличие Brenk-структур (нежелательные фрагменты)."""
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog(params)
    return catalog.HasMatch(mol)

def apply_programmatic_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Применяет набор фильтров к DataFrame на основе словаря с критериями.

    Args:
        df (pd.DataFrame): DataFrame с рассчитанными свойствами.
        filters (dict): Словарь с критериями фильтрации. 
                        Пример: {'qed': '> 0.7', 'sa_score': '< 6.0'}

    Returns:
        pd.DataFrame: Отфильтрованный DataFrame.
    """
    query_parts = []
    for key, condition in filters.items():
        query_parts.append(f"`{key}` {condition}") 
    query = " & ".join(query_parts)
    
    filtered_df = df.query(query).reset_index(drop=True)
    return filtered_df

def prepare_for_swissadme(df: pd.DataFrame, smiles_col: str, output_path: Path):
    """Сохраняет SMILES в текстовый файл для загрузки в SwissADME."""
    df[smiles_col].to_csv(output_path, index=False, header=False)

def apply_swissadme_filters(
    filtered_df: pd.DataFrame, 
    smiles_col: str, 
    swiss_results_path: Path, 
    output_path: Path
):
    """
    Финальная фильтрация на основе CSV-файла, полученного из SwissADME.

    Args:
        filtered_df (pd.DataFrame): DataFrame после первичной программной фильтрации.
        smiles_col (str): Название колонки со SMILES.
        swiss_results_path (Path): Путь к CSV файлу из SwissADME.
        output_path (Path): Путь для сохранения финального списка хитов.
    """
    df_swiss = pd.read_csv(swiss_results_path, index_col=0)
    
    # SwissADME может изменить каноническое представление SMILES
    # Переименовываем колонку для объединения
    df_swiss.rename(columns={'Canonical SMILES': smiles_col}, inplace=True)
    
    # Фильтруем результаты SwissADME по ключевым параметрам
    swiss_mask = (
        (df_swiss['BBB permeant'] == 'Yes') &
        (df_swiss['GI absorption'] == 'High') &
        (df_swiss['Lipinski #violations'] == 0) &
        (df_swiss['Brenk #alerts'] == 0) &
        (df_swiss['PAINS #alerts'] == 0) &
        (df_swiss['Synthetic Accessibility'] < 6.0)
    )
    final_hits = df_swiss[swiss_mask].reset_index()
    
    final_hits.to_csv(output_path, index=False)
    return final_hits