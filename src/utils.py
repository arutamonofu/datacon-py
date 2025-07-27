# src/utils.py

from pathlib import Path
import yaml

def load_config(config_path: Path) -> dict:
    print(f"Загрузка конфигурации из {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("✓ Конфигурация успешно загружена.")
    return config