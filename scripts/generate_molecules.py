import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from utils import load_config

def main():
    # ====== 0. КОНФИГУРАЦИЯ ======

    config_path = PROJECT_ROOT / "configs" / "main_config.yaml"
    config = load_config(config_path)
    data_config = config["data"]
    generated_dir = PROJECT_ROOT / data_config['generated_dir']
    generated_path = generated_dir / "sampling.csv"

    # ====== 1. ГЕНЕРАЦИЯ ======
    print(f"""
        **********************************************************
        Необходимо выполнить ручное действие!
        1. Установите REINVENT4.
        2. Выполните блокнот REINVENT.ipynb
        3. Проверьте полученный csv в: {generated_path}
        **********************************************************
    """)

    while not generated_path.is_file():
        input(">>> Нажмите Enter, когда файл будет сохранен, чтобы продолжить...")
    
        if not generated_path.is_file():
            print(f"\n❌ Файл не найден по пути '{generated_path.resolve()}'")
            print("Пожалуйста, выполните инструкцию и нажмите Enter.\n")

if __name__ == '__main__':
    main()