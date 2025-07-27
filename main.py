#!/usr/bin/env python
# main.py

"""
Основной интерфейс командной строки (CLI) для управления пайплайном проекта
по открытию лекарств.

Этот скрипт позволяет запускать отдельные этапы пайплайна (подготовка данных,
обучение модели, генерация и отбор молекул) или весь процесс целиком.
Выбор модели (XGBoost или GNN) и другие параметры настраиваются
в файле configs/main_config.yaml.

Примеры использования:
--------------------
# 1. Запустить только этап подготовки данных:
python main.py prepare-data

# 2. Запустить только этап обучения модели (тип модели берется из конфига):
python main.py prepare-predictor

# 3. Запустить только этап генерации:
python main.py generate-molecules

# 4. Запустить только этап отбора:
python main.py select-molecules

# 4. Запустить весь пайплайн последовательно:
python main.py run-all
--------------------
"""

import sys
import argparse
from pathlib import Path

# Добавляем корень проекта в sys.path, чтобы корректно импортировать
# модули из `src` и скрипты из `scripts`.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# Импортируем главные функции из каждого скрипта с псевдонимами для ясности
try:
    from scripts.prepare_data import main as prepare_data_main
    from scripts.prepare_predictor import main as prepare_predictor_main
    from scripts.generate_molecules import main as generate_molecules_main
    from scripts.select_molecules import main as select_molecules_main
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что вы запускаете этот скрипт из корневой директории проекта,")
    print("и что структура папок 'scripts' и 'src' верна.")
    sys.exit(1)


def setup_cli():
    """
    Настраивает и возвращает парсер аргументов командной строки.
    """
    parser = argparse.ArgumentParser(
        description="CLI для пайплайна проекта по открытию лекарств.",
        formatter_class=argparse.RawTextHelpFormatter  # для красивого отображения примеров
    )
    
    # Создаем субпарсеры для каждой команды
    subparsers = parser.add_subparsers(dest='command', required=True, help='Доступные команды')

    # Команда для подготовки данных
    subparsers.add_parser(
        'prepare-data', 
        help="Этап 1: Скачивание, обработка и создание набора данных."
    )

    # Команда для обучения модели
    subparsers.add_parser(
        'prepare-predictor', 
        help="Этап 2: Обучение предсказательной модели (тип определяется в config)."
    )

    # Команда для генерации и отбора
    subparsers.add_parser(
        'generate-molecules', 
        help="Этап 3: Генерация новых молекул."
    )

    # Команда для генерации и отбора
    subparsers.add_parser(
        'select-molecules', 
        help="Этап 4: Скоринг и отбор новых молекул-хитов."
    )

    # Команда для полного запуска
    subparsers.add_parser(
        'run-all', 
        help="Запустить все этапы пайплайна последовательно: prepare -> train -> generate."
    )

    return parser


def main():
    """
    Главная функция, которая парсит аргументы и запускает соответствующий пайплайн.
    """
    parser = setup_cli()
    args = parser.parse_args()

    if args.command == 'prepare-data':
        print("--- [ЗАПУСК] Этап 1: Подготовка набора данных ---")
        prepare_data_main()
        print("\n--- [ЗАВЕРШЕНО] Этап 1: Подготовка набора данных ---")
        
    elif args.command == 'prepare-predictor':
        print("--- [ЗАПУСК] Этап 2: Обучение модели ---")
        prepare_predictor_main()
        print("\n--- [ЗАВЕРШЕНО] Этап 2: Обучение модели ---")
        
    elif args.command == 'generate-molecules':
        print("--- [ЗАПУСК] Этап 3: Генерация и отбор молекул ---")
        generate_molecules_main()
        print("\n--- [ЗАВЕРШЕНО] Этап 3: Генерация и отбор молекул ---")

    elif args.command == 'select-molecules':
        print("--- [ЗАПУСК] Этап 4: Отбор молекул ---")
        generate_molecules_main()
        print("\n--- [ЗАВЕРШЕНО] Этап 4: Отбор молекул ---")

    elif args.command == 'run-all':
        print("--- [ЗАПУСК] Полный пайплайн ---")
        
        print("\n>>> Этап 1/4: Подготовка набора данных...")
        prepare_data_main()
        print(">>> Этап 1/4 завершен.")

        print("\n>>> Этап 2/4: Обучение модели...")
        prepare_predictor_main()
        print(">>> Этап 2/4 завершен.")

        print("\n>>> Этап 3/4: Генерация молекул...")
        generate_molecules_main()
        print(">>> Этап 3/4 завершен.")
        
        print("\n>>> Этап 4/4: Отбор молекул...")
        select_molecules_main()
        print(">>> Этап 4/4 завершен.")

        print("\n--- [ЗАВЕРШЕНО] Полный пайплайн выполнен успешно! ---")


if __name__ == '__main__':
    main()