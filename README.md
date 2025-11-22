# Fraud Detection - Credit Card Transactions

Проект по обнаружению мошеннических операций с кредитными картами с использованием машинного обучения.

## Структура проекта

- `EDA.ipynb` - разведочный анализ данных и feature engineering
- `baseline1.ipynb` - базовые модели и эксперименты  
- `hyperparam_tuning.ipynb` - тонкая настройка гиперпараметров с Optuna (не дала результатов)
- `improvements.ipynb` - улучшения моделей и финальные результаты
- `creditcard.csv` - исходные данные транзакций
- `optuna_study.pkl` - результаты оптимизации гиперпараметров

## Основные подходы

### Feature Engineering
- Логарифмирование суммы транзакции (`log_amount`)
- Временные фичи через косинусное преобразование (`cos_time`)

### Модели
- **XGBoost** с учетом дисбаланса классов (`scale_pos_weight`)
- Метрика качества: **AUPRC** (Precision-Recall AUC)
- Кросс-валидация и early stopping

### Гиперпараметры
- Широкий поиск с Optuna (300 trials, 4 часа)
- TPE sampler + Hyperband pruner
- 3-fold stratified CV

## Результаты

- Baseline AUPRC: ~0.880
- После тюнинга: ~0.878  
- Оптимальный порог классификации: ~0.35

Бейзлайн показал excellent качество, дальнейший тюнинг не дал значимого улучшения из-за ceiling effect.

## Запуск

```bash
# Установка зависимостей
uv sync
