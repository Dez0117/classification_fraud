# Fraud Detection - Credit Card Transactions

Проект по обнаружению мошеннических операций с кредитными картами с использованием машинного обучения.

## Структура проекта

- `EDA.ipynb` - разведочный анализ данных 
- `baseline1.ipynb` - базовые модели и эксперименты  
- `hyperparam_tuning.ipynb` - тонкая настройка гиперпараметров с Optuna (не дала результатов)
- `improvements.ipynb` - улучшения моделей 

## Основные подходы

### Feature Engineering
- Логарифмирование суммы транзакции (`log_amount`)
- Временные фичи через косинусное преобразование (`cos_time`)

### Модели
- **XGBoost** с учетом дисбаланса классов (`scale_pos_weight`)
- Метрика качества: **AUPRC** (Precision-Recall AUC)
- Кросс-валидация и early stopping

### Гиперпараметры
- Широкий поиск с Optuna 
- TPE sampler + Hyperband pruner
- 3-fold stratified CV

## Результаты

- Before optuna AUPRC(XGBoost + ClassWeight): ~0.880
- После тюнинга: ~0.878  

Бейзлайн показал довольно высокое качество, дальнейший тюнинг не дал значимого улучшения.

## Запуск

```bash
# Установка зависимостей
uv sync

