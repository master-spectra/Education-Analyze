import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# Настройка вывода кириллицы
import sys

# Настройка кодировки для Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Создаем директорию для сохранения графиков
if not os.path.exists('pic'):
    os.makedirs('pic')

# Настройка отображения в pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

# Настройка кодировки для matplotlib
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    """
    Загрузка данных из CSV файла
    """
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
        print("Данные успешно загружены")
        return data
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None

def initial_analysis(data):
    """
    Первичный анализ данных
    """
    # Базовая информация о датасете
    print("\nОбщая информация о датасете:")
    print(data.info())

    print("\nПервые 5 строк датасета:")
    print(data.head())

    print("\nСтатистическое описание числовых признаков:")
    print(data.describe())

    # Проверка пропущенных значений
    print("\nКоличество пропущенных значений:")
    print(data.isnull().sum())

def visualize_distributions(data):
    """
    Визуализация распределений числовых признаков
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[col], kde=True)
        plt.title(f'Распределение {col}')
        plt.savefig(f'pic/{col}_distribution.png')
        plt.close()

def correlation_analysis(data):
    """
    Анализ корреляций между признаками
    """
    numeric_data = data.select_dtypes(include=[np.number])

    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Корреляционная матрица')
    plt.tight_layout()
    plt.savefig('pic/correlation_matrix.png')
    plt.close()

def preprocess_data(data):
    """
    Предобработка данных:
    - One-hot encoding для категориальных переменных
    - Стандартизация числовых признаков
    """
    print("\nНачало предобработки данных...")

    # Создаем копию данных
    df = data.copy()

    # Разделение на категориальные и числовые признаки
    categorical_columns = df.select_dtypes(include=['object']).columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    print("\nКатегориальные признаки:", list(categorical_columns))
    print("Числовые признаки:", list(numeric_columns))

    # One-hot encoding для категориальных переменных
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Стандартизация числовых признаков (кроме целевой переменной)
    scaler = StandardScaler()
    numeric_features = [col for col in numeric_columns if col != 'Exam_Score']
    df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

    print("\nРазмерность данных после предобработки:", df_encoded.shape)
    print("Количество признаков после one-hot кодирования:", len(df_encoded.columns))

    return df_encoded, scaler

def validate_preprocessing(data_processed):
    """
    Проверка качества предобработки
    """
    print("\nПроверка результатов предобработки:")

    # Проверка на пропущенные значения
    missing_values = data_processed.isnull().sum()
    print("\nПропущенные значения после предобработки:")
    print(missing_values[missing_values > 0] if any(missing_values > 0) else "Пропущенных значений нет")

    # Проверка распределения числовых признаков
    numeric_cols = data_processed.select_dtypes(include=[np.number]).columns

    plt.figure(figsize=(15, 5))
    for i, col in enumerate(numeric_cols[:3], 1):  # Показываем первые 3 признака
        plt.subplot(1, 3, i)
        sns.histplot(data_processed[col], kde=True)
        plt.title(f'Распределение\n{col}')
    plt.tight_layout()
    plt.savefig('pic/preprocessed_distributions.png')
    plt.close()

def perform_t_tests(data):
    """
    Проведение t-тестов для анализа различий между группами
    """
    print("\nРезультаты t-тестов:")

    # Словарь для хранения результатов
    t_test_results = {}

    # T-test для гендерных различий
    male_scores = data[data['Gender'] == 'Male']['Exam_Score']
    female_scores = data[data['Gender'] == 'Female']['Exam_Score']
    t_stat, p_value = stats.ttest_ind(male_scores, female_scores)
    t_test_results['Gender'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_male': male_scores.mean(),
        'mean_female': female_scores.mean()
    }

    # T-test для типа школы
    public_scores = data[data['School_Type'] == 'Public']['Exam_Score']
    private_scores = data[data['School_Type'] == 'Private']['Exam_Score']
    t_stat, p_value = stats.ttest_ind(public_scores, private_scores)
    t_test_results['School_Type'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_public': public_scores.mean(),
        'mean_private': private_scores.mean()
    }

    # Визуализация результатов
    plt.figure(figsize=(12, 5))

    # График для гендерных различий
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Gender', y='Exam_Score', data=data)
    plt.title('Распределение оценок по полу')

    # График для типа школы
    plt.subplot(1, 2, 2)
    sns.boxplot(x='School_Type', y='Exam_Score', data=data)
    plt.title('Распределение оценок по типу школы')

    plt.tight_layout()
    plt.savefig('pic/t_test_results.png')
    plt.close()

    # Вывод результатов
    print("\n1. Анализ гендерных различий:")
    print(f"t-статистика: {t_test_results['Gender']['t_statistic']:.4f}")
    print(f"p-значение: {t_test_results['Gender']['p_value']:.4f}")
    print(f"Средний балл (муж.): {t_test_results['Gender']['mean_male']:.2f}")
    print(f"Средний балл (жен.): {t_test_results['Gender']['mean_female']:.2f}")

    print("\n2. Анализ различий по типу школы:")
    print(f"t-статистика: {t_test_results['School_Type']['t_statistic']:.4f}")
    print(f"p-значение: {t_test_results['School_Type']['p_value']:.4f}")
    print(f"Средний балл (гос.): {t_test_results['School_Type']['mean_public']:.2f}")
    print(f"Средний балл (част.): {t_test_results['School_Type']['mean_private']:.2f}")

    return t_test_results

def perform_extended_statistical_analysis(data):
    """
    Расширенный статистический анализ данных
    """
    print("\nРасширенный статистический анализ:")

    # 1. T-тесты для различных категориальных переменных
    categorical_vars = {
        'Parental_Involvement': ['High', 'Low'],
        'Access_to_Resources': ['High', 'Low'],
        'Internet_Access': ['Yes', 'No'],
        'Learning_Disabilities': ['Yes', 'No'],
        'Extracurricular_Activities': ['Yes', 'No']
    }

    results = {}

    for var, groups in categorical_vars.items():
        group1_scores = data[data[var] == groups[0]]['Exam_Score']
        group2_scores = data[data[var] == groups[1]]['Exam_Score']

        # T-test
        t_stat, p_value = stats.ttest_ind(group1_scores, group2_scores)

        # Эффект размера (Cohen's d)
        cohens_d = (group1_scores.mean() - group2_scores.mean()) / np.sqrt(
            ((len(group1_scores) - 1) * group1_scores.std()**2 +
             (len(group2_scores) - 1) * group2_scores.std()**2) /
            (len(group1_scores) + len(group2_scores) - 2))

        results[var] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_group1': group1_scores.mean(),
            'mean_group2': group2_scores.mean(),
            'cohens_d': cohens_d
        }

        # Визуализация
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=var, y='Exam_Score', data=data)
        plt.title(f'Распределение оценок по {var}')
        plt.savefig(f'pic/boxplot_{var}.png')
        plt.close()

    # 2. ANOVA для уровня образования родителей
    education_groups = data.groupby('Parental_Education_Level')['Exam_Score'].apply(list)
    f_stat, p_value = stats.f_oneway(*education_groups)

    # Визуализация для ANOVA
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Parental_Education_Level', y='Exam_Score', data=data)
    plt.xticks(rotation=45)
    plt.title('Распределение оценок по уровню образования родителей')
    plt.tight_layout()
    plt.savefig('pic/anova_education.png')
    plt.close()

    # 3. Тест Шапиро-Уилка на нормальность распределения
    shapiro_stat, shapiro_p = stats.shapiro(data['Exam_Score'])

    # Вывод результатов
    print("\n1. Результаты t-тестов для различных факторов:")
    for var, res in results.items():
        print(f"\nАнализ влияния фактора {var}:")
        print(f"t-статистика: {res['t_statistic']:.4f}")
        print(f"p-значение: {res['p_value']:.4f}")
        print(f"Размер эффекта (Cohen's d): {res['cohens_d']:.4f}")
        print(f"Средние значения: {groups[0]}: {res['mean_group1']:.2f}, {groups[1]}: {res['mean_group2']:.2f}")

    print("\n2. Результаты ANOVA для уровня образования родителей:")
    print(f"F-статистика: {f_stat:.4f}")
    print(f"p-значение: {p_value:.4f}")

    print("\n3. Тест на нормальность распределения оценок:")
    print(f"Статистика Шапиро-Уилка: {shapiro_stat:.4f}")
    print(f"p-значение: {shapiro_p:.4f}")
    print("Распределение", "нормальное" if shapiro_p > 0.05 else "не нормальное")

    return results

def prepare_data_for_modeling(data_processed):
    """
    Подготовка данных для моделирования
    """
    # Разделяем признаки и целевую переменную
    X = data_processed.drop('Exam_Score', axis=1)
    y = data_processed['Exam_Score']

    # Разделение на train, validation и test (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 от 80% = 20% от всего
    )

    print("Размеры выборок:")
    print(f"Обучающая выборка: {X_train.shape}")
    print(f"Валидационная выборка: {X_val.shape}")
    print(f"Тестовая выборка: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(X_train, X_val, y_train, y_val):
    """
    Обучение модели RandomForest
    """
    # Базовая модель RandomForest
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )

    # Обучение модели
    model.fit(X_train, y_train)

    # Оценка на валидационной выборке
    val_score = model.score(X_val, y_val)
    print(f"\nR2 score на валидационной выборке: {val_score:.4f}")

    return model

def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Оценка качества модели
    """
    # Предсказания
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Метрики
    results = {
        'train': {
            'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'MAE': mean_absolute_error(y_train, y_train_pred),
            'R2': r2_score(y_train, y_train_pred)
        },
        'val': {
            'RMSE': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'MAE': mean_absolute_error(y_val, y_val_pred),
            'R2': r2_score(y_val, y_val_pred)
        },
        'test': {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'R2': r2_score(y_test, y_test_pred)
        }
    }

    print("\nРезультаты оценки модели:")
    for dataset, metrics in results.items():
        print(f"\n{dataset.upper()} set:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    # Визуализация важности признаков
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Топ-10 важных признаков')
    plt.tight_layout()
    plt.savefig('pic/feature_importance.png')
    plt.close()

    return results, feature_importance

def perform_grid_search(X_train, y_train, cv=5):
    """
    Подбор гиперпараметров с помощью GridSearchCV
    """
    print("\nНачало подбора гиперпараметров...")

    # Определяем параметры для поиска
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'n_estimators': [100, 200],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['auto', 'sqrt']
    }

    # Создаем базовую модель
    base_model = RandomForestRegressor(random_state=42)

    # Определяем скоринг
    scoring = {
        'rmse': make_scorer(lambda y, y_pred: -np.sqrt(mean_squared_error(y, y_pred))),
        'r2': make_scorer(r2_score)
    }

    # Создаем GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit='rmse',
        verbose=2,
        n_jobs=-1
    )

    # Выполняем поиск
    grid_search.fit(X_train, y_train)

    print("\nЛучшие параметры:", grid_search.best_params_)
    print("Лучший RMSE:", -grid_search.best_score_)

    return grid_search.best_estimator_

def visualize_predictions(model, X_test, y_test):
    """
    Визуализация предсказаний модели
    """
    # Получаем предсказания
    y_pred = model.predict(X_test)

    # График сравнения реальных и предсказанных значений
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Реальные значения')
    plt.ylabel('Предсказанные значения')
    plt.title('Сравнение реальных и предсказанных значений')
    plt.tight_layout()
    plt.savefig('pic/predictions_comparison.png')
    plt.close()

    # График остатков
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Остатки')
    plt.ylabel('Частота')
    plt.title('Распределение остатков')
    plt.tight_layout()
    plt.savefig('pic/residuals_distribution.png')
    plt.close()

    # График остатков от предсказанных значений
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Остатки')
    plt.title('Остатки vs Предсказанные значения')
    plt.tight_layout()
    plt.savefig('pic/residuals_vs_predicted.png')
    plt.close()

def main():
    """
    Основная функция
    """
    # Загрузка данных
    data = load_data('data/StudentPerformanceFactors.csv')

    if data is not None:
        # Выполнение начального анализа
        initial_analysis(data)
        visualize_distributions(data)
        correlation_analysis(data)

        # Предобработка данных
        data_processed, scaler = preprocess_data(data)

        # Проверка результатов предобработки
        validate_preprocessing(data_processed)

        # Проведение t-тестов
        t_test_results = perform_t_tests(data)  # Используем исходные данные для t-тестов

        # Расширенный статистический анализ
        statistical_results = perform_extended_statistical_analysis(data)

        # Подготовка данных для моделирования
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_for_modeling(data_processed)

        # Подбор гиперпараметров с помощью GridSearchCV
        best_model = perform_grid_search(X_train, y_train)

        # Оценка лучшей модели
        results, feature_importance = evaluate_model(
            best_model, X_train, X_val, X_test, y_train, y_val, y_test
        )

        # Визуализация предсказаний
        visualize_predictions(best_model, X_test, y_test)

if __name__ == "__main__":
    main()
