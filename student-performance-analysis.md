# Отчет по анализу успеваемости студентов

## 1. Общая характеристика данных

- Датасет содержит информацию о **6607 студентах**
- Имеется **20 различных характеристик**, включая целевую переменную (Exam_Score)
- Присутствует небольшое количество пропущенных значений:
  - Teacher_Quality: 78
  - Parental_Education_Level: 90
  - Distance_from_Home: 67

## 2. Анализ распределений ключевых показателей

### Целевая переменная (Exam_Score)
- Среднее значение: **67.24**
- Распределение близко к нормальному, но имеет небольшую правостороннюю асимметрию
- Основная масса оценок находится в диапазоне **65-70 баллов**

### Ключевые факторы

#### Hours_Studied
- Нормальное распределение
- Большинство студентов учатся **15-25 часов**
- Средняя продолжительность: **20 часов**

#### Attendance
- Относительно равномерное распределение
- Средняя посещаемость: **80%**
- Заметный пик в районе **80-85%**

#### Sleep_Hours
- Большинство студентов спят **6-8 часов**
- Наиболее частое значение: **7 часов**

## 3. Статистический анализ

### Корреляционный анализ
Наиболее сильные корреляции с Exam_Score:
1. Attendance (0.58)
2. Hours_Studied (0.45)
3. Previous_Scores (0.18)

### T-тесты показали

#### Гендерные различия
- Статистически незначимые (p-value > 0.05)
- Практически одинаковые средние баллы

#### Тип школы
- Незначительные различия
- Частные школы показывают немного лучшие результаты

### Другие факторы
Значимое влияние оказывают:
- Parental_Involvement
- Access_to_Resources
- Internet_Access

## 4. Моделирование

### Качество модели
- R² на тестовой выборке: **0.5432**
- RMSE: **2.5412**
- MAE: **1.5990**

### Важность признаков (топ-5)
1. Attendance (0.45)
2. Hours_Studied (0.30)
3. Previous_Scores (0.08)
4. Tutoring_Sessions (0.05)
5. Access_to_Resources (0.03)

## 5. Основные выводы

### Наиболее важные факторы успеваемости
1. Посещаемость занятий
2. Количество часов самостоятельной подготовки
3. Предыдущие академические результаты

- Демографические факторы (пол, тип школы) оказывают минимальное влияние
- Социально-экономические факторы (доступ к ресурсам, вовлеченность родителей) имеют умеренное влияние

## 6. Рекомендации

1. Фокус на повышение посещаемости
2. Организация эффективного учебного времени
3. Обеспечение доступа к образовательным ресурсам
4. Вовлечение родителей в образовательный процесс

## 7. Ограничения модели

- Умеренная предсказательная способность (R² ≈ 0.54)
- Наличие выбросов в предсказаниях
- Небольшое количество пропущенных данных