# Программирование (Python)
# 6 семестр, тема 1

# Лабораторная работа 2
"""
Используя обучающий набор данных о пассажирах Титаника, находящийся в проекте (оригинал: https://www.kaggle.com/c/titanic/data), найдите ответы на следующие вопросы: 

1. Какое количество мужчин и женщин ехало на пароходе? Приведите два числа через пробел.

2. Подсчитайте сколько пассажиров загрузилось на борт в различных портах? Приведите три числа через пробел.

3. Посчитайте долю (процент) погибших на параходе (число и процент)?

4. Какие доли составляли пассажиры первого, второго, третьего класса?

5. Вычислите коэффициент корреляции Пирсона между количеством супругов (SibSp) и количеством детей (Parch).

6. Выясните есть ли корреляция (вычислите коэффициент корреляции Пирсона) между:
1) возрастом и параметром Survived;
2) полом человека и параметром Survived;
3) классом, в котором пассажир ехал, и параметром Survived.

7. Посчитайте средний возраст пассажиров и медиану.
8. Посчитайте среднюю цену за билет и медиану.

9. Какое самое популярное мужское имя на корабле?
10. Какие самые популярные мужское и женские имена людей, старше 15 лет на корабле?


Для вычисления 3, 4, 5, 6, 7, 8 используйте тип данных float с точностью два знака в дробной части. 
"""

import pandas as pd # импортирование библиотеки для считывания данных
from numpy import isnan

# считаем данных из файла, в качестве столбца индексов используем PassengerId
data = pd.read_csv('train.csv', index_col="PassengerId")

mark_spend = pd.read_csv('MarketingSpend.csv')

# TODO #1
def get_sex_distrib(data):
    """
    1. Какое количество мужчин и женщин ехало на параходе? Приведите два числа через пробел.
    """

    n_male, n_female = data['Sex'].value_counts()
    return f"{n_male}, {n_female}"

print("1. ", get_sex_distrib(data))

# TODO #2
def get_port_distrib(data):
    """  
    2. Подсчитайте сколько пассажиров загрузилось на борт в различных портах? Приведите три числа через пробел.
    """

    port_S, port_C, port_Q = data['Embarked'].value_counts()
    return f"{port_S}, {port_C}, {port_Q}"

print("2. ", get_port_distrib(data))
  
# TODO #3
def get_surv_percent(data):
    """
    3. Посчитайте долю погибших на пароходе (число и процент)?
    """
    n_died = data['Survived'].value_counts()[0]
    perc_died = data['Survived'].value_counts(normalize=True)[0] * 100
  
    return f"{n_died}, {perc_died:.2f}%"

print("3. ", get_surv_percent(data))

# TODO #4
def get_class_distrib(data):
    """
    4. Какие доли составляли пассажиры первого, второго, третьего класса?    
    """
    n_pas_3_cl, n_pas_1_cl, n_pas_2_cl = data['Pclass'].value_counts(normalize=True) * 100
    
    return "1 class: {:.2f}%, 2 class: {:.2f}%, 3 class: {:.2f}%".format(n_pas_1_cl, n_pas_2_cl, n_pas_3_cl)


print("4. ", get_class_distrib(data))
  
# TODO #5-6
"""
    5. Вычислите коэффициент корреляции Пирсона между количеством супругов (SibSp) и количеством детей (Parch).

    6. Выясните есть ли корреляция (вычислите коэффициент корреляции Пирсона) между:
    - возрастом и параметром Survived;
    - полом человека и параметром Survived;
    - классом, в котором пассажир ехал, и параметром Survived.
"""

class Corr:
      
    #метод для поиска корреляции числовых данных
    @staticmethod
    def find_corr(data, firstColumn, secondColumn):
        
        firstData = []
        secondData = []

        for i in data.iloc:
            if not(isnan(i[firstColumn])) and not(isnan(i[secondColumn])):
                firstData.append(i[firstColumn])
                secondData.append(i[secondColumn])

        return Corr.count_corr(firstData, secondData)

    #метод для поиска корелляции с полом
    @staticmethod
    def find_corr_sex(data, column):
        
        columnData = []
        sexes = []

        for i in data.iloc:
            if not(isnan(i[column])):
                columnData.append(i[column])
                sexes.append(0) if i['Sex'] == "male" else sexes.append(1)

        return Corr.count_corr(sexes, columnData)

    #модуль для вычисления корреляции
    @staticmethod
    def count_corr(firstData, secondData):
        
        firstData, secondData = pd.Series(firstData), pd.Series(secondData)

        corr_val = firstData.corr(secondData)
        return f"{corr_val:.2f}"

  
print("5. ", Corr.find_corr(data, 'SibSp', 'Parch'))
print("6.1. ", Corr.find_corr(data, 'Age', 'Survived'))
print("6.2. ", Corr.find_corr_sex(data, 'Survived'))
print("6.3. ", Corr.find_corr(data, 'Pclass', 'Survived'))
 
# TODO #7-8
def find_mean_median(data, column):
    """
    7. Посчитайте средний возраст пассажиров и медиану.

    8. Посчитайте среднюю цену за билет и медиану.
    """

    #эти методы по дефолту исключают NaN
    mean, median = data[column].mean(), data[column].median()
    return f"Среднее: {mean:.2f}, медиана: {median:.2f}"

print("7. ", find_mean_median(data, 'Age'))
print("8. ", find_mean_median(data, 'Fare'))

# TODO #9
def find_popular_name(data):
    """
    9. Какое самое популярное мужское имя на корабле?
    """
    return pd.Series([i['Name'].partition(". ")[2].partition("(")[0].strip() for i in data.iloc if i['Sex'] == "male"]).value_counts().idxmax()

print("9. Самое популярное имя: ", find_popular_name(data))

# TODO #10
def find_popular_adult_names(data):
    """
    10. Какие самые популярные мужское и женские имена людей, старше 15 лет на корабле?
    """

    male_names = []
    female_names = []

    for i in data.iloc:
        name = i['Name'].partition(". ")[2].partition("(")[0].strip()
        
        if i['Age'] > 15 and i['Sex'] == "male":
            male_names.append(name)
        elif i['Age'] > 15 and i['Sex'] == "female":
            if "(" in i['Name']:
                name = i['Name'].partition("(")[2].strip()[:-1]
            female_names.append(name)
    
    popular_male_name = pd.Series(male_names).value_counts().idxmax()
    popular_female_name = pd.Series(female_names).value_counts().idxmax()
    return f"Мужское: {popular_male_name}, женское: {popular_female_name}"

print(find_popular_adult_names(data))

'''
Часть 2. Для набора данных из лабораторной работы 1 посчитать средние значения, медианы, максимальные и минимальные значения для столбцов Offline Spend, Online Spend.
'''

print("Online ", find_mean_median(mark_spend, 'Online Spend'))
print("Offline ", find_mean_median(mark_spend, 'Offline Spend'))

def get_max_and_min(data, column):
    max, min = data[column].max(), data[column].min()
    return f"Для колонки {column} макс: {max}, мин: {min}"

print(get_max_and_min(mark_spend, 'Online Spend'))
print(get_max_and_min(mark_spend, 'Offline Spend'))
# ------------------------------

# Реализуем вычисление количества пассажиров на параходе и опишем предварительные действия с данными (считывание)

# После загрузки данных с помощью метода read_csv и индексации его по первому столбцу данных (PassangerId) становится доступно выборка данных по индексу.
# С помощью запроса ниже мы можем получить имя сотого пассажира
print(type(data.iloc[100]))
print(data.iloc[100]['Name'])

print((data['Name'], data['Sex']))


def get_number_of_pass(data_file):
    """
        Подсчет количества пассажиров. 
        data_file - str
        В качестве аргумента удобнее всего использовать строковую переменную, куда будет передаваться название файла (т. к. далее, возможно, потребуется подсчитать параметры для другого набора данных test.csv)
    """
    male_int, female_int = 0, 0
    # считывание и обработка данных
    data = pd.read_csv(data_file, index_col="PassengerId")

    # считать данных из столбца возможно с помощью метода value_counts()
    res = data['Sex'].value_counts()
    # res будет содержать ассоциативный массив, ключами в котором являются значения столбца sex, а целочисленные значениями - количества пассажиров обоих полов
    male_int, female_int = res['male'], res['female']
    return male_int, female_int

print(get_sex_distrib(data))
print("Проценты классов ", get_class_distrib(data))
#print(find_corr_sex_survival(data))
print(data['Pclass'].value_counts().to_dict())

def test_get_number_of_pass():
    assert get_number_of_pass('train.csv') == (
        577, 314), " Количество мужчин и женщин на Титанике"


# аналогично протестировать остальные функции
