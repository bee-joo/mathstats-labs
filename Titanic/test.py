import pytest
import main
from pandas import read_csv


data = read_csv('train.csv')
data2 = read_csv('MarketingSpend.csv')


def test_get_sex_distrib():
    assert main.get_sex_distrib(data) == "577, 314"


def test_get_port_distrib():
    assert main.get_port_distrib(data) == "644, 168, 77"


def test_get_surv_percent():
    assert main.get_surv_percent(data) == "549, 61.62%"


def test_get_class_distrib():
    assert main.get_class_distrib(data) == "1 class: 24.24%, 2 class: 20.65%, 3 class: 55.11%"


@pytest.mark.corr
def test_corr_class1():
    assert main.Corr.find_corr(data, 'SibSp', 'Parch') == "0.41"


@pytest.mark.corr
def test_corr_class2():
    assert main.Corr.find_corr(data, 'Age', 'Survived') == "-0.08"


@pytest.mark.corr
def test_corr_class3():
    assert main.Corr.find_corr_sex(data, 'Survived') == "0.54"


@pytest.mark.corr
def test_corr_class4():
    assert main.Corr.find_corr(data, 'Pclass', 'Survived') == "-0.34"


@pytest.mark.mean
def test_find_mean_median1():
    assert main.find_mean_median(data, 'Age') == "Среднее: 29.70, медиана: 28.00"


@pytest.mark.mean
def test_find_mean_median2():
    assert main.find_mean_median(data, 'Fare') == "Среднее: 32.20, медиана: 14.45"


@pytest.mark.mean
def test_find_mean_median3():
    assert main.find_mean_median(data2, 'Online Spend') == "Среднее: 1905.88, медиана: 1881.94"


@pytest.mark.mean
def test_find_mean_median4():
    assert main.find_mean_median(data2, 'Offline Spend') == "Среднее: 2843.56, медиана: 3000.00"


def test_find_popular_name():
    assert main.find_popular_name(data) == "John"


def test_find_popular_adult_names():
    assert main.find_popular_adult_names(data) == "Мужское: William, женское: Mary"
