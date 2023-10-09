import csv
import pickle

import pandas as pd
import numpy as np

MODEL = 'pickle_model/model_cbr.pkl'
CALENDAR = 'pickle_model/calendar_mini.csv'

from pyexpat import model

PERIOD = 14


def prepare_data(data_list):
    def covert_data(data_list):
        # загружаем данные, представдленные в виде словарей и преобразовываеи их в табличный вид
        chunk_size = 1000
        columns = list(data_list[0].keys())
        data = pd.DataFrame(columns=columns)
        # Итеративно добавляем данные из списка словарей в DataFrame по чанкам
        for i in range(0, len(data_list), chunk_size):
            chunk = data_list[i:i + chunk_size]
            data_chunk = pd.DataFrame.from_records(chunk)
            data = pd.concat([data, data_chunk], ignore_index=True)
        return data

    data = covert_data(data_list)
    # data = data_list.copy()
    # определяем типы данных
    columns_to_convert = ['sales_type', 'type_format', 'loc', 'size',
                          'is_active', 'uom', 'holiday']
    # df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce', downcast='integer')
    data['date'] = pd.to_datetime(data['date'])
    data[columns_to_convert] = data[columns_to_convert].apply(pd.to_numeric,
                                                              errors='coerce').astype(
        'Int64')
    # создаем новые признаки
    # data.reset_index('date', inplace=True)
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['weekday'] = data['date'].dt.weekday
    data['day'] = data['date'].dt.day
    data.info()
    # изменим порядок столбцов на более удобный
    new_order = ['date', 'is_active', 'store', 'sku', 'sales_type',
                 'sales_units', 'sales_units_promo',
                 'sales_rub', 'sales_rub_promo', 'group', 'category',
                 'subcategory', 'uom', 'division',
                 'city', 'type_format', 'loc', 'size', 'year', 'month', 'day',
                 'weekday', 'holiday']
    data = data[new_order]
    data.set_index('date', inplace=True)
    data.info()
    # Удалим магазины с малой активностью
    # список магазинов
    list_store = data['store'].unique().tolist()
    # список для хранения данных о магазинах и их количестве дней работы
    activ_store_data = []
    store_data = []

    for st in list_store:
        min_date = data.loc[data['store'] == st].index.min().date()
        max_date = data.loc[data['store'] == st].index.max().date()
        delta = max_date - min_date
        days = delta.days
        if days > 180:
            activ_store_data.append({'stores': st, 'days_worked': days})
        else:
            store_data.append({'stores': st, 'days_worked': days})
    # получим список идентификаторов магазинов из списка store_data
    store_ids = [entry['stores'] for entry in store_data]
    mask = data['store'].isin(store_ids)  # cоздаем маску для строк на удаления
    activ_store = data[~mask]
    activ_store = activ_store.drop('is_active',
                                   axis=1)  # удалим столбцец 'is_active'

    # Удалим отрицательные значения
    mask_negativ = (
            (activ_store['sales_units'] < 0) | (
                activ_store['sales_units_promo'] < 0) | (
                    activ_store['sales_rub'] < 0) |
            (activ_store['sales_rub_promo'] < 0))
    # удаляем строки, соответствующие маске, из activ_store
    activ_store = activ_store[~mask_negativ]

    # Обработка нулевых значений (заполним нули в столбцах sales_units и sales_rub)
    activ_store['sales_units'] = activ_store['sales_units'].replace(0,
                                                                    np.nan)  # меням 0 на пропуски
    activ_store['sales_rub'] = activ_store['sales_rub'].replace(0, np.nan)
    activ_store['sales_units'] = activ_store['sales_units'].fillna(
        method='ffill')  # заменим пропуски на значение
    activ_store['sales_rub'] = activ_store['sales_rub'].fillna(method='ffill')

    # первую стороку заполним средним значением
    activ_store['sales_units'] = activ_store['sales_units'].fillna(
        activ_store['sales_units'].mean())
    activ_store['sales_rub'] = activ_store['sales_rub'].fillna(
        activ_store['sales_rub'].mean())
    # приведем к целым значениям
    activ_store['sales_units'] = activ_store['sales_units'].astype(int)
    activ_store['sales_units_promo'] = activ_store['sales_units_promo'].astype(
        int)
    activ_store.reset_index()

    # Добавим долю продаж промо 'promo_part' в таблицу
    # сделаем группировку и агрегацию по товарам, сорртировака по сумме
    ales_type_sorted = activ_store.groupby('sku')[
        ['sales_units', 'sales_units_promo']] \
        .agg(['sum']).sort_values(by=('sales_units', 'sum'), ascending=False)
    # добавим столбец с долей товаров с промо в процентах
    activ_store['promo_part'] = activ_store['sku'].map(
        activ_store.groupby('sku')[['sales_units', 'sales_units_promo']]
        .agg({'sales_units': 'sum', 'sales_units_promo': 'sum'})
        .eval('sales_units_promo / sales_units').round(3))

    # удалим те строки с промо, у которых есть продажи без промо
    activ_store = activ_store[
        ~(activ_store['sales_type'] == 1) & (
            activ_store['sku'].isin(
                activ_store.loc[activ_store['sales_type'] == 0, 'sku']))]
    # удалим столбцы 'sales_type', 'sales_units_promo', 'sales_rub_promo'
    activ_store = activ_store.drop(
        ['sales_type', 'sales_units_promo', 'sales_rub_promo'], axis=1)

    # Добавим признак цена за единицу товара 'price_units'
    activ_store['price_units'] = (
            activ_store['sales_rub'] / activ_store['sales_units']).round(2)

    # Проведем НОРМАЛИЗАЦИЮ числовых признаков: sales_units, price_units и sales_rub
    # нормализация продажи в штуках
    sales_units_mean = activ_store.sales_units.mean()
    sales_units_sd = activ_store.sales_units.std()
    # Сформируем целевой признак 'sales_units_stand'
    activ_store['sales_units_stand'] = (
                                               activ_store.sales_units - sales_units_mean) / sales_units_sd

    # нормализация продажи в рублях
    sales_rub_mean = activ_store.sales_rub.mean()
    sales_rub_sd = activ_store.sales_rub.std()
    # Сформируем признак 'sales_rub_stand'
    activ_store['sales_rub_stand'] = (
                                             activ_store.sales_rub - sales_rub_mean) / sales_rub_sd

    # нормализация доли промо продажи в рублях за штуку
    price_units_mean = activ_store.price_units.mean()
    price_units_sd = activ_store.price_units.std()
    # Сформируем признак 'price_units_stand'
    activ_store['price_units_stand'] = (
                                               activ_store.price_units - price_units_mean) / price_units_sd

    # Разделим признаки на числовые и категориальные, создадим списки:
    numeric = activ_store[activ_store.select_dtypes(include='number').columns]
    categorical = activ_store[
        activ_store.select_dtypes(include='object').columns]
    numeric_columns = numeric.columns.tolist()
    categorical_columns = categorical.columns.tolist()

    # Добавим признаки: доли продаж товаров по категориям и по магазинам (в шт и в руб)
    def sales_share(data, name_col, name, columns_to_process):
        for column in columns_to_process:
            unique_counts = data.groupby(column)[
                'sku'].nunique()  # количество уникальных товаров в каждой категории/типа магазина
            sales_sum = data.groupby(column)[
                name_col].sum()  # сумма продаж для каждой категории/типа магазина
            sku_sales = unique_counts / sales_sum  # доля продаж уникальных товаров в каждой категории/типа магазина
            # доля продаж уникальных товаров в таблицу
            data[f'sales_share_{column}_{name}'] = data[column].map(sku_sales)
        return data

    columns_to_process = ['group', 'category',
                          'subcategory']  # перебор по категориям
    columns_to_process_sku = ['store', 'sku', 'year', 'weekday', 'month',
                              'division', 'city', 'type_format', 'loc',
                              'size']  # перебор по магазинам
    # Добавим доли продаж товаров по категориям
    activ_store = sales_share(activ_store, 'sales_units', 'unit',
                              columns_to_process)  # в шт
    activ_store = sales_share(activ_store, 'sales_rub', 'rub',
                              columns_to_process)  # в rub
    # Добавим доли продаж товаров по магазинам
    activ_store = sales_share(activ_store, 'sales_units', 'sku_unit',
                              columns_to_process_sku)  # в шт
    activ_store = sales_share(activ_store, 'sales_rub', 'sku_rub',
                              columns_to_process_sku)  # в rub

    # КЛАСТЕРИЗАЦИЯ
    clast_activ_store = activ_store.copy()  # создадим копию данных
    # переведем столбцы в категориальные признаки
    clast_activ_store['type_format'] = clast_activ_store['type_format'].astype(
        'object')
    clast_activ_store['loc'] = clast_activ_store['loc'].astype('object')
    clast_activ_store['size'] = clast_activ_store['size'].astype('object')
    # список числовых столбцов
    numeric_columns = clast_activ_store.select_dtypes(
        include=['number']).columns
    # переберем столбцы и преобразуем их в 'category', если они не являются числовыми
    for column in clast_activ_store.columns:
        if column not in numeric_columns:
            clast_activ_store[column] = clast_activ_store[column].astype(
                'category')

    # Новые признаки
    def clast_futures(data, name_col, name, columns_to_process):
        for column in columns_to_process:
            data[f'avg_sales_by_{column}_{name}'] = data.groupby(column)[
                name_col].transform('mean')
        # средние продажи по выходным и праздникам в рублях/шт
        data[f'avg_holiday_sales_{name}'] = data[name_col] * data['holiday']
        return data

    columns_to_process_cl = ['store', 'sku', 'year', 'weekday', 'month',
                             'category', 'subcategory',
                             'division', 'city', 'type_format', 'loc', 'size']
    # средние продажи
    clast_activ_store = clast_futures(clast_activ_store, 'sales_rub', 'rub',
                                      columns_to_process_cl)  # в rub
    clast_activ_store = clast_futures(clast_activ_store, 'sales_units', 'units',
                                      columns_to_process_cl)  # в шт

    # группируем данные по 'sku' и 'store' и считаем сумму продаж
    sku_sales_sum = clast_activ_store.groupby(['sku', 'store'])[
        'sales_units'].sum().reset_index()
    n_clusters = 11  # количество кластеров
    X = sku_sales_sum[
        ['sales_units']]  # таблица с количеством продаж 'sales_units'
    kmeans = KMeans(n_clusters=n_clusters)  # KMeans с количеством кластеров

    # кластеризация
    sku_sales_sum['cluster'] = kmeans.fit_predict(X)
    # признаки для кластеризации
    cluster_features = ['sales_units', 'avg_sales_by_store_units',
                        'avg_sales_by_sku_units',
                        'avg_sales_by_weekday_units',
                        'avg_sales_by_month_units', 'avg_holiday_sales_units']
    # масштабирование признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clast_activ_store[cluster_features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clast_activ_store['cluster'] = kmeans.fit_predict(X_scaled)

    # соединим с исходной таблицей activ_store
    clast_activ_store = clast_activ_store.reset_index().merge(
        sku_sales_sum[['sku', 'store', 'cluster']], \
        on=['sku', 'store', 'cluster'], how='left').set_index('date')
    data = clast_activ_store.copy()
    return data

    def predict_sale(data):
        # Выполнение предсказания с использованием обученной модели model
        try:
            predicted_sale = model.predict(data)
        except:
            print(
                'по данному товару прогноз невозможен, проверьте данные по товару или магазину')

        return predicted_sale[0]

    # формирование списка предсказанной цены

    predicted_sales = data.apply(
        lambda row: predict_sale(row['pr_sales_in_units'], row['st_id'],
                                 row['pr_sku_id']), axis=1).tolist()


def cleaning_data(data):
    # обработка данных

    return data


def forecast_real(sales: dict, item_info: dict, store_info: dict) -> list:
    """
    Функция для предсказания продажЖ
    :params sales: исторические данные по продажам
    :params item_info: характеристики товара
    :params store_info: характеристики магазина
    """

    # Загрузка модели
    with open(MODEL, 'rb') as fid:
        model = pickle.load(fid)

    with open(CALENDAR, newline='') as csv_file:
        calendar = {row['date']: int(row['holiday'])
                    for row in csv.DictReader(csv_file)}

    try:
        for sale in sales:
            predicted_sale = model.predict(
                cleaning_data({**sale, **store_info, **item_info,
                               'holiday': calendar[sale['date']]}))
    except KeyError:
        print('Ошибка в data')

        print('по данному товару прогноз невозможен, '
              'проверьте данные по товару или магазину')
    # predicted_sale == [10,15,20,30,20,2,...]
    return predicted_sale[0].tolist()


def forecast_test(sales: dict, item_info: dict, store_info: dict) -> list:
    """
    Функция для предсказания продажЖ
    :params sales: исторические данные по продажам
    :params item_info: характеристики товара
    :params store_info: характеристики магазина

    """

    sales = [el["sales_units"] for el in sales]
    mean_sale = sum(sales) / len(sales)
    return [mean_sale] * 14
