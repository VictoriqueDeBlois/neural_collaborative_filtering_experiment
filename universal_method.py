import os
import sqlite3
import time
from datetime import date, datetime, time
from typing import Union

import mysql.connector
import numpy as np


def load_original_matrix_data(file_name):
    original_data = np.loadtxt(file_name)
    return original_data


def load_csv_file(sparseness, index, matrix_type='rt', training_set=True, ext='csv'):
    return './CSV_{matrix_type}/sparseness{sparseness}/{data}{index}.{ext}' \
        .format(matrix_type=matrix_type, sparseness=sparseness,
                data='training' if training_set else 'test',
                index=index,
                ext=ext)


def mkdir(path):
    if os.path.exists(path) is not True:
        os.mkdir(path)


def load_data(file):
    data = np.loadtxt(file)
    user_id, item_id, rating = data[..., 0], data[..., 1], data[..., 2]
    user_id = np.array(user_id, dtype=int)
    item_id = np.array(item_id, dtype=int)
    rating = np.array(rating, dtype=float)
    return user_id, item_id, rating


def insert_database(path: str, table: str, data: dict):
    database = sqlite3.connect(path)
    keys = ', '.join(data.keys())
    holder = ', '.join(['?'] * len(data.keys()))
    command = 'insert into {} ({}) values ({})'.format(table, keys, holder)
    values = list(data.values())
    try:
        database.execute(command, values)
    except Exception as error:
        print(error)
    database.commit()
    database.close()


def auto_insert_database(config: dict,
                         data: dict,
                         table: Union[str, None] = None):
    data = key_to_lower(data)
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    # check table
    table_query = "show tables"
    cursor.execute(table_query)
    tables = set(cursor)
    if (table,) not in tables:
        auto_create_table(cursor, data, table)
    # check column
    else:
        column_query = 'desc `{}`'.format(table)
        cursor.execute(column_query)
        columns = set(map(lambda c: c[0].lower(), cursor))
        keys = set(data.keys())
        diff = keys - columns
        len_diff = len(diff)
        if len_diff > 0:
            auto_add_column(cursor, diff, data, table)
    # insert data
    keys = ', '.join(data.keys())
    holder = ', '.join(map(lambda k: '%({})s'.format(k), data.keys()))
    insert_query = 'insert into `{}` ({}) values ({})'.format(table, keys, holder)
    cursor.execute(insert_query, data)
    cnx.commit()
    cursor.close()
    cnx.close()


def key_to_lower(data: dict):
    new_dict = {}
    for k, v in data.items():
        new_dict[k.lower()] = v
    return new_dict


def convert_type(python_type: type) -> str:
    if python_type == int:
        mysql_type = 'int'
    elif python_type == float:
        mysql_type = 'double'
    elif python_type == str:
        mysql_type = 'char'
    elif python_type == bool:
        mysql_type = 'tinyint(1)'
    elif python_type == date:
        mysql_type = 'date'
    elif python_type == datetime:
        mysql_type = 'datetime'
    elif python_type == time:
        mysql_type = 'time'
    else:
        mysql_type = 'char'
    return mysql_type


def auto_create_table(cursor, data: dict, table: str):
    create_query = "create table `{}`(`table_no` int NOT NULL AUTO_INCREMENT, {}, PRIMARY KEY (`table_no`))" \
        .format(table,
                ','.join(map(lambda item: '`{}` {} null'.format(item[0], convert_type(type(item[1]))), data.items())))
    cursor.execute(create_query)


def auto_add_column(cursor, diff_keys: set, data: dict, table: str):
    add_query = "alter table `{}` add `{}` {} null"
    for key in diff_keys:
        cursor.execute(add_query.format(table, key, convert_type(type(data[key]))))


def evaluate(sparseness, index, fit_matrix):
    u, i, r = load_data(load_csv_file(sparseness, index, training_set=False))
    y = fit_matrix[u, i]
    mae = np.sum(np.abs(y - r)) / len(r)
    rmse = np.sqrt(np.sum(np.square(y - r)) / len(r))
    return mae, rmse


def create_sparse_matrix(filename, user_num=339, ws_num=5825):
    user_ls, ws_ls, rt = load_data(filename)
    array_obj = np.zeros((user_num, ws_num))
    array_obj[user_ls, ws_ls] = rt
    return array_obj


if __name__ == '__main__':
    pass
