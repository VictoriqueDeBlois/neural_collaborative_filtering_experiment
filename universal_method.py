import os
import smtplib
import sqlite3
import time
from datetime import date, datetime, time
from email.header import Header
from email.mime.text import MIMEText
from os.path import join, dirname
from typing import Union

import mysql.connector
import numpy as np

user_num = 339
ws_num = 5825


def load_original_matrix_data(file_name):
    original_data = np.loadtxt(file_name)
    return original_data


def load_csv_file(sparseness, index, matrix_type='rt', training_set=True, ext='csv'):
    file = 'CSV_{matrix_type}/sparseness{sparseness}/{data}{index}.{ext}' \
        .format(matrix_type=matrix_type, sparseness=sparseness,
                data='training' if training_set else 'test',
                index=index,
                ext=ext)
    return join(dirname(__file__), file)


def save_csv_file(sparseness, index, extend, matrix_type='rt', ext='csv'):
    file = 'CSV_{matrix_type}/sparseness{sparseness}/{data}{index}_ex{extend}.{ext}' \
        .format(matrix_type=matrix_type, sparseness=sparseness, extend=extend,
                data='training',
                index=index,
                ext=ext)
    return join(dirname(__file__), file)


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


def load_training_data(sparseness, index, extend_near_num):
    if extend_near_num <= 0:
        training_file = load_csv_file(sparseness, index)
    else:
        training_file = save_csv_file(sparseness, index, extend_near_num)
        if os.path.exists(training_file) is not True:
            from improve_distance_calculate import save_extend_array
            save_extend_array(sparseness, index, extend_near_num)
    return load_data(training_file)


def save_data(file, user_id, item_id, rating):
    np.savetxt(file, np.array([user_id, item_id, rating]).T, "%.6f")


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
    keys = ', '.join(map(lambda k: '`{}`'.format(k), data.keys()))
    holder = ', '.join(map(lambda k: '%({})s'.format(k), data.keys()))
    insert_query = 'insert into `{}` ({}) values ({})'.format(table, keys, holder)
    data = convert_unknown_type_to_str(data)
    cursor.execute(insert_query, data)
    cnx.commit()
    cursor.close()
    cnx.close()


def key_to_lower(data: dict):
    new_dict = {}
    for k, v in data.items():
        new_dict[k.lower()] = v
    return new_dict


def convert_unknown_type_to_str(data: dict):
    new_dict = {}
    for k, v in data.items():
        new_dict[k] = str(v) if convert_type(type(v)) == 'text' else v
    return new_dict


def convert_type(python_type: type) -> str:
    if python_type == int:
        mysql_type = 'int'
    elif python_type == float:
        mysql_type = 'double'
    elif python_type == str:
        mysql_type = 'text'
    elif python_type == bool:
        mysql_type = 'tinyint(1)'
    elif python_type == date:
        mysql_type = 'date'
    elif python_type == datetime:
        mysql_type = 'datetime'
    elif python_type == time:
        mysql_type = 'time'
    else:
        mysql_type = 'text'
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


def create_sparse_matrix(filename, num_users=user_num, num_items=ws_num):
    user_ls, ws_ls, rt = load_data(filename)
    array_obj = np.zeros((num_users, num_items))
    array_obj[user_ls, ws_ls] = rt
    return array_obj


def send_email(receiver, title, text, mail_host=None, mail_user=None, mail_pass=None, **kwargs):
    if 'mail_host' in kwargs:
        mail_host = kwargs['mail_host']
    if 'mail_user' in kwargs:
        mail_user = kwargs['mail_user']
    if 'mail_pass' in kwargs:
        mail_pass = kwargs['mail_pass']
    if mail_pass is None or mail_user is None or mail_host is None:
        return
    # 第三方 SMTP 服务
    sender = mail_user

    message = MIMEText(text, 'plain', 'utf-8')
    subject = title
    message['Subject'] = Header(subject, 'utf-8')
    message['from'] = sender
    message['to'] = receiver

    try:
        smtp_obj = smtplib.SMTP_SSL(host=mail_host)
        smtp_obj.connect(mail_host, smtplib.SMTP_SSL_PORT)
        smtp_obj.login(mail_user, mail_pass)
        smtp_obj.sendmail(sender, receiver, message.as_string())
        print("邮件发送成功")
        smtp_obj.quit()
    except smtplib.SMTPException as e:
        print(e)
        print("Error: 无法发送邮件")


def get_table_count(config: dict, table: str):
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    # check table
    table_query = "select max(`table_no`) from `{}`".format(table)
    cursor.execute(table_query)
    count = list(cursor)[0]
    cursor.close()
    cnx.close()
    return count


def query_table_where_table_no_greater_than(config: dict, table: str, table_no: int):
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    query = "select * from `{}` where `table_no` > {}".format(table, table_no)
    cursor.execute(query)
    cursor.column_names

    cursor.close()
    cnx.close()


if __name__ == '__main__':
    print(mysql.connector.version.VERSION)
    from sensitive_info import database_config

    i = get_table_count(database_config, 'ncf_rt')
    print(i)
