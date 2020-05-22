import math
from multiprocessing import Pool
from time import time

import numpy as np

from sensitive_info import email_config
from universal_method import load_csv_file, load_data, save_csv_file, save_data, send_email


def calculate_distance(file) -> list:
    data = np.loadtxt(file)
    users_dict = sorted_data(sort_user(data))
    items_dict = sorted_data(sort_item(data))
    users = list(users_dict.keys())
    U = len(users)

    lambda_s = {}
    for k, v in items_dict.items():
        lambda_s[k] = math.log(U / len(v))

    result = []
    for i in range(U):
        for j in range(i):
            u = users[i]
            v = users[j]
            d_u_v = distance(users_dict[u], users_dict[v], lambda_s)
            result.append([u, v, d_u_v])
    return result


def sort_user(data) -> map:
    data = sorted(data, key=lambda x: x[0])
    data = map(lambda x: [int(x[0]), [int(x[1]), x[2]]], data)
    return data


def sort_item(data) -> map:
    data = sorted(data, key=lambda x: x[1])
    data = map(lambda x: [int(x[1]), [int(x[0]), x[2]]], data)
    return data


def sorted_data(data) -> dict:
    data_dict = {}
    first = next(data)
    index = first[0]
    cur_list = [first[1]]
    for i in data:
        if index == i[0]:
            cur_list.append(i[1])
        else:
            data_dict[index] = dict(cur_list)
            index = i[0]
            cur_list = [i[1]]
    data_dict[index] = dict(cur_list)
    return data_dict


def distance(u: dict, v: dict, lambda_s: dict) -> float:
    u_len = len(u)
    v_len = len(v)
    u_mean = sum(u.values()) / u_len
    v_mean = sum(u.values()) / v_len
    u_set = set(u.keys())
    v_set = set(v.keys())
    m_set = u_set & v_set
    if len(m_set) == 0:
        return 0.0
    u_array = np.array(list(map(lambda x: u[x], m_set)))
    v_array = np.array(list(map(lambda x: v[x], m_set)))
    lambda_array = np.array(list(map(lambda x: lambda_s[x], m_set)))
    d = np.sqrt(np.mean(np.square((u_array - u_mean) - (v_array - v_mean)) * lambda_array))
    # return 1.0 / (1.0 + d)
    return d


def convert_distance_result(result: list) -> dict:
    result_dict = {}
    for i in result:
        u, v, d = i
        if u in result_dict:
            result_dict[u].append([v, d])
        else:
            result_dict[u] = [[v, d]]
        if v in result_dict:
            result_dict[v].append([u, d])
        else:
            result_dict[v] = [[u, d]]
    result_dict.update(map(lambda k: (k, sorted(result_dict[k], key=lambda x: x[1])), result_dict.keys()))
    return result_dict


def extend_array(times: int, distance: dict, user_id: np.ndarray, item_id: np.ndarray, rating: np.ndarray):
    users = np.array(list(map(lambda x: np.append(x, np.array(distance[x])[:times][:, 0]), user_id)),
                     dtype=int).flatten()
    extend_times = list(map(lambda x: len(distance[x][:times]) + 1, user_id))
    items = np.array(list(map(lambda x, y: [x] * y, item_id, extend_times)), dtype=int).flatten()
    ratings = np.array(list(map(lambda x, y: [x] * y, rating, extend_times)), dtype=float).flatten()
    return users, items, ratings


def extend_array_and_save(sparseness, index, extend_near_num, distance, userId, itemId, rating):
    u, i, r = extend_array(extend_near_num, distance, userId, itemId, rating)
    save_data(save_csv_file(sparseness, index, extend_near_num), u, i, r)


def save_extend_array(sparseness, index, extend_near_nums):
    userId, itemId, rating = load_data(load_csv_file(sparseness, index))
    result = calculate_distance(load_csv_file(sparseness, index))
    distance = convert_distance_result(result)
    for e in extend_near_nums:
        extend_array_and_save(sparseness, index, e, distance, userId, itemId, rating)


def main_fork():
    with Pool() as pool:
        extends = (1, 2, 3, 4, 5, 10, 15, 20)
        # extends = (1, 2)
        # pool.map_async(save_extend_array, [(s, i, extends, pool) for s in [5, 10, 15, 20] for i in range(1, 6)])
        pool.starmap(save_extend_array, [(s, i, extends) for s in [5, 10, 15, 20] for i in range(1, 6)])

    send_email(receiver='haoran.x@outlook.com',
               title='实验结束',
               text="",
               **email_config)


def main_1():
    sparseness = 5
    index = 1
    extend_near_num = 1
    userId, itemId, rating = load_data(load_csv_file(sparseness, index))
    result = calculate_distance(load_csv_file(sparseness, index))
    distance = convert_distance_result(result)

    file = save_csv_file(sparseness, index, extend_near_num)
    print(file)
    u, i, r = extend_array(extend_near_num, distance, userId, itemId, rating)
    save_data(file, u, i, r)


def main():
    t1 = time()
    r = calculate_distance(load_csv_file(5, 1))
    c_r = convert_distance_result(r)
    print(time() - t1)
    for k, v in c_r.items():
        print(k, v[:3])


if __name__ == '__main__':
    main_fork()
