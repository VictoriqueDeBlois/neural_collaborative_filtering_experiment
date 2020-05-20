import mysql.connector
from sensitive_info import database_remote_config


def one_query(cursor, performance, approach, sparseness, index):
    query = 'select min({}) from {}_rt where sparseness = {} and `index` = {}'.format(performance,
                                                                                      approach,
                                                                                      sparseness,
                                                                                      index)
    query1 = 'select min({}) from {}_rt where sparseness = {} and `data_index` = {}'.format(performance,
                                                                                            approach,
                                                                                            sparseness,
                                                                                            index)
    try:
        cursor.execute(query)
    except mysql.connector.errors.ProgrammingError as e:
        cursor.execute(query1)
    for i in cursor:
        return str(i[0])


def one_query_plus(cursor, performance, approach, sparseness, index, extend):
    query = 'select min({}) from {}_rt where sparseness = {} and `index` = {} and extend_near_num = {}'.format(
        performance,
        approach,
        sparseness,
        index,
        extend)
    query1 = 'select min({}) from {}_rt where sparseness = {} and `data_index` = {} and extend_near_num = {}'.format(
        performance,
        approach,
        sparseness,
        index,
        extend)
    try:
        cursor.execute(query)
    except mysql.connector.errors.ProgrammingError as e:
        cursor.execute(query1)
    for i in cursor:
        return str(i[0])


def extend():
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    output = []

    for sparseness in [5, 10, 15, 20, 30]:
        for index in [1, 2, 3, 4, 5]:
            for performance in ['mae', 'rmse']:
                row = []
                for approach in ['gmf', 'ncf']:
                    for extend in [0, 1, 2, 3, 4, 5, 10, 15, 20]:
                        row.append(one_query_plus(cursor, performance, approach, sparseness, index, extend))
                output.append(','.join(row))
    out = '\n'.join(output)
    print(out)
    with open("./output.csv", 'w') as f:
        f.write(out)
    cursor.close()
    cnx.close()


if __name__ == '__main__':
    cnx = mysql.connector.connect(**database_remote_config)
    cursor = cnx.cursor()
    output = []

    for sparseness in [15]:
        for index in [1, 2, 3, 4, 5]:
            for performance in ['mae', 'rmse']:
                row = []
                for approach in ['gmf']:
                    for extend in [0, 1, 2, 3, 4, 5, 10, 15, 20]:
                        row.append(one_query_plus(cursor, performance, approach, sparseness, index, extend))
                output.append(','.join(row))
    out = '\n'.join(output)
    print(out)
    with open("./output.csv", 'w') as f:
        f.write(out)
    cursor.close()
    cnx.close()
