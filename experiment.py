import os
import experiment.extend_mlp_model_exp
from experiment_structure import ExperimentData
from universal_method import send_email
from sensitive_info import email_config
from datetime import datetime
import experiment.mlp_exp
import experiment.ncf_exp
import experiment.gmf_exp


def gmf_exp():
    for s in [1, 3]:
        for i in [1, 2, 3, 4, 5]:
            for a in ['relu']:
                for d in [128]:
                    for e in [0, 1, 2, 3, 4, 5, 10, 15, 20]:
                        exp_data = {
                            'sparseness': s,
                            'index': i,
                            'epochs': 30,
                            'batch_size': 128,
                            'mf_dim': d,
                            'regs': [0, 0],
                            'last_activation': a,
                            'learning_rate': 0.007,
                            'extend_near_num': e,
                            'learner': 'adagrad'
                        }
                        experiment.gmf_exp.experiment(**exp_data)
    send_email(receiver='haoran.x@outlook.com',
               title='GMF实验结束',
               text="实验结束时间：{}".format(datetime.now()),
               **email_config)


def extend_mlp_exp():
    for s in [5]:
        for i in [1, 3, 4, 5]:
            for a in ['relu']:
                for e in [1, 2, 3, 4, 5, 10, 15]:
                    exp_data = {
                        'sparseness': s,
                        'index': i,
                        'epochs': 50,
                        'batch_size': 256,
                        'layers': [64, 32, 16, 8],
                        'reg_layers': [0, 0, 0, 0],
                        'fake_layers': [64, 32, 16, 8],
                        'fake_reg_layers': [0, 0, 0, 0],
                        'last_activation': a,
                        'fake_last_activation': a,
                        'learning_rate': 0.001,
                        'extend_near_num': e,
                        'learner': 'adagrad'
                    }
                    experiment.extend_mlp_model_exp.experiment(**exp_data)
    send_email(receiver='haoran.x@outlook.com',
               title='EXMLP实验结束',
               text="实验结束时间：{}".format(datetime.now()),
               **email_config)


def mlp_exp():
    for s in [1, 3]:
        for i in [1, 2, 3, 4, 5]:
            for a in ['relu']:
                for e in [0, 1, 2, 3, 4, 5, 10, 15, 20]:
                    exp_data = {
                        'sparseness': s,
                        'index': i,
                        'epochs': 30,
                        'batch_size': 128,
                        'layers': [64, 32, 16],
                        'reg_layers': [0, 0, 0],
                        'last_activation': a,
                        'learning_rate': 0.007,
                        'extend_near_num': e,
                        'learner': 'adagrad'
                    }
                    experiment.mlp_exp.experiment(**exp_data)
    send_email(receiver='haoran.x@outlook.com',
               title='MLP实验结束',
               text="",
               **email_config)


def ncf_exp():
    for s in [1, 3]:
        for i in [1, 2, 3, 4, 5]:
            for e in [0, 1, 2, 3, 4, 5, 10, 15, 20]:
                data = ExperimentData()
                data.sparseness = s
                data.data_index = i
                data.mf_dim = 8
                data.epochs = 30
                data.batch_size = 128
                data.layers = [64, 32, 16]
                data.reg_layers = [0, 0, 0]
                data.learning_rate = 0.007
                data.extend_near_num = e
                data.learner = 'adagrad'
                experiment.ncf_exp.experiment(data, 'relu')

    send_email(receiver='haoran.x@outlook.com',
               title='NCF实验结束',
               text="",
               **email_config)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gmf_exp()
    mlp_exp()
    ncf_exp()
