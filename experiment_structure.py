class ExperimentData(object):
    def __init__(self):
        self.sparseness = 0
        self.data_index = 0
        self.mf_dim = 0
        self.epochs = 0
        self.batch_size = 0
        self.layers = []
        self.reg_layers = []
        self.learning_rate = 0.0
        self.learner = ''
        self.extend_near_num = 0
        self.loss = 0.0
        self.mae = 0.0
        self.rmse = 0.0
        self.model = ''

    def to_dict(self) -> dict:
        return {
            'sparseness': self.sparseness,
            'data_index': self.data_index,
            'mf_dim': self.mf_dim,
            'epochs': self.epochs,
            'batch_size':
                self.batch_size,
            'layers':
                str(self.layers),
            'reg_layers':
                str(self.reg_layers),
            'learning_rate':
                self.learning_rate,
            'learner':
                self.learner,
            'extend_near_num':
                self.extend_near_num,
            'loss':
                self.loss,
            'mae':
                self.mae,
            'rmse':
                self.rmse,
            'model':
                self.model
        }
