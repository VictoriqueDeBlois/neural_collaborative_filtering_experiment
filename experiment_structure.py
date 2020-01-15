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
            'sparseness': int(self.sparseness),
            'data_index': int(self.data_index),
            'mf_dim': int(self.mf_dim),
            'epochs': int(self.epochs),
            'batch_size':
                int(self.batch_size),
            'layers':
                str(self.layers),
            'reg_layers':
                str(self.reg_layers),
            'learning_rate':
                float(self.learning_rate),
            'learner':
                str(self.learner),
            'extend_near_num':
                int(self.extend_near_num),
            'loss':
                float(self.loss),
            'mae':
                float(self.mae),
            'rmse':
                float(self.rmse),
            'model':
                str(self.model)
        }
