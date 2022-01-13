

class Normalization:
    # def __init__(self, matrix, cb, weights):
    #     if matrix.shape[1] != len(weights) or matrix.shape[1] != len(cb) or len(cb) != len(weights):
    #         raise ValueError(f'Data shape, cost-benefit vector or weights vector does not match')
    #     self.matrix = matrix
    #     self.cb = cb
    #     self.weights = weights

    def __init__(self, matrix, cb):
        if matrix.shape[1] != len(cb):
            raise ValueError(f'Data shape and cost-benefit vector does not match')
        self.matrix = matrix
        self.cb = cb

    def normalization_zero_one(self):
        result = self.matrix.copy()
        pos = 0
        for col in self.matrix.columns:
            max_value = self.matrix[col].max()
            min_value = self.matrix[col].min()
            dif_max_min = self.matrix[col].max() - self.matrix[col].min()
            if self.cb[pos] == 'benefit':
                result[col] = (self.matrix[col] - min_value) / dif_max_min
            else:
                result[col] = (max_value - self.matrix[col]) / dif_max_min
            pos += 1
        return result.round(4)
