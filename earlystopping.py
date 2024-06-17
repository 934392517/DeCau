import torch

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        '''
        用于在训练过程中实现早停功能
        :param mode: 评估指标的模式，min max分别表示指标越小越好/越大越好
        :param min_delta: 评估指标的最小变化量，用于判断是否达到可接受的改善阈值
        :param patience: 指定连续几个训练轮次内没有性能改善时停止训练
        :param percentage: 是否将 min_delta 解释为百分比而不是绝对值
        '''
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta   # 指标a是否小于历史最佳指标best-最小变化量
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta  # 指标时候大于历史最佳+最小变化量
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)