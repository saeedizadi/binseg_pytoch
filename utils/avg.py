class AverageMeter(object):
    def __init__(self):
        self.val = 0.0
        self.count = 0.0
        self.sum = 0.0
        self.avg = 0.0

    def reset(self) :
        self.val = 0.0
        self.count = 0.0
        self.sum = 0.0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val= float(val)
        self.count += n
        self.sum += n*self.val
        self.avg = self.sum / self.count
