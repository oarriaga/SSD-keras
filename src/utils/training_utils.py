class LearningRateManager():
    def __init__(self, decay=0.94, learning_rate=3e-3):
        self.decay = decay
        self.learning_rate = learning_rate

    def schedule(self, epoch):
        return self.learning_rate * (self.decay**(epoch))
