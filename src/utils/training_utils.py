"""
class LearningRateManager():
    def __init__(self, learning_rate=.001, decay=.1,
                 step_values=[160, 190, 220]):
        self.decay = decay
        self.learning_rate = learning_rate
        self.step_values = step_values

    def schedule(self, epoch):
        if epoch in self.step_values:
            self.learning_rate = self.learning_rate * .1
        return self.learning_rate * (self.decay**(epoch))
"""

"""
class LearningRateManager():
    def __init__(self, learning_rate=.001, decay=.1,
                 step_values=[160, 190, 220]):
        self.learning_rate = learning_rate
        self.decay = decay
        self.step_values = step_values

    def schedule(self, epoch):
        if epoch in self.step_values:
            self.learning_rate = self.learning_rate * self.decay
        return self.learning_rate

learning_rate_manager = LearningRateManager()
for epoch in range(250):
    print(epoch)
    print('sch', learning_rate_manager.schedule(epoch))
    print('lr', learning_rate_manager.learning_rate)

class LearningRateManager():
    def __init__(self, learning_rate, decay=0.98):
        self.decay = decay
        self.learning_rate = learning_rate

    def schedule(self, epoch):
        return self.learning_rate * (self.decay**(epoch))

"""


class LearningRateManager():
    def __init__(self, learning_rate, decay=0.1, step_epochs=None):
        self.decay = decay
        self.learning_rate = learning_rate
        self.step_epochs = step_epochs

    def schedule(self, epoch):
        if epoch in self.step_epochs:
            self.learning_rate = self.learning_rate * self.decay
        return self.learning_rate


"""

lr_manager = LearningRateManager(1e-3, .1, [80, 100, 120])
for epoch in range(140):
    print(epoch, lr_manager.schedule(epoch))
"""
