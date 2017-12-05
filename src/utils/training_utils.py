from keras.callbacks import Callback

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


class MultiGPUModelCheckpoint(Callback):
    def __init__(self, save_path, cpu_model):
        self.save_path = save_path
        self.cpu_model = cpu_model

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        self.cpu_model.save(self.save_path.format(str(epoch), str(val_loss)))
