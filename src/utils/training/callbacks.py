from keras.callbacks import Callback


class MultiGPUModelCheckpoint(Callback):
    def __init__(self, save_path, cpu_model):
        self.save_path = save_path
        self.cpu_model = cpu_model

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        self.cpu_model.save(self.save_path.format(epoch, val_loss))


class LearningRateManager():
    def __init__(self, learning_rate, gamma_decay, scheduled_epochs):
        self.learning_rate = learning_rate
        self.gamma_decay = gamma_decay
        self.scheduled_epochs = scheduled_epochs

    def schedule(self, epoch):
        if epoch in self.scheduled_epochs:
            self.learning_rate = self.learning_rate * self.gamma_decay
        return self.learning_rate


def scheduler(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001
