def scheduler(epoch, decay=0.1, base_learning_rate=3e-3):
    return base_learning_rate * decay**(epoch)


class Scheduler():
    def __init__(self, scheduled_epochs=[80, 100, 120], decay=0.1,
                 base_learning_rate=3e-3):
        self.scheduled_epochs = scheduled_epochs
        self.decay = decay
        self.base_learning_rate = base_learning_rate

    def schedule(self, epoch):
        if epoch in self.scheduled_epochs:
            self.base_learning_rate = (self.base_learning_rate *
                                       (self.decay**(epoch)))
        return self.base_learning_rate


def split_data(ground_truths, training_ratio=.8):
    ground_truth_keys = sorted(ground_truths.keys())
    num_train = int(round(training_ratio * len(ground_truth_keys)))
    train_keys = ground_truth_keys[:num_train]
    validation_keys = ground_truth_keys[num_train:]
    return train_keys, validation_keys
