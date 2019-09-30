from tensorflow.keras.callbacks import Callback


class StopAtAccuracy(Callback):

    def __init__(self, validation_accuracy_threshold):
        super(StopAtAccuracy, self).__init__()
        self.validation_accuracy_threshold = validation_accuracy_threshold

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs['val_acc']
        if val_acc > self.validation_accuracy_threshold:
            print("Validation accuracy reached! {:4.1f} > {:4.1f}".format(val_acc, self.validation_accuracy_threshold))
            self.model.stop_training = True


class DebugCallback(Callback):

    def on_batch_end(self, epoch, logs=None):
        print(logs)
