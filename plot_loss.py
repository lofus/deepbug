import keras
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow

def plot_3d_matrice(mat):
    plt.imshow(mat)
    plt.colorbar()
    plt.show()

# A class works to plot the loss while training.
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        #clear_output(wait=True)
        if self.i % 50 == 0:
            plt.plot(self.x, self.losses, label="loss")
            plt.plot(self.x, self.val_losses, label="val_loss")
            plt.legend()
            plt.show()
            filename = "loss_epoch" + str(self.i) + ".png"
            plt.savefig(filename)

