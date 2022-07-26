from tensorflow import keras
from tensorflow.keras.models import load_model
from architecture.aps import APSLayer, APSDownsampleGivenPolyIndices

def load_train_dataset():
    (x_train, y_train), (_,_) = keras.datasets.cifar10.load_data()
    return x_train, y_train

def train_model(model, x_train, y_train):
    save_callback = keras.callbacks.ModelCheckpoint(
        "./models/aps_resnet_trained.h5", 
        monitor = "val_accuracy",
        verbose = 1,
        save_freq = "epoch"
    )
    model.fit(
        x_train,
        y_train,
        batch_size = 256,
        epochs = 250,
        verbose = 1,
        callbacks = [save_callback],
        validation_split = 0.1
    )

if __name__ == "__main__":
    x_train, y_train = load_train_dataset()
    aps_resnet_untrained = load_model(
        "./models/aps_resnet_untrained.h5", 
        custom_objects={
            'APSLayer': APSLayer,
            'APSDownsampleGivenPolyIndices': APSDownsampleGivenPolyIndices
        }
    )
    train_model(aps_resnet_untrained, x_train, y_train)
