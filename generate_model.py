from tensorflow.keras.models import save_model
from architecture.resnet_aps import build_resnet_20_aps

def save_untrained_model(shape):
    aps_resnet = build_resnet_20_aps(shape)
    aps_resnet.summary()
    save_model(aps_resnet, "./models/aps_resnet_untrained.h5")

if __name__ == "__main__":
    save_untrained_model((32,32,3))
