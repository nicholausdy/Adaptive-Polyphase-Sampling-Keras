from tensorflow import keras
from tensorflow.keras.models import load_model
from architecture.aps import APSLayer, APSDownsampleGivenPolyIndices
from util_eval import create_shifted_dataset, compute_consistency

def load_and_shift_test_dataset():
    (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_test_shifted = create_shifted_dataset(x_test)
    return x_test, x_test_shifted, y_test

def test_consistency(model, x, x_shifted):
    # test consistency between unshifted and shifted test dataset
    consistency = compute_consistency(model, x, x_shifted)
    print("Consistency after training (from 0.0 to 1.0): ", consistency)

if __name__ == "__main__":
    x_test, x_test_shifted, y_test = load_and_shift_test_dataset()
    aps_resnet_trained = load_model(
        "./models/aps_resnet_trained.h5", 
        custom_objects={
            'APSLayer': APSLayer,
            'APSDownsampleGivenPolyIndices': APSDownsampleGivenPolyIndices
        }
    )
    test_consistency(aps_resnet_trained, x_test, x_test_shifted)
    aps_resnet_trained.evaluate(x_test, y_test, batch_size=100)
    