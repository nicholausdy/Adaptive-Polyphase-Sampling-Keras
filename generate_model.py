from tensorflow.keras.models import save_model
from architecture.resnet_aps import build_resnet_20_aps


def save_untrained_model(shape):

    aps_resnet = build_resnet_20_aps(shape)
    print(aps_resnet.summary())
    save_model(aps_resnet, "./models/aps_resnet_untrained.h5")

    return aps_resnet


if __name__ == "__main__":

    # Create the model
    model = save_untrained_model((32,32,3))

    # Create a test tensor with the form (batch_size, height, width, channels)
    x_test = tf.random.uniform((1, 32, 32, 1))
    y_orig = model(x_test)

    # Create a shifted tensor
    img_roll = tf.roll(x_test, shift=[1, 1], axis=[-1, -2])
    y_roll = model(img_roll)

    print(f"y_orig : {y_orig.numpy()}")
    print(f"y_roll : {y_roll.numpy()}")

    # Displacement invariance check
    assert tf.reduce_all(tf.abs(y_orig - y_roll) < 1e-7), "They are not the same"
    print("\nNorm(y_orig - y_roll): %e " % tf.norm(y_orig-y_roll).numpy())