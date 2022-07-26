import numpy as np
from scipy import ndimage

def create_shifted_dataset(x):
    x_shifted_arr = []
    for i in range(x.shape[0]):
        row_shift = np.random.randint(1,4)
        col_shift = np.random.randint(1,4)
        x_shift = ndimage.shift(x[i], shift=[row_shift, col_shift,0])
        x_shifted_arr.append(x_shift)
    x_shifted_arr = np.array(x_shifted_arr)  
    return x_shifted_arr

def compute_consistency(model, x, x_shifted):
    same_pred = 0
    predictions_unshifted = model.predict(x)
    predictions_shifted = model.predict(x_shifted)
    for i in range(predictions_unshifted.shape[0]):
        one_instance_pred_unshifted = predictions_unshifted[i]
        one_instance_pred_shifted = predictions_shifted[i]
        pred_unshifted_max_prob = np.argmax(one_instance_pred_unshifted)
        pred_shifted_max_prob = np.argmax(one_instance_pred_shifted)
        if pred_unshifted_max_prob == pred_shifted_max_prob:
            same_pred +=1
    consistency = same_pred / x.shape[0]
    return consistency
