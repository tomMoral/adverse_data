import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

# import the model
from importlib import import_module

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser('Programme to launch experiemnt')
    parser.add_argument('--exp', type=str, default='lista_gaussian',
                        help='Choose the experiement to run')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Choose the experiement to run')
    parser.add_argument('--gpu', type=float, default=.95,
                        help='Ratio of the GPU memory used by this program')

    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu
    set_session(tf.Session(config=config))

    try:
        exp = import_module('experiments.{}'.format(args.exp))
    except AttributeError:
        raise ValueError("experiment \"{}\" not found".format(args.exp))

    model = exp.model

    if hasattr(exp, "data"):
        # Keep 30% of the data for validation
        N_train = int(.3 * exp.N)
        x_train, x_test = exp.data[:N_train], exp.data[N_train:]
        y_train, y_test = exp.labels[:N_train], exp.labels[N_train:]

        # Baseline
        y_raw_test, y_pred_test = exp.y[:N_train], exp.y_pred[:N_train]
        acc = np.mean(y_raw_test == y_pred_test)
        print(f"Baseline: {acc}")

        # Train the model, iterating on the data in batches of 32 samples
        history = model.fit(exp.data, exp.labels, epochs=args.epochs,
                            batch_size=256, validation_split=.3, shuffle=True)
    else:
        # Train the model, iterating on the data in batches of 32 samples
        steps = exp.N // exp.batch_size
        steps_test = exp.N_test // exp.batch_size
        history = model.fit_generator(exp.generator(), steps,
                                      epochs=args.epochs,
                                      validation_data=exp.generator(),
                                      validation_steps=steps_test, workers=1)

    import IPython
    IPython.embed()
    plt.figure(args.exp)
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.hlines(acc, 0, args.epochs, linestyle="--")
    plt.title('model loss')
    plt.ylabel('loss')

    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f"{args.exp}_lay{exp.N_layers}_cost.pdf", dpi=150)
    plt.show()
