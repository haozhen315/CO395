import numpy as np

from nn_lib import (
MultiLayerNetwork,
Trainer,
Preprocessor,
save_network,
load_network,
)
from illustrate import illustrate_results_FM


def main():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    np.random.shuffle(dataset)

    prep = Preprocessor(dataset)
    dataset = prep.apply(dataset)
    split_idx = int(0.8 * len(dataset))

    x_train_pre= dataset[:split_idx,:3]
    y_train_pre = dataset[:split_idx,3:]
    x_val_pre = dataset[split_idx:,:3]
    y_val_pre = dataset[split_idx:,3:]
    np.random.seed(10)
    input_dim = 3
    neurons = [512,128,32,3]
    activations = ["relu","relu","relu","identity"]
    network = MultiLayerNetwork(input_dim, neurons, activations)
    trainer = Trainer(
    network=network,
    batch_size=8,
    nb_epoch=100,
    learning_rate=0.01,
    loss_fun="mse",
    shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train_pre)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train_pre))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val_pre))

    # R^2 metric
    y_pred_pre = network(x_val_pre)
    u =((y_val_pre - y_pred_pre) ** 2).sum()
    v =((y_val_pre - y_val_pre.mean(axis=0)) ** 2).sum()
    R2 = 1 - u/v

    # MAE metric
    mae =(np.abs(y_pred_pre-y_val_pre)).mean()
    print("Validation Score MAE: {}".format(mae))
    print("Validation Score R2: {}".format(R2))

    save_network(network,'saved_network_2.pt')
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    illustrate_results_FM(network, prep)

def predict_hidden(dataset):
    prep = Preprocessor(dataset)
    dataset=prep.apply(dataset)
    fpath = 'saved_network_2.pt'
    net = load_network(fpath)
    pred = net(dataset[:,:3])
    dataset[:,3:] =pred
    dataset = prep.revert(dataset)
    out = dataset[:,3:]
    return out

if __name__ == "__main__":
    main()
