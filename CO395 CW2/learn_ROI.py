import numpy as np

from nn_lib import (
MultiLayerNetwork,
Trainer,
Preprocessor,
save_network,
load_network,
)

from illustrate import illustrate_results_ROI


def main():
    dataset = np.loadtxt("ROI_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    np.random.shuffle(dataset)

    x = dataset[:, :3]
    y = dataset[:, 3:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep = Preprocessor(x_train)

    x_train_pre = prep.apply(x_train)
    x_val_pre = prep.apply(x_val)

    np.random.seed(10)
    input_dim = 3
    neurons = [32,32,256,128,4]
    activations = ["relu","relu","relu","relu","identity"]
    net2 = MultiLayerNetwork(input_dim, neurons, activations)
    trainer = Trainer(
        network=net2,
        batch_size=32,
        nb_epoch=200,
        learning_rate=0.073,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )
    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net2(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))

    save_network(net2,'saved_network_3.pt')
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    illustrate_results_ROI(net2, prep)

def predict_hidden(dataset):
    prep = Preprocessor(dataset)
    dataset=prep.apply(dataset)
    x_test= dataset[:,:3]
    fpath = 'saved_network_3.pt'
    net = load_network(fpath)
    pred = net(x_test).argmax(axis=1)
    pred= np.eye(4)[pred]
    return pred

if __name__ == "__main__":
    main()
