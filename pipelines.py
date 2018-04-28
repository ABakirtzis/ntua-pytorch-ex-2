import math
import sys

import torch

from torch.autograd import Variable


def progress(loss, epoch, batch, batch_size, dataset_size):
    """
    Print the progress of the training for an epoch
    """
    batches = math.ceil(float(dataset_size) / batch_size)
    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    status = 'Epoch {}, Loss: {:.4f}'.format(epoch, loss)
    _progress_str = "\r \r [{}] ...{}".format(bar, status)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()

    if batch == batches:
        print()


def train_dataset(_epoch, dataloader, model, loss_function, optimizer):
    # switch to train mode -> enable regularization layers, such as Dropout
    model.train()
    running_loss = 0.0

    for index, batch in enumerate(dataloader, 1):

        # get the inputs (batch)
        inputs, labels, lengths = batch

        inputs = Variable(inputs)
        labels = Variable(labels)
        lengths = Variable(lengths)

        # convert to CUDA Variables
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
            lengths = lengths.cuda()

        # 1 - zero the gradients
        optimizer.zero_grad()

        # 2 - forward pass: compute predicted y by passing x to the model
        outputs = model(inputs, lengths)

        # 3 - compute loss
        loss = loss_function(outputs, labels)

        # 4 - backward pass: compute gradient wrt model parameters
        loss.backward()

        # 5 - update weights
        optimizer.step()

        running_loss += loss.data[0]

        # print statistics
        progress(loss=loss.data[0],
                 epoch=_epoch,
                 batch=index,
                 batch_size=dataloader.batch_size,
                 dataset_size=len(dataloader.dataset))

    return running_loss / index


def eval_dataset(dataloader, model, loss_function):
    # switch to eval mode -> disable regularization layers, such as Dropout
    model.eval()

    y_pred = []  # the predicted labels
    y = []  # the gold labels

    total_loss = 0
    for index, batch in enumerate(dataloader, 1):
        # get the inputs (batch)
        inputs, labels, lengths = batch

        inputs = Variable(inputs)
        labels = Variable(labels)
        lengths = Variable(lengths)

        # convert to CUDA Variables
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
            lengths = lengths.cuda()
        outputs = model(inputs, lengths)

        loss = loss_function(outputs, labels)
        total_loss += loss.data[0]

        _, predicted = torch.max(outputs.data, 1)

        # append the predictions and the gold labels, to the placeholders
        y.extend(list(labels.data.cpu().numpy().squeeze()))
        y_pred.extend(list(predicted.squeeze()))

    avg_loss = total_loss / index

    return avg_loss, (y, y_pred)
