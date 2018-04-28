import torch

from torch import nn

from modules.layers import GaussianNoise


class BaselineModel(nn.Module):
    def __init__(self, embeddings, out_size, **kwargs):
        """
        Define the layers and initialize them.

        Pytorch initializes the layers by default, with random weights,
        sampled from certain distribution. However, in some cases
        you might want to explicitly initialize some layers,
        either by sampling from a different distribution,
        or by using pretrained weights (word embeddings / transfer learning)

        Args:
            embeddings (numpy.ndarray): the 2D ndarray with the word vectors
            nclasses ():
        """
        super(BaselineModel, self).__init__()

        ########################################################
        # Optional Parameters
        ########################################################
        rnn_size = kwargs.get("rnn_size", 100)
        rnn_layers = kwargs.get("layers", 1)
        bidirectional = kwargs.get("rnn_bidirectional", False)
        noise = kwargs.get("emb_noise", 0.)
        dropout_words = kwargs.get("emb_dropout", 0.2)
        dropout_rnn = kwargs.get("rnn_dropout", 0.2)
        trainable_emb = kwargs.get("trainable_emb", False)
        ########################################################

        # define the embedding layer, with the corresponding dimensions
        self.embedding = nn.Embedding(num_embeddings=embeddings.shape[0],
                                      embedding_dim=embeddings.shape[1])

        # initialize the weights of the Embedding layer,
        # with the given pre-trained word vectors
        self.init_embeddings(embeddings, trainable_emb)

        # the dropout "layer" for the word embeddings
        self.drop_emb = nn.Dropout(dropout_words)
        # the gaussian noise "layer" for the word embeddings
        self.noise_emb = GaussianNoise(noise)

        # the RNN layer (or layers)
        self.rnn = nn.LSTM(input_size=embeddings.shape[1],
                           hidden_size=rnn_size,
                           num_layers=rnn_layers,
                           bidirectional=bidirectional,
                           dropout=dropout_rnn,
                           batch_first=True)

        # the dropout "layer" for the output of the RNN
        self.drop_rnn = nn.Dropout(dropout_rnn)

        if self.rnn.bidirectional:
            rnn_size *= 2

        # the final Linear layer which maps the representation of the sentence,
        # to the classes
        self.linear = nn.Linear(in_features=rnn_size, out_features=out_size)

    def init_embeddings(self, weights, trainable):
        self.embedding.weight = nn.Parameter(torch.from_numpy(weights),
                                             requires_grad=trainable)

    def last_timestep(self, unpacked, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(unpacked.size(0),
                                               unpacked.size(2)).unsqueeze(1)
        return unpacked.gather(1, idx).squeeze()

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        embs = self.embedding(x)
        embs = self.noise_emb(embs)
        embs = self.drop_emb(embs)

        outputs, _ = self.rnn(embs)

        # get the outputs from the last *non-masked* timestep for each sentence
        last_outputs = self.last_timestep(outputs, lengths)
        # last_outputs = self.last_timestep(self.rnn, h)

        # apply dropout to the outputs of the RNN
        last_outputs = self.drop_rnn(last_outputs)

        # project to the classes using a linear layer
        # Important: we do not apply a softmax on the logits, because we use
        # CrossEntropyLoss() as our loss function, which applies the softmax
        # and computes the loss.
        logits = self.linear(last_outputs)

        return logits
