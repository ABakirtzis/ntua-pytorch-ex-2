# download from http://nlp.stanford.edu/data/glove.twitter.27B.zip
# WORD_VECTORS = "../embeddings/glove.twitter.27B.50d.txt"
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from modules.dataloaders import SentenceDataset
from modules.models import BaselineModel
from pipelines import train_dataset, eval_dataset
from utils.load_data import load_semeval2017A
from utils.load_embeddings import load_word_vectors

########################################################
# PARAMETERS
########################################################

EMBEDDINGS = "../embeddings/glove.twitter.27B.50d.txt"
EMB_DIM = 50
BATCH_SIZE = 128
EPOCHS = 50

config = {
    "emb_dropout": 0.1,
    "emb_noise": 0.1,
    "trainable_emb": False,

    "layers": 1,
    "rnn_size": 100,
    "rnn_bidirectional": False,
    "rnn_dropout": 0.2,
}
########################################################
# Define the datasets/dataloaders
########################################################

# 1 - load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# 2 - load the raw data
train = load_semeval2017A("datasets/Semeval2017A/train_dev")
val = load_semeval2017A("datasets/Semeval2017A/gold")

X_train = [x[1] for x in train]
y_train = [x[0] for x in train]
X_val = [x[1] for x in val]
y_val = [x[0] for x in val]

# 3 - convert labels from strings to integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_train)

y_train = label_encoder.transform(y_train)
y_val = label_encoder.transform(y_val)

# 4 - define the datasets
train_set = SentenceDataset(X_train, y_train, word2idx, name="train")
val_set = SentenceDataset(X_val, y_val, word2idx, name="val")

# 5 - define the dataloaders
loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
loader_val = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
classes = label_encoder.classes_.size
model = BaselineModel(embeddings=embeddings, out_size=classes, **config)

print(model)
if torch.cuda.is_available():
    model.cuda()

criterion = torch.nn.CrossEntropyLoss()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters)


#############################################################################
# Training Pipeline
#############################################################################
def acc(y, y_hat):
    return accuracy_score(y, y_hat)


def f1(y, y_hat):
    return f1_score(y, y_hat, average='macro')


for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, loader_train, model, criterion, optimizer)

    # evaluate the performance of the model, on both data sets
    avg_train_loss, (y, y_pred) = eval_dataset(loader_train, model, criterion)
    print("\tTrain: loss={:.4f}, acc={:.4f}, f1={:.4f}".format(avg_train_loss,
                                                               acc(y, y_pred),
                                                               f1(y, y_pred)))

    avg_val_loss, (y, y_pred) = eval_dataset(loader_val, model, criterion)
    print("\tTest:  loss={:.4f}, acc={:.4f}, f1={:.4f}".format(avg_val_loss,
                                                               acc(y, y_pred),
                                                               f1(y, y_pred)))
