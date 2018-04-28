import os
import pickle

from torch.utils.data import Dataset
from tqdm import tqdm

from config import BASE_PATH
from utils.nlp import tokenize, vectorize


class SentenceDataset(Dataset):
    """
    A PyTorch Dataset
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...
        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """

    def __init__(self, X, y, word2idx, name=None, max_length=0):
        """
        Args:
            X (): List of training samples
            y (): List of training labels
            word2idx (dict): a dictionary which maps words to indexes
            name (str): the name of the dataset. It is needed for caching.
                if None then caching is disabled.
            max_length (int): the max length for each sentence.
                if 0 then use the maximum length in the dataset
        """

        self.data = X
        self.labels = y
        self.word2idx = word2idx
        self.name = name
        self.data = self.preprocess_dataset()

        if max_length == 0:
            self.max_length = max([len(x) for x in self.data])
        else:
            self.max_length = max_length

    @staticmethod
    def preprocess(name, dataset):
        desc = "PreProcessing dataset {}...".format(name)

        data = [tokenize(x) for x in tqdm(dataset, desc=desc)]
        return data

    def __get_cache_filename(self):
        return os.path.join(BASE_PATH, "_cache", "{}.p".format(self.name))

    @staticmethod
    def __check_cache():
        """
        Check if the cache dir exists and if not the create it
        """
        cache_dir = os.path.join(BASE_PATH, "_cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def __write_cache(self, data):
        """
        Write the preprocessed data to a cache file
        """
        self.__check_cache()
        cache_file = self.__get_cache_filename()
        with open(cache_file, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)

    def preprocess_dataset(self):
        # NOT using cache
        if self.name is None:
            print("cache deactivated!")
            return self.preprocess(self.name, self.data)

        # using cache
        cache_file = self.__get_cache_filename()

        if os.path.exists(cache_file):
            print("Loading {} from cache!".format(self.name))
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            print("No cache file for {} ...".format(self.name))
            data = self.preprocess(self.name, self.data)
            self.__write_cache(data)
            return data

    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches

        Returns:
            (int): the length of the dataset
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (int): the class label
                * length (int): the length (tokens) of the sentence

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['this', 'is', 'really', 'simple']
                self.target[index] = "neutral"

            the function will have to return return:
            ::
                example = [  533  3908  1387   649   0     0     0     0
                             0     0     0     0     0     0     0     0
                             0     0     0     0     0     0     0     0]
                label = 1
        """
        sample, label = self.data[index], self.labels[index]
        length = min(self.max_length, len(sample))
        sample = vectorize(sample, self.word2idx, self.max_length)
        return sample, label, length
