import numpy


def tokenize(text, lowercase=True):
    """
    no need to get funky. use a stupid simple tokenizer.
    this is not the point of the assignment.
    but if you want, feel free to change it.
    """
    if lowercase:
        text = text.lower()
        
    return text.split()


def vectorize(text, word2idx, max_length):
    """
    Covert array of tokens, to array of ids, with a fixed length
    and zero padding at the end
    Args:
        text (): the wordlist
        word2idx (): dictionary of word to ids
        max_length (): the maximum length of the input sequences

    Returns: zero-padded list of ids

    """
    words = numpy.zeros(max_length).astype(int)

    # trim tokens after max length
    text = text[:max_length]

    for i, token in enumerate(text):
        if token in word2idx:
            words[i] = word2idx[token]
        else:
            words[i] = word2idx["<unk>"]

    return words
