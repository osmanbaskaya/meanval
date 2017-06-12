from gensim.models import KeyedVectors
import numpy as np
import sys
from constant import PADDING_TOKEN_ID
from preprocess import Vocabulary


WORD2VEC_PATH = '../data/embeddings/GoogleNews-vectors-negative300.bin.gz'
WORD2VEC_EMB_DIMENSION = 300


def get_word2vec_model(path=WORD2VEC_PATH):
    return KeyedVectors.load_word2vec_format(path, binary=True)


def get_embedding_weights(vocabulary, path=WORD2VEC_PATH):
    model = get_word2vec_model(path)

    # existing_words = [word for word in vocabulary if word in model]

    # Refresh the vocabulary according to the new word list.
    # vocabulary = Vocabulary.build(existing_words, 1, vocabulary.num_of_already_allocated_tokens)

    embedding_weights = np.random.rand(vocabulary.size, WORD2VEC_EMB_DIMENSION)
    embedding_weights[PADDING_TOKEN_ID, :] = np.zeros(WORD2VEC_EMB_DIMENSION)  # zeros for padding token.

    missing = set()
    for word, idx in vocabulary:
        if word in model:
            embedding_weights[idx, :] = model[word]
        else:
            print(u"{} not found.".format(word), file=sys.stderr)
            missing.add(word)

    print(u"Total number of missing words: {}".format(len(missing)), file=sys.stderr)
    return embedding_weights

