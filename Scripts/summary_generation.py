

import numpy as np


def sentence_ranking(data):
    return np.sum(data, axis=1) # summing up all sentences