import nltk


class nltk.collocations.BigramCollocationFinder(word_fd, bigram_fd, window_size=2):
    default_ws = 2


def __init__(self, word_fd, bigram_fd, window_size=2):
    AbstractCollocationFinder.__init__(self, word_fd, bigram_fd)
    self.window_size = window_size


@classmethod
def from_words(cls, words, window_size=2):
    wfd = FreqDist()
    bfd = FreqDist()

    if window_size < 2:
        raise ValueError("Specify window_size at least 2")

    for window in ngrams(words, window_size, pad_right=True):
        w1 = window[0]
        if w1 is None:
            continue
        wfd[w1] += 1
        for w2 in window[1:]:
            if w2 is not None:
                bfd[(w1, w2)] += 1
    return cls(wfd, bfd, window_size=window_size)


def score_ngram(self, score_fn, w1, w2):
    n_all = self.N
    n_ii = self.ngram_fd[(w1, w2)] / (self.window_size - 1.0)
    if not n_ii:
        return
    n_ix = self.word_fd[w1]
    n_xi = self.word_fd[w2]
    return score_fn(n_ii, (n_ix, n_xi), n_all)
