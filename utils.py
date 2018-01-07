import numpy as np


iteration_str = "\nEnd iter {} - k/lr: {}/{} momentum: {} - MAE/RMSE: {}/{}"


def chunker(seq, size):
    '''
    function: return a generator of sequences in size of 'size'
    parameter:
    seq: sequence
    size: int
    '''
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def avg(x):
    return sum(x)/len(x)


def _expand_line(line, k=5):
    expanded = [0] * (len(line) * k)
    for i, el in enumerate(line):
        if float(el) != 0.:
            el = float(el)
            expanded[(i*k) + int(round(el)) - 1] = 1
    return expanded


def expand(data, k=5):
    '''
    function: return a rating matrix, with every movie_id's kth w.r.t rating(rounded) being 1, and 0 elsewhere
    '''
    new = []
    for m in data:
        new.extend(_expand_line(m.tolist()))

    return np.array(new).reshape(data.shape[0], data.shape[1] * k)


def revert_expected_value(m, k=5, do_round=True):
    '''
    usage: calculate the expected ratings of movies (with rounding the ratings or not)
    '''
    mask = list(range(1, k+1))
    vround = np.vectorize(round)

    if do_round:
        users = vround((m.reshape(-1, k) * mask).sum(axis=1))
    else:
        users = (m.reshape(-1, k) * mask).sum(axis=1)

    return np.array(users).reshape(m.shape[0], m.shape[1] // k)
