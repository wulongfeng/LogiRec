import numpy as np

def hit_at_k(ans, predicted, k):
    assert k >= 1
    for answer in ans:
        if answer in predicted[:k]:
            return 1
    return 0

def dcg_at_k(ans, method=1):
    if len(ans) == 0:
        return 0
    r = np.array(ans)
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(ans, predicted, k, method=1):
    rel = [int(pred in ans) for pred in predicted[:k]]
    dcg = dcg_at_k(rel, method)

    rel.sort(reverse=True)
    dcg_max = dcg_at_k(rel, method)
    if not dcg_max:
        return 0.
    return dcg / dcg_max


def MRR(ans, predicted):
    if len(ans) == 0 or len(predicted) == 0:
        return 0
    idxList = []
    for answer in ans:
        firstIdx = predicted.index(answer)
        idxList.append(firstIdx)
    idxList.sort()
    if not idxList:
        return 0
    return 1/(idxList[0] + 1)


def dcg_at_label(r, k, method=1):
    r = r[:k]
    r = np.array(r)

    if len(r):
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_label(r, k, method=1):
    dcg_max = dcg_at_label(sorted(r[:k], reverse=True), k, method)
    print ("dcg max: {}".format(dcg_max))
    if not dcg_max:
        return 0.
    return dcg_at_label(r, k, method) / dcg_max

