import numpy as np

def eval_q2m(scores, q2m_gts):
  '''
  Image -> Text / Text -> Image
  Args:
    scores: (n_query, n_memory) matrix of similarity scores
    q2m_gts: list, each item is the positive memory ids of the query id
  Returns:
    scores: (recall@1, 5, 10, median rank, mean rank)
    gt_ranks: the best ranking of ground-truth memories
  '''
  n_q, n_m = scores.shape
  gt_ranks = np.zeros((n_q, ), np.int32)

  for i in range(n_q):
    s = scores[i]
    sorted_idxs = np.argsort(-s)

    rank = n_m
    for k in q2m_gts[i]:
      tmp = np.where(sorted_idxs == k)[0][0]
      if tmp < rank:
        rank = tmp
    gt_ranks[i] = rank

  # compute metrics
  r1 = 100 * len(np.where(gt_ranks < 1)[0]) / n_q
  r5 = 100 * len(np.where(gt_ranks < 5)[0]) / n_q
  r10 = 100 * len(np.where(gt_ranks < 10)[0]) / n_q
  medr = np.median(gt_ranks) + 1
  meanr = gt_ranks.mean() + 1
  
  return (r1, r5, r10, medr, meanr)

