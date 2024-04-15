import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_coarse(all_scores, all_matchability = None, mode = "avg"):
    if mode == "avg":
        avg_score = sum(all_scores) / len(all_scores)
        scores = avg_score
    elif mode == "final":
        scores = all_scores[-1]
    if all_matchability is not None:
        matchability_A = all_matchability[-1][0]
        matchability_B = all_matchability[-1][1]
        scores = scores + matchability_A[...,None] + matchability_B[...,None,:]

    score_threshold = torch.topk(scores.flatten(), k = 256).values[-1]
    matches = torch.nonzero(scores >= score_threshold, as_tuple = True)
    confidence = F.sigmoid(scores[matches[0], matches[1]])
    return matches[1], matches[0], confidence