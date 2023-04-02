import torch
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler


class CRL:
    """
    ===================================================================
    Correctness Ranking Loss (CRL)
    ===================================================================

    CRL computes the ordinal ranking of confidence estimates/
    
    Args:
    preds           : Prediction output of the model before softmax

    Returns:
    xxx
    """

    def __init__(self, ranking_criterion, tr_datapoints):
        self.ranking_criterion = ranking_criterion
        self.max_correctness = 0.0
        self.correctness = np.zeros((tr_datapoints))
        self.confidences = np.zeros((tr_datapoints))

        
    # ======================= Compute Target Margin =========================
    def compute_margin(self, index_1, index_2):
        index_1 = np.array(index_1)
        index_2 = index_2.cpu().detach().numpy()

        idx1_cum_corr = self.correctness[index_1].cpu()
        idx2_cum_corr = self.correctness[index_2].cpu()
        
        # Normalizing
        scaler = MinMaxScaler((self.correctness.min(), self.max_correctness))
        idx1_cum_corr = scaler.fit_transform(idx1_cum_corr)
        idx2_cum_corr = scaler.fit_transform(idx2_cum_corr)


    # ============== Compute Confidence using Margin Estimator ===============
    def margin(self, logits):
        top_probs, _  = torch.topk(F.softmax(logits, dim=1), 2, dim=1)
        confidence    = top_probs[:, 0] - top_probs[:, 1]
        return confidence
        

    # ================== Compute Correctness Ranking Loss ====================
    def correctness_ranking_loss(self, logits, idx):
        confidence = self.margin(logits)

        rank_1  = confidence
        rank_2  = torch.roll(confidence, -1)

        index_1 = idx
        index_2 = torch.roll(index_1, -1)
        target, margin = self.compute_margin(index_1, index_2)
        rank_2  += margin / (target + 1e-10)

        crl = self.ranking_criterion(rank_1, rank_2, target)
        return crl


class Correctness(CRL):
    def increment_max_correctness(self, current_epoch):
        super().max_correctness += int(current_epoch > 1)

    def update(self, data_idx, correctness, logits):
        confidence = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        super().correctness[data_idx] += correctness
        super().confidence[data_idx] = confidence



