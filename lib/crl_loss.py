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
        self.max_correctness = 1
        self.correctness = np.zeros((tr_datapoints))
        self.confidences = np.zeros((tr_datapoints))

        
    # ======================= Compute Target Margin =========================
    def compute_margin(self, index_1, index_2):
        index_1 = index_1.cpu().detach().numpy()
        index_2 = index_2.cpu().detach().numpy()
        idx1_cum_corr = self.correctness[index_1].reshape(-1,1)
        idx2_cum_corr = self.correctness[index_2].reshape(-1,1)

        # Normalizing
        scaler = MinMaxScaler((self.correctness.min(), self.max_correctness))
        idx1_cum_corr = scaler.fit_transform(idx1_cum_corr)
        idx2_cum_corr = scaler.fit_transform(idx2_cum_corr)

        target_1 = idx1_cum_corr[: len(index_1)]
        target_2 = idx2_cum_corr[: len(index_2)]

        greater = np.zeros_like(target_1, dtype='float')
        lesser = np.zeros_like(target_1, dtype='float')

        np.greater(target_1, target_2, out=greater)
        np.less(target_1, target_2, out=lesser)
        lesser *= -1

        # greater_vals = np.array(target_1 > target_2, dtype='float')
        # lesser_vals = np.array(target_1 < target_2, dtype='float') * -1

        target = greater + lesser
        target = target.ravel()
        target = torch.from_numpy(target).float().cuda()

        margin = abs(target_1 - target_2)
        margin = margin.ravel()
        margin = torch.from_numpy(margin).float().cuda()

        return target, margin

    # ================= Increment Correctness at each Epoch ==================
    def increment_max_correctness(self, current_epoch):
        self.max_correctness += int(current_epoch+1 > 1)


    # ========== Update Correctness Value at each Dataloader Iterate ==========
    def update_correctness(self, data_idx, correctness, logits):
        data_idx = np.array(data_idx)
        confidence = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        self.correctness[data_idx] += correctness.cpu().detach().numpy()
        self.confidences[data_idx] = confidence.cpu().detach().numpy()


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
        index_1 = torch.tensor(idx)
        index_2 = torch.roll(index_1, -1)
        target, margin = self.compute_margin(index_1, index_2)
        rank_2  += margin / (target + 1e-10)
        crl = self.ranking_criterion(rank_1, rank_2, target)
        return crl