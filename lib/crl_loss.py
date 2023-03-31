import torch
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

    def __init__(self, preds, tr_datapoints):
        self.preds = preds
        self.correctness = torch.zeros((tr_datapoints), dtype=torch.float32)
        self.confidences = torch.zeros((tr_datapoints), dtype=torch.float32)
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def negative_entropy(self):
        softmax = torch.softmax(self.preds, dim=1)
        logrithmic_softmax = torch.log_softmax(self.preds, dim=1)
        neg_entropy = -torch.sum(softmax * logrithmic_softmax, dim=1)
        normalized_confidence = MinMaxScaler().fit_transform(neg_entropy)
        return normalized_confidence

    def normalization(self):
        min_val = self.correctness.min()
        max_val = float(self.max_correctness_val)
        result = (self.)

    def compute_margin(self, index_1, index_2):
        idx1_cumulative_corr = self.correctness[index_1].cpu()
        idx2_cumulative_corr = self.correctness[index_2].cpu()
        
        # Normalizing
        idx1_cumulative_corr = self.scaler.fit_transform(idx1_cumulative_corr)
        idx2_cumulative_corr = self.scaler.fit_transform(idx2_cumulative_corr)


def max_class_prob(data):
    output = torch.nn.Softmax(dim=1)(data)
    max_prob, max_class = torch.max(output, dim=1)
    return max_prob