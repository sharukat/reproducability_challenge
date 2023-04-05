import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve, precision_recall_curve
import torchvision
from pathlib import Path
import os

from lib.m_vgg16 import VGG16
import lib.global_settings as settings


class MetricsComputation():
    def __init__(self, results: list):
        self.preds = results[2]
        self.targets = results[3]
        self.conf_vals = np.array(results[5])
        self.bn_labels = np.array(results[4])
        

    def compute_optimal_risk(self):
        """Compute the optimal risk using predicted and target values.

        Args:
            preds (torch.Tensor): Predicted values after applying argmax on the output of softmax.
            targets (torch.Tensor): True labels/target values.

        Returns:
            float: The calculated optimal risk.
        """
        incorr_count = (self.preds != self.targets).sum().item()
        incorr_frac = incorr_count / len(self.preds) # fraction of incorrect predictions
        optimal_risk = incorr_frac + (1 - incorr_frac) * np.log(1 - incorr_frac)
        return optimal_risk


    def compute_aurc_eaurc(self):
        # Sort predictions and targets based on maximum softmax probabilities in descending order
        sorted_preds, indices = torch.sort(self.preds, descending=True)
        sorted_labels = self.targets[indices]

        # Compute risk and coverage values
        risk_vals = []
        coverage_vals = []
        num_incorrect = 0
        for i in range(len(sorted_labels)):
            if sorted_labels[i] != 1:
                num_incorrect += 1
            risk_vals.append(num_incorrect / (i+1))
            coverage_vals.append((i+1) / len(sorted_labels))

        # Compute optimal risk value
        optimal_risk = self.compute_optimal_risk()

        # Compute area under the risk-coverage curve and excessive area under the risk-coverage curve
        aurc = np.trapz(risk_vals, coverage_vals)
        eaurc = aurc - optimal_risk

        return aurc, eaurc

    def calc_aurc_eaurc(softmax, correct):
        softmax = np.array(softmax)
        correctness = np.array(correct)
        softmax_max = np.max(softmax, 1)
        sort_indices = np.argsort(softmax_max)[::-1]
        sorted_softmax_max = softmax_max[sort_indices]
        sorted_correctness = correctness[sort_indices]
        num_incorrect = 0
        risk_list = []
        coverage_list = []
        for i in range(len(sorted_correctness)):
            if sorted_correctness[i] == 0:
                num_incorrect += 1
            risk = num_incorrect / (i+1)
            coverage = (i+1) / len(sorted_correctness)
            risk_list.append(risk)
            coverage_list.append(coverage)
        optimal_risk = (num_incorrect / len(sorted_correctness)) + \
                      (1 - (num_incorrect / len(sorted_correctness))) * np.log(1 - (num_incorrect / len(sorted_correctness)))
        aurc = np.trapz(risk_list, coverage_list)
        eaurc = aurc - optimal_risk
        print("AURC {0:.2f}".format(aurc*1000))
        print("EAURC {0:.2f}".format(eaurc*1000))
        return aurc, eaurc

        
    
    def compute_pr_auc(self):
        precision, recall, thresholds = precision_recall_curve(
          self.bn_labels, np.max(self.conf_vals, 1))

        pr_auc = auc(recall, precision)
        return pr_auc

    def compute_fpr_tpr(self):
        fpr, tpr, thresholds = roc_curve(self.bn_labels, np.max(self.conf_vals, 1))
        return fpr, tpr
    

    def compute_metrics(self):
        fpr, tpr = self.compute_fpr_tpr()
        fpr_in_tpr_95 = fpr[np.argmin(np.abs(tpr-0.95))]
        aurc, eaurc = self.compute_aurc_eaurc()
        pr_auc = self.compute_pr_auc()

        return aurc, eaurc, fpr, fpr_in_tpr_95, pr_auc





# Calculate accuracy (a classification metric)
def accuracy_func(y_pred, y_true):
    # y_pred = torch.tensor(y_pred)
    # y_true = torch.tensor(y_true)
    correct = torch.eq(y_pred, y_true).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_true)) * 100 
    return acc



def select_loss_func(name: str, crl_loss_fn = None):
    """
    ========================================================
    Returns a PyTorch loss function based on the given name.
    ========================================================

    Args:
        name (str): The name of the loss function to use.['baseline', 'crl']
        crl_loss_fn (callable, optional): Correctness ranking loss function

    Returns:
        A PyTorch loss function object.
    """

    if name == 'baseline':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif name == 'crl':
        loss_fn = torch.nn.CrossEntropyLoss + crl_loss_fn
    return loss_fn


def load_model_opt_sch(model_name: str, num_classes: int):
    if model_name == 'vgg16':
      model = VGG16(num_classes)
    else:
      pass
    
    optimizer = optim.SGD(
      model.parameters(), 
      lr=settings.LR, 
      momentum=settings.MOMENTUM, 
      weight_decay=0.0001, nesterov=False)

    scheduler = optim.lr_scheduler.MultiStepLR(
      optimizer=optimizer, 
      milestones=[10, 150, 250], gamma=0.1) # Decays learning rate  

    return model, optimizer, scheduler


def save_model(
    checkpoint_dir:str, 
    dataset:str, 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    epoch: int, is_CRL:bool = False):
    
    MODEL_PATH = '/content/drive/MyDrive/NN_Course_Project/project/models'
  

    if not os.path.exists(os.path.join(MODEL_PATH, checkpoint_dir)):
        os.makedirs(os.path.join(MODEL_PATH, checkpoint_dir))



    to_save = {
      'model': model.state_dict(), 
      'optimizer':optimizer.state_dict(), 
      'scheduler':scheduler.state_dict()}
    folder_name = 'crl_models'
    model_name = f"{checkpoint_dir}_{dataset}_{epoch+1}.pt"
    if is_CRL:
      if not os.path.exists(os.path.join(MODEL_PATH, checkpoint_dir, folder_name)):
        os.makedirs(os.path.join(MODEL_PATH, checkpoint_dir, folder_name))
      MODEL_SAVE_PATH = os.path.join(MODEL_PATH, checkpoint_dir, folder_name, model_name)
    else:
      MODEL_SAVE_PATH = os.path.join(MODEL_PATH, checkpoint_dir, model_name)

    # Save the model state dict
    print(f"\n===== Checkpoint saved at epoch {epoch+1} =====")
    torch.save(obj=to_save, f=MODEL_SAVE_PATH)

        

def print_decos(mode):
    if mode == 'train':
        print("========================================")
        print("TRAINING INITIATED")
        print("========================================\n")

    elif mode == 'eval':
        print("\n========================================")
        print("EVALUATION RESULTS")
        print("========================================\n")




