import torch
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve, precision_recall_curve
import torchvision
from pathlib import Path
import os
from ignite.engine import Engine, Events, create_supervised_trainer
from ignite.handlers import Checkpoint, DiskSaver, ModelCheckpoint


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
        # Convert predictions and targets to numpy arrays
        # preds = preds.cpu().detach().numpy()
        # targets = targets.cpu().detach().numpy()

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



def select_model(name: str = None) -> torch.nn.Module:
    if name == 'resnet':
        model = ''
    elif name == 'densenet':
        model = ''
    else:
        weights = torchvision.models.VGG16_Weights.DEFAULT
        model = torchvision.models.vgg16(weights=weights)

    return model



def save_model(
    checkpoint_dir:str, 
    dataset:str, 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    epoch: int):
    
    MODEL_PATH = '/content/drive/MyDrive/NN_Course_Project/project/models'
  

    if not os.path.exists(os.path.join(MODEL_PATH, checkpoint_dir)):
        os.makedirs(os.path.join(MODEL_PATH, checkpoint_dir))


    to_save = {
      'model': model.state_dict(), 
      'optimizer':optimizer.state_dict(), 
      'scheduler':scheduler.state_dict()}

    # trainer.add_event_handler(Events.EPOCH_COMPLETED, training_saver, to_save)
    MODEL_SAVE_PATH = os.path.join(MODEL_PATH, checkpoint_dir, f"{checkpoint_dir}_{dataset}_{epoch+1}.pt")

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




