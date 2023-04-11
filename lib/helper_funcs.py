import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
import torchvision
from pathlib import Path
import os

from lib.m_vgg16 import VGG16
from lib.m_resnet import ResNet110
import lib.global_settings as settings


class MetricsComputation():
    def __init__(self, results: list):
        self.preds = np.array(results[-1])
        self.targets = np.array(results[3])
        self.conf_vals = np.array(results[5])
        self.bn_labels = np.array(results[4])


    def compute_aurc_eaurc(self):
        sorted_preds = np.sort(self.preds)[::-1]
        bn_labels = self.bn_labels[np.argsort(self.preds)[::-1]]
        
        # Compute risk and coverage values
        risk_vals = []
        coverage_vals = []
        num_incorrect = 0
        for i in range(len(sorted_preds)):
            if bn_labels[i] == 0:
              num_incorrect += 1
            risk_vals.append(num_incorrect / (i+1))
            coverage_vals.append((i+1) / len(sorted_preds))

        # Compute optimal risk value
        optimal_risk = risk_vals[-1] + (1 - risk_vals[-1]) * np.log(1 - risk_vals[-1])
        # Compute area under the risk-coverage curve and excessive area under the risk-coverage curve
        aurc = np.trapz(risk_vals, coverage_vals)
        eaurc = aurc - optimal_risk

        return aurc, eaurc

    def compute_metrics(self):
        fpr, tpr, thresholds = roc_curve(self.bn_labels, self.preds)
        fpr_in_tpr_95 = fpr[np.argmin(np.abs(tpr-0.95))]
        aurc, eaurc = self.compute_aurc_eaurc()
        pr_auc = average_precision_score(-1 * self.bn_labels + 1, -1 * self.preds)


        return aurc, eaurc, fpr_in_tpr_95, pr_auc



# Calculate accuracy (a classification metric)
def accuracy_func(y_pred, y_true):
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
      model = ResNet110(num_classes)
    
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

    if not os.path.exists(os.path.join(settings.MODEL_PATH, checkpoint_dir)):
        os.makedirs(os.path.join(settings.MODEL_PATH, checkpoint_dir))



    to_save = {
      'model': model.state_dict(), 
      'optimizer':optimizer.state_dict(), 
      'scheduler':scheduler.state_dict()}
    folder_name = 'crl_models'
    model_name = f"{checkpoint_dir}_{dataset}.pt"
    if is_CRL:
      if not os.path.exists(os.path.join(settings.MODEL_PATH, checkpoint_dir, folder_name)):
        os.makedirs(os.path.join(settings.MODEL_PATH, checkpoint_dir, folder_name))
      MODEL_SAVE_PATH = os.path.join(settings.MODEL_PATH, checkpoint_dir, folder_name, model_name)
    else:
      MODEL_SAVE_PATH = os.path.join(settings.MODEL_PATH, checkpoint_dir, model_name)

    # Save the model state dict
    print(f"\n===== Checkpoint saved at epoch {epoch+1} =====")
    torch.save(obj=to_save, f=MODEL_SAVE_PATH)


def load_trained_model(model_name:str, dataset:str, num_classes: int, is_CRL: bool = False):
    if is_CRL:
      path = os.path.join(settings.MODEL_PATH, f"{model_name}/crl_models", f'{model_name}_{dataset}_300.pt')
    else:
      path = os.path.join(settings.MODEL_PATH, model_name, f'{model_name}_{dataset}_300.pt')
      
    checkpoint = torch.load(path)
    model = VGG16(num_classes) if model_name == 'vgg16' else ResNet110()
    model.load_state_dict(checkpoint['model'])
    print(f"========== Successfully loaded {model_name} model trained using {dataset} for testing ==========")
    return model


def print_decos(mode):
    if mode == 'train':
        print("========================================")
        print("TRAINING INITIATED")
        print("========================================\n")

    elif mode == 'eval':
        print("\n========================================")
        print("EVALUATION RESULTS")
        print("========================================\n")

def print_results(accuracy, aurc, eaurc, pr_auc, fpr_in_tpr_95):
    print(f"Test Accuracy: {accuracy:.2f}")
    print(f"Area Under Risk Curve (AURC): {aurc*1000:.2f}")
    print(f"Excessive-AURC (E-AURC): {eaurc*1000:.2f}")
    print(f"Area Under Precision-Recall Curve (AUPR): {pr_auc*100:.2f}")
    print(f"False Positive Rate (FPR) at 95% True Positive Rate (TPR): {fpr_in_tpr_95*100:.2f}\n")


