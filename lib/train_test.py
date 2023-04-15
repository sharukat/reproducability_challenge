import torch
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import torch.nn.functional as F
from termcolor import colored

from lib.helper_funcs import accuracy_func, save_model
from lib.crl_loss import CRL
import lib.global_settings as settings

class TrainTestModel():
    def __init__(self):
        pass

    def train_step(self, 
                   model: torch.nn.Module, 
                   dataloader: torch.utils.data.DataLoader, 
                   optimizer: torch.optim.Optimizer, 
                   scheduler: torch.optim.lr_scheduler, 
                   device: torch.device,
                   epoch: int,
                   is_CRL: bool) -> Tuple[float, float]:
        
        model.train()
        train_loss, train_acc = 0, 0

        if is_CRL:
          CRL_LOSS = CRL(ranking_criterion = settings.MRL, tr_datapoints = len(dataloader.dataset))

        for im, label, idx in dataloader:
            im, label = im.to(device), label.to(device)
            logits = model(im)

            if is_CRL:
              crl = CRL_LOSS.correctness_ranking_loss(logits, idx)
              loss = settings.CEL(logits, label) + crl
            else:
              loss = settings.CEL(logits, label)

            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Calculate and accumulate accuracy metric across all batches
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            train_acc += accuracy_func(preds, label)

            if is_CRL:
              correctness = torch.eq(preds, label).int()
              CRL_LOSS.update_correctness(idx, correctness, logits)

        if is_CRL:
          CRL_LOSS.increment_max_correctness(epoch)

        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)
        return train_loss, train_acc
    

    def test(self,
                  model: torch.nn.Module, 
                  dataloader: torch.utils.data.DataLoader, 
                  loss_fn: torch.nn.Module,
                  device: torch.device, 
                  is_CRL: bool = False) -> Tuple[float, float]:
        """
        ===================================================================
        Tests the model for a single epoch.
        ===================================================================

        Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        
        Returns:
        A tuple of testing loss and testing accuracy metrics.
        """
        model.to(device)
        model.eval() 
        test_loss, test_acc = 0, 0
        binary_labels = []
        confidence_scores = []
        targets = []
        xx=[]

        # Defining losses
        CEL = torch.nn.CrossEntropyLoss()
        if is_CRL:
          CRL_LOSS = CRL(ranking_criterion = settings.MRL, tr_datapoints = len(dataloader.dataset))
        
        with torch.no_grad():
            for X, labels, idx in dataloader:
                X, labels = X.to(device), labels.to(device)
                test_logits = model(X)

                if is_CRL:
                  crl = CRL_LOSS.correctness_ranking_loss(test_logits, idx)
                  loss = settings.CEL(test_logits, labels) + crl
                else:
                  loss = settings.CEL(test_logits, labels)

                test_loss += loss.item()

                # Calculate and accumulate accuracy
                softmax = torch.softmax(test_logits, dim=1)
                test_preds = torch.argmax(softmax, dim=1)
                test_acc += accuracy_func(test_preds, labels)

                for p, t in zip(test_preds, labels):
                    binary_labels.append(int(p == t))

                softmax_np = softmax.cpu().detach().numpy()
                confidence_scores.extend(np.max(softmax_np, 1))
                targets.extend(labels.cpu().detach().numpy())

        # Adjust metrics to get average loss and accuracy per batch 
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        result = [test_loss, test_acc, test_preds, targets, binary_labels, confidence_scores]
        return result
    
    


    def train(self,
            model: torch.nn.Module,
            model_name: str,
            dataset: str,
            train_dataloader: torch.utils.data.DataLoader, 
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            epochs: int,
            device: torch.device, 
            resume_epoch: int = None, 
            checkpoint = None, 
            is_CRL: bool = False) -> Dict[str, List]:
        
        """
        ===================================================================
        Trains the model.
        ===================================================================

        Passes the target models through train_step() and test_step()
        functions for a number of epochs, training and testing the model
        in the same epoch loop.
        
        Args:
        model           : A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader : A DataLoader instance for the model to be tested on.
        optimizer       : A PyTorch optimizer to help minimize the loss function.
        epochs          : An integer indicating how many epochs to train for.
        device          : A target device to compute on (e.g. "cuda" or "cpu").

        Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        """

        model.to(device)
        initial_epoch = 0

        if resume_epoch is not None:
          initial_epoch = resume_epoch

        for epoch in tqdm(range(initial_epoch, epochs)):
            train_loss, train_acc = self.train_step(model=model,
                                                    dataloader=train_dataloader,
                                                    optimizer=optimizer,
                                                    scheduler=scheduler,
                                                    device=device,
                                                    epoch=epoch,  
                                                    is_CRL=is_CRL)
        
            print(
            f"{colored('Epoch', 'blue')}: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f}% | "
            )


            # Save model
            if epoch+1 in settings.CHECKPOINT_EPOCHS:
              if is_CRL:
                save_model(model_name, dataset, model, optimizer, scheduler, epoch, is_CRL=True)
              else:
                save_model(model_name, dataset, model, optimizer, scheduler, epoch)
