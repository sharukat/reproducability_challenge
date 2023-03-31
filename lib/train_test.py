import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import torch.nn.functional as F
import sys
sys.path.append('/content/drive/MyDrive/NN_Course_Project/project/lib')

from helper_funcs import accuracy_func, save_model


class TrainTestModel():
    def __init__(self):
        pass

    def train_step(self, 
                   model: torch.nn.Module, 
                   dataloader: torch.utils.data.DataLoader, 
                   loss_fn: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer, 
                   scheduler: torch.optim.lr_scheduler, 
                   device: torch.device) -> Tuple[float, float]:
        
        model.train()
        train_loss, train_acc = 0, 0

        for im, label in dataloader:
            im, label = im.to(device), label.to(device)
            logits = model(im)
            loss = loss_fn(logits, label)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Calculate and accumulate accuracy metric across all batches
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            # train_acc += torch.eq(label, pred_class).sum().item()/len(pred_class)
            train_acc += accuracy_func(preds, label)

        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)
        return train_loss, train_acc
    

    def test(self,
                  model: torch.nn.Module, 
                  dataloader: torch.utils.data.DataLoader, 
                  loss_fn: torch.nn.Module,
                  device: torch.device) -> Tuple[float, float]:
        """
        Tests the model for a single epoch.

        Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        Returns:
        A tuple of testing loss and testing accuracy metrics.
        """

        model.eval() 
        test_loss, test_acc = 0, 0
        binary_labels = []
        confidence_scores = []

        with torch.inference_mode():
            for batch, (X, labels) in enumerate(dataloader):
                X, labels = X.to(device), labels.to(device)
                test_logits = model(X)
                loss = loss_fn(test_logits, labels)
                test_loss += loss.item()

                # Calculate and accumulate accuracy
                softmax = torch.softmax(test_logits, dim=1)
                test_preds = torch.argmax(softmax, dim=1)
                test_acc += accuracy_func(test_preds, labels)

                for p, t in zip(test_preds, labels):
                    binary_labels.append(int(p == t))

                confidence_scores.extend(softmax.cpu().detach().numpy())

        # Adjust metrics to get average loss and accuracy per batch 
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        result = [test_loss, test_acc, test_preds, labels, binary_labels, confidence_scores]
        return result
    
    


    def train(self,
            model: torch.nn.Module,
            model_name: str,
            dataset: str,
            train_dataloader: torch.utils.data.DataLoader, 
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            loss_fn: torch.nn.Module,
            epochs: int,
            device: torch.device, 
            resume_epoch: int = None, 
            checkpoint = None) -> Dict[str, List]:
        
        """
        ===================================================================
        Trains and tests the model.
        ===================================================================

        Passes the target models through train_step() and test_step()
        functions for a number of epochs, training and testing the model
        in the same epoch loop.
        
        Args:
        model           : A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader : A DataLoader instance for the model to be tested on.
        optimizer       : A PyTorch optimizer to help minimize the loss function.
        loss_fn         : A PyTorch loss function to calculate loss on both datasets.
        epochs          : An integer indicating how many epochs to train for.
        device          : A target device to compute on (e.g. "cuda" or "cpu").

        Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        """
        results = {"train_loss": [],
                   "train_acc": [],
                   "test_loss": [],
                   "test_acc": []}

        torch.manual_seed(42)

        

        
        # Make sure model on target device
        model.to(device)

        if resume_epoch is not None:
          
            model.load_state_dict(checkpoint['model'])           # weights
            optimizer.load_state_dict(checkpoint['optimizer'])   # optimizer
            scheduler.load_state_dict(checkpoint['scheduler'])   # lr_scheduler
 
            for epoch in range(resume_epoch, epochs):
                train_loss, train_acc = self.train_step(model=model,
                                                        dataloader=train_dataloader,
                                                        loss_fn=loss_fn,
                                                        optimizer=optimizer,
                                                        scheduler=scheduler,
                                                        device=device)

                print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f}% | "
                )

                results["train_loss"].append(train_loss)
                results["train_acc"].append(train_acc)

                # Save model
                if epoch+1 in [50,100,150,200,250,300]:
                  save_model(model_name, dataset, model, optimizer, scheduler, epoch)

            return results

        else:
            for epoch in tqdm(range(epochs)):
                train_loss, train_acc = self.train_step(model=model,
                                                        dataloader=train_dataloader,
                                                        loss_fn=loss_fn,
                                                        optimizer=optimizer,
                                                        scheduler=scheduler,
                                                        device=device)
            
                print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f}% | "
                )

                results["train_loss"].append(train_loss)
                results["train_acc"].append(train_acc)

                # Save model
                if epoch+1 in [50,100,150,200,250,300]:
                  save_model(model_name, dataset, model, optimizer, scheduler, epoch)

            return results