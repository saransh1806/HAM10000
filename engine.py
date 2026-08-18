import torch
from torch import nn
#importing tqdm for progress bar
from tqdm.auto import tqdm
# creating function for training our model
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
#setting model in train mode
  model.train()

  train_loss, train_acc = 0, 0
# iterating over different batches
  for batch, (X, y) in enumerate(dataloader):
      X, y = X.to(device), y.to(device)
      #forward pass
      y_pred = model(X)
      # calculating loss
      loss = loss_fn(y_pred, y)
      #summing up loss from differnt batches
      train_loss += loss.item()
      #setting gradients to zero
      optimizer.zero_grad()
      #performing backpropagation
      loss.backward()
      # optimization step
      optimizer.step()
      # getting predicted class
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      #calculating train accuracy
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)
  #calculating average loss
  train_loss = train_loss / len(dataloader)
  #calculating average loss
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
  # setting model into evaluation mode
  model.eval()

  test_loss, test_acc = 0, 0
  # opening context manager
  with torch.inference_mode():
      # iterating over batches
      for batch, (X, y) in enumerate(dataloader):
          X, y = X.to(device), y.to(device)
          #forward pass
          test_pred_logits = model(X)
          #calcualting loss
          loss = loss_fn(test_pred_logits, y)
          #summing losses over batches
          test_loss += loss.item()
          # calcualting predicted labels
          test_pred_labels = torch.argmax(test_pred_logits,dim=1)
          #calculating accuraacy and summing over batches
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
  # calculating average loss
  test_loss = test_loss / len(dataloader)
  #calculating average accuracy
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device):
  # storing results into dictionary
  results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }

  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      # train step
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      # test step
      test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)
      # print current epoch result
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
      )
      # store current epoch result
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)
  # return result
  return results