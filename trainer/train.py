import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

class Trainer:

    def __init__(self, train_loader, val_loader, model, criterion, optimizer, device):

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion


    def train(self):

        self.model.train()

        # Load in the data in batches using the train_loader object
        losses = 0
        acc = 0

        batch = 0
        for i, (images, labels) in enumerate(self.train_loader):
            # Move tensors to the configured device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            y = labels
            y_true = y.detach().cpu().numpy()
            y_pred = outputs.data.detach().cpu().numpy().argmax(axis=1)
            acc += accuracy_score(y_true, y_pred) * 100
            losses += loss.data.detach().cpu().numpy()
            batch += 1
        # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

        return batch, acc, losses

    def val(self):

        self.model.eval()
        v_losses = 0
        v_acc = 0
        v_recall = 0
        v_f1 = 0
        v_precision = 0
        y_preds = []
        y_trues = []

        # Load in the data in batches using the train_loader object
        batch = 0
        for i, (images, labels) in enumerate(self.val_loader):
            # Move tensors to the configured device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            y = labels
            y_true = y.detach().cpu().numpy()
            y_pred = outputs.data.detach().cpu().numpy().argmax(axis=1)
            v_acc += accuracy_score(y_true, y_pred) * 100
            v_recall += recall_score(y_true, y_pred,average='weighted',zero_division=1.0) * 100
            v_f1 += f1_score(y_true, y_pred,average='weighted',zero_division=1.0) * 100
            v_precision += precision_score(y_true, y_pred,average='weighted',zero_division=1.0) *100
            v_losses += loss.data.detach().cpu().numpy()
            y_trues += y_true.tolist()
            y_preds += y_pred.tolist()
            batch += 1
        # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

        return batch, v_acc, v_recall,v_f1,v_precision,v_losses, y_trues, y_preds
