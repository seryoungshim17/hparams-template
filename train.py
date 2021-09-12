import torch
from train import calculateAcc
class Train():
    def __init__(self, model, num_classes, trainloader, valloader):
        self.model = model
        self.num_classes = num_classes
        self.trainloader = trainloader
        self.valloader = valloader
        
    def train(self, optim, learning_rate=1e-5, epochs=20, writer=None):
        optims = {
            'Adam': torch.optim.Adam,
            'SGD': torch.optim.SGD
        }
        optimizer = optims[optim](self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        best_val_loss = 1e9
        best_val_f1 = 0

        for e in range(1, epochs+1):
            self.model.train()
            for (X_batch, y_batch) in tqdm(self.trainloader, desc=f"Epoch {e}"):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()        
                y_pred = self.model(X_batch)

                loss = criterion(y_pred, y_batch.squeeze())
                acc = calculateAcc(y_pred, y_batch.squeeze(), num_classes=num_classes)

                loss.backward()
                optimizer.step()
            
            # validation set eval
            with torch.no_grad(): 
                self.model.eval() 
                val_loss = 0
                val_acc = 0
                val_f1_score = 0
                for x_val, y_val in self.valloader:  
                    x_val = x_val.to(device)  
                    y_val = y_val.to(device)   

                    yhat = self.model(x_val)  
                    val_loss += criterion(yhat, y_val.squeeze()).item()
                    acc = calculateAcc(yhat, y_val.squeeze(), num_classes=num_classes)
                    val_acc += acc[0]
                    val_f1_score += acc[1]

            if best_val_f1 <= val_f1_score and best_val_loss >= val_loss:
                best_val_loss = val_loss
                best_val_f1 = val_f1_score
                self.best_weight = {
                    'epoch': e,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }
                print("best weight updated")
        
        return self.best_weight