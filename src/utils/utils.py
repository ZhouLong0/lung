import torch
from tqdm import tqdm 

seed=2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def training(model, train_loader, criterion, optimizer, device):
    '''
    Train the model for one epoch 

    Args:
        model: model to be trained
        train_loader: DataLoader with the training data
        criterion: loss function
        optimizer: optimizer
        device: device where the model is allocated

    Returns:
        (float) epoch_train_loss: average loss of the epoch
        (float) epoch_train_accuracy: accuracy of the epoch        
    '''
    current_loss = 0.0
    current_corrects = 0
    train_total = 0
    model.train()
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):  
        optimizer.zero_grad() 

        data = data.to(device)
        targets = targets.to(device)

        outputs = model(data)
        loss = criterion(outputs, targets)
        preds = torch.softmax(outputs, dim=1).argmax(dim=1)
        current_loss += loss.item() 
        current_corrects += (preds == targets).sum() 
        train_total += len(outputs)
        loss.backward() 
        optimizer.step() 
#         if (batch_idx + 1 )%(len(train_loader)//10) == 0 :
#             print('[{:.0f}%]  Loss={:0.4f}'.format(100.*(batch_idx+1)/len(train_loader),loss.item()))                                                                                                                         
    epoch_train_loss, epoch_train_accuracy  = current_loss/len(train_loader), (100*current_corrects/train_total).item()
    return epoch_train_loss, epoch_train_accuracy
    
def testing(model, test_loader, criterion, device):
    '''
    Test the model on the test data

    Args:
        model: model to be tested
        test_loader: DataLoader with the test data
        criterion: loss function
        device: device where the model is allocated

    Returns:
        (float) epoch_test_loss: average loss on the training data
        (float) epoch_test_accuracy: accuracy on the training data
    '''
    model.eval()
    current_losses = 0
    correct = 0
    test_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            
            outputs = model(data)
            losses = criterion(outputs, target)   
            predictions = torch.softmax(outputs, dim=1).argmax(dim=1)
            correct += (predictions == target).sum() 
            current_losses += losses.item() 
            test_total += len(outputs)
        epoch_test_loss, epoch_test_accuracy = current_losses/len(test_loader), (100*correct/test_total).item()
    return epoch_test_loss, epoch_test_accuracy

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, current_loss):
        if self.best_loss is None or current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
def get_y_true_preds(train_loader, test_loader, model):
    y_true_train = torch.zeros(0, dtype=torch.long, device='cpu')
    y_preds_train = torch.zeros(0, dtype=torch.long, device='cpu')
    y_true_test = torch.zeros(0, dtype=torch.long, device='cpu')
    y_preds_test = torch.zeros(0, dtype=torch.long, device='cpu')
    
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):   
        data = data.cuda()
        targets = targets.cuda()
        scores = model(data)
        preds = torch.softmax(scores, dim=1).argmax(dim=1) 
        y_true_train = torch.cat([y_true_train, targets.view(-1).cpu()])
        y_preds_train = torch.cat([y_preds_train, preds.view(-1).cpu()])
        del data, targets, scores, preds
        
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            predictions = torch.softmax(output, dim=1).argmax(dim=1)
            y_true_test = torch.cat([y_true_test, target.view(-1).cpu()])
            y_preds_test= torch.cat([y_preds_test, predictions.view(-1).cpu()])
            del data, target, output, predictions
    return y_true_train, y_preds_train, y_true_test , y_preds_test