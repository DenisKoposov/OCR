import torch
import numpy as np
from tqdm import tqdm
#from tensorboardX import SummaryWriter

def load_model(model_class, path, device):
    model = model_class()
    model.load_state_dict(torch.load(path)["state_dict"])
    model = model.to(device)
    return model


def validate(model, val_loader, criterion, device):
    
    val_loss = 0.
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():

        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            target = labels.detach().cpu().numpy()
            labels = labels.view(-1)

            # Forward pass
            outputs = model(images)
            predicted = torch.argmax(outputs.detach().cpu(), dim=2).numpy()
            outputs = outputs.view(-1, outputs.size(-1))
            #print(outputs.size(), labels.size())
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            total += labels.size(0)

            # Compare predicted and true words
            for i in labels.size(0):
                l = np.where(target[i, :] == -100)[0]

                if np.all(predicted[i, :l] == target[i, :l]):
                    correct += 1

    val_loss /= len(val_loader)
    val_acc = correct / total

    return val_loss, val_acc

def train(model, train_loader, val_loader, criterion, optimizer,
          num_epochs, save_best, save_ckpt, device,
          val_each=100, print_each=10, start_from_ckpt=None):
    
    train_curve = []
    valid_curve = []
    epochs = []
    
    #writer = SummaryWriter()
    best_val_loss = np.inf
    last_epoch = 0
    
    if start_from_ckpt:
        model_state = torch.load(start_from_ckpt)
        last_epoch = model_state['epoch']
        model.load_state_dict(model_state['state_dict'])

    for epoch in range(last_epoch+1, last_epoch+num_epochs+1):
            
        train_loss = 0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.
        iter = 0
        # Train model
        model.train()

        for images, labels in tqdm(train_loader, total=len(train_loader)):
            iter += 1
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            if iter % 100 == 0:
                predicted = torch.argmax(outputs.detach().cpu(), dim=2).numpy()[0, :]
                true = labels.detach().cpu().numpy()[0, :]
                word = "".join([model.vocab[code] for code in predicted if code != -100])
                true = "".join([model.vocab[code] for code in true if code != -100])
                print(word)
                print(true)

            labels = labels.view(-1)
            outputs = outputs.view(-1, outputs.size(-1))
            #print(outputs.size(), labels.size())
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            if iter % 100 == 0:
                print(loss.item())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            # Optimize
            optimizer.step()

        train_loss /= len(train_loader)

        if epoch % print_each == 0 or epoch == 1:

            if epoch % val_each == 0 or epoch == 1:

                val_loss, val_acc = validate(model, val_loader, criterion, device)
                
                if val_loss < best_val_loss:
                    model_state = {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }
                    best_val_loss = val_loss
                    torch.save(model_state, save_best)
                
                train_curve.append(train_loss)
                valid_curve.append(val_loss)
                epochs.append(epoch)
                
                #writer.add_scalars('logs/loss', {
                #    'training': train_loss,
                #    'validation': val_loss
                #}, epoch)
                #writer.add_scalar('logs/val_accuracy', val_acc, epoch)

                print('Epoch [{}/{}], Loss: {:.4f}, Val acc: {:.2f}' 
                   .format(epoch, last_epoch+num_epochs, train_loss, val_acc))
            else:
                print('Epoch [{}/{}], Loss: {:.4f}' 
                   .format(epoch, last_epoch+num_epochs, train_loss))

        # Save the model checkpoint
        model_state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(model_state, save_ckpt)

    #writer.close()
    return train_curve, valid_curve, epochs