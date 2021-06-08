import torch


def initialize_checkpoint(model, optimizer, path):
    save_checkpoint(0, model, optimizer, 0.0, [], path)


def save_checkpoint(epoch, model, optimizer, running_loss, total_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss,
        'total_loss': total_loss
    }, path)
