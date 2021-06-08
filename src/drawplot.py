import torch

from constants import GENERATOR_CHECKPOINT, DISCRIMINATOR_CHECKPOINT, CURRENT_RESULT_PATH
import matplotlib.pyplot as plt

if __name__ == "__main__":
    generator_path = GENERATOR_CHECKPOINT
    discriminator_path = DISCRIMINATOR_CHECKPOINT
    checkpoint = torch.load(generator_path)
    print(generator_path)
    e = checkpoint['epoch']
    generator_total_loss = checkpoint['total_loss']
    checkpoint = torch.load(discriminator_path)
    print(discriminator_path)
    discriminator_total_loss = checkpoint['total_loss']
    print('Loaded Checkpoint')
    print("epoch", e)
    plt.plot(generator_total_loss, label='Generator Loss')
    plt.plot(discriminator_total_loss, label='Discriminator Loss')
    plt.legend()
    plt.savefig(CURRENT_RESULT_PATH + '/loss.jpg')
    plt.clf()
