import time

import matplotlib.pyplot as plt
import torch
from torch import nn

from constants import GENERATOR_CHECKPOINT, DISCRIMINATOR_CHECKPOINT, GENERATOR_CHECKPOINT_PATH, \
    DISCRIMINATOR_CHECKPOINT_PATH, CURRENT_RESULT_PATH, GENERATOR_RESULT_PATH, CURRENT_CHECKPOINT_PATH
from utils.checkpoint import save_checkpoint
from utils.image import save_image


def train(epochs, generator_model, discriminator_model, generator_optimizer, discriminator_optimizer, data_loader,
          val_data_loader):
    generator_total_loss = []
    discriminator_total_loss = []
    generator_path = GENERATOR_CHECKPOINT
    discriminator_path = DISCRIMINATOR_CHECKPOINT
    bce_loss = nn.BCELoss().cuda()
    l1_loss = nn.L1Loss().cuda()
    e = 0
    if e == 0:
        checkpoint = torch.load(generator_path)
        print('INFO::Generator Path: '+generator_path)
        generator_model.load_state_dict(checkpoint['model_state_dict'])
        generator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        e = checkpoint['epoch']
        print('INFO::Generator Epochs: ' + str(e))
        generator_total_loss = checkpoint['total_loss']
        generator_model.train()
        checkpoint = torch.load(discriminator_path)
        print('INFO::Discriminator Path: ' + discriminator_path)
        discriminator_model.load_state_dict(checkpoint['model_state_dict'])
        discriminator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        e = checkpoint['epoch']
        print('INFO::Discriminator Epochs: ' + str(e))
        discriminator_total_loss = checkpoint['total_loss']
        discriminator_model.train()
        print('INFO::Loaded Checkpoint')
        e = e+1
    print('INFO::Starting Training with epoch :' + str(e))
    for param_group in generator_optimizer.param_groups:
        param_group['lr'] = 0.00005
    for param_group in discriminator_optimizer.param_groups:
        param_group['lr'] = 0.00005
    while e < epochs:
        print('INFO::epoch :' + str(e + 1))
        generator_running_loss = 0
        discriminator_running_loss = 0
        gan_running_loss = 0
        mask_running_loss = 0
        image_running_loss = 0
        tic1 = time.time()
        i = 0
        for input_image, reference in data_loader:
            print("VERBOSE::Epoch: " + str(e+1) + ' Batch: ' + str(i))
            tic2 = time.time()
            i = i + 1
            input_image = input_image.cuda()
            reference = reference.cuda()

            generator_output = generator_model(input_image)
            generated_image = generator_output[:, 0:3, :, :]
            generated_mask = generator_output[:, 3:, :, :]
            cloth_image = input_image[:, -3:, :, :]

            reference_image = reference[:, 0:3, :, :]
            reference_mask = reference[:, 3:, :, :]
            discriminator_loss = 0

            for _ in range(1):
                discriminator_optimizer.zero_grad()

                discriminator_output = discriminator_model(reference_image, cloth_image)
                discriminator_real_loss = bce_loss(discriminator_output, torch.ones(discriminator_output.size()).cuda())

                discriminator_output = discriminator_model(generated_image.detach(), cloth_image)
                discriminator_fake_loss = bce_loss(discriminator_output,
                                                   torch.zeros(discriminator_output.size()).cuda())

                discriminator_loss = (discriminator_fake_loss + discriminator_real_loss) * 0.5

                discriminator_loss.backward()
                discriminator_optimizer.step()

            mask_loss = l1_loss(generated_mask, reference_mask)
            lambda_mask = 10.0
            mask_loss = lambda_mask * mask_loss

            image_loss = l1_loss(generated_image, reference_image)
            lambda_image = 4.76
            image_loss = lambda_image * image_loss

            generator_output = generator_output[:, :3, :, :]
            discriminator_output = discriminator_model(generator_output, cloth_image)

            gan_loss = bce_loss(discriminator_output, torch.ones(discriminator_output.size()).cuda())
            lambda_gan = 0.8
            gan_loss = lambda_gan * gan_loss

            gan_running_loss += gan_loss.item()
            mask_running_loss += mask_loss.item()
            image_running_loss += image_loss.item()
            generator_loss = gan_loss + mask_loss + image_loss
            generator_running_loss += generator_loss.item()
            discriminator_running_loss += discriminator_loss.item()

            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            toc = time.time()
            print('VERBOSE::Batch processing time is %.5f' % (toc - tic2))
        else:
            toc = time.time()
            print('INFO::Epoch processing time is %.5f' % (toc - tic1))
            print('INFO::GAN Running Loss: {:.4f}'.format(gan_running_loss))
            print('INFO::GAN Running Loss Batch: {:.4f}'.format(gan_running_loss / len(data_loader)))
            print('INFO::Mask Running Loss: {:.4f}'.format(mask_running_loss))
            print('INFO::Mask Running Loss Batch: {:.4f}'.format(mask_running_loss / len(data_loader)))
            print('INFO::Image Running Loss: {:.4f}'.format(image_running_loss))
            print('INFO::Image Running Loss Batch: {:.4f}'.format(image_running_loss / len(data_loader)))
            # if e==400:
            #     break

            generator_epoch_loss = generator_running_loss / len(data_loader.dataset)
            generator_total_loss.append(generator_epoch_loss)
            generator_save_path = CURRENT_CHECKPOINT_PATH + '/generator.tar'
            save_checkpoint(e, generator_model, generator_optimizer, generator_running_loss, generator_total_loss,
                            generator_save_path)

            discriminator_epoch_loss = discriminator_running_loss / len(data_loader.dataset)
            discriminator_total_loss.append(discriminator_epoch_loss)
            discriminator_save_path = CURRENT_CHECKPOINT_PATH + '/discriminator.tar'
            save_checkpoint(e, discriminator_model, discriminator_optimizer, discriminator_running_loss,
                            discriminator_total_loss, discriminator_save_path)

            save_image(e, val_data_loader, CURRENT_RESULT_PATH, generator_model)
            plt.plot(generator_total_loss, label='Generator Loss')
            plt.plot(discriminator_total_loss, label='Discriminator Loss')
            plt.legend()
            plt.savefig(CURRENT_RESULT_PATH + '/loss.jpg')
            plt.clf()

            if e % 100 == 0:
                checkpoint_path = GENERATOR_CHECKPOINT_PATH + '/' + str(e).zfill(5) + '.tar'
                save_checkpoint(e, generator_model, generator_optimizer, generator_running_loss, generator_total_loss,
                                checkpoint_path)
                checkpoint_path = DISCRIMINATOR_CHECKPOINT_PATH + '/' + str(e).zfill(5) + '.tar'
                save_checkpoint(e, discriminator_model, discriminator_optimizer, discriminator_running_loss,
                                discriminator_total_loss, checkpoint_path)
                save_image(e, val_data_loader, GENERATOR_RESULT_PATH, generator_model)
            print('INFO::Generator Training Loss: {:.4f}'.format(generator_epoch_loss))
            print('INFO::Discriminator Training Loss: {:.4f}'.format(discriminator_epoch_loss))
            e = e + 1
