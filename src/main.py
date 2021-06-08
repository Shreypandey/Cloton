import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from constants import PROJECT_PATH, GENERATOR_BATCH_SIZE, VALIDATION_BATCH_SIZE, GENERATOR_CHECKPOINT, \
    DISCRIMINATOR_CHECKPOINT, GENERATOR_RESULT_PATH
from datasets.fake_images_dataset import FakeDataset
from datasets.generator_dataset import ClothDataset
from datasets.real_images_dataset import RealDataset
from inception import inception_score
from models.discriminator import Discriminator
from models.generator import Generator
from testgenerator import test
# from tps import tps_transform
from traingenerator import train
from utils.checkpoint import initialize_checkpoint

from oct2py import octave

print(torch.cuda.is_available())
print(torch.cuda.device_count())

data_path = PROJECT_PATH + '/dataset'
transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
#
dataset_size = 2000
# cloth_dataset = ClothDataset(data_path, dataset_size, transform_input=transform_image, transform_cloth=transform_image)
# val_dataset = ClothDataset(data_path, dataset_size, mode='test', transform_input=transform_image, transform_cloth=transform_image)
#
# cloth_data_loader = DataLoader(cloth_dataset, batch_size=GENERATOR_BATCH_SIZE, shuffle=False)
# val_data_loader = DataLoader(val_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=False)
#
# generator_model = Generator()
# generator_model = generator_model.cuda()
# discriminator_model = Discriminator()
# discriminator_model = discriminator_model.cuda()
#
# generator_optimizer = torch.optim.Adam(generator_model.parameters(), lr=0.00002, betas=(0.5, 0.999))
# discriminator_optimizer = torch.optim.Adam(discriminator_model.parameters(), lr=0.00002, betas=(0.5, 0.999))

# initialize_checkpoint(generator_model, generator_optimizer, GENERATOR_CHECKPOINT)
# initialize_checkpoint(discriminator_model, discriminator_optimizer, DISCRIMINATOR_CHECKPOINT)

# epochs = 500
# train(epochs, generator_model=generator_model, discriminator_model=discriminator_model, generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, data_loader=cloth_data_loader, val_data_loader=val_data_loader)
# test(100, generator_model, cloth_data_loader, val_data_loader)
# test(200, generator_model, cloth_data_loader, val_data_loader)
# test(300, generator_model, val_data_loader)
# test(400, generator_model, val_data_loader)
# test(400, generator_model, cloth_data_loader, val_data_loader)
fake_data_path = GENERATOR_RESULT_PATH
print(data_path)
real_image_dataset = RealDataset(data_path, 2032, mode='test', transform_input=transform_image)
print('VERBOSE::Inception Score for real dataset is')
print(inception_score(real_image_dataset, resize=True))
fake_image_dataset = FakeDataset(fake_data_path, dataset_size, transform_input=transform_image)
print('VERBOSE::Inception Score for fake dataset is')
print(inception_score(fake_image_dataset, resize=True))
# for i in range(32):
#     # octave.shape_content(PROJECT_PATH, 'test', str(i).zfill(5)+'.jpg', str(i).zfill(5)+'.jpg')
#     tps_transform(image_name=i, cloth=str(i).zfill(5) + '.jpg', control_point=str(i).zfill(5) + '.jpg_' + str(i).zfill(5) + '.jpg_tps.mat')
