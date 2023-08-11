import argparse

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToPILImage, ToTensor

from model import GeneratorRRDB

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['CPU', 'GPU'], help='using CPU or GPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='generator_199.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

model = GeneratorRRDB(channels=3, num_res_blocks=23).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('saved_models/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('saved_models/' + MODEL_NAME, map_location=lambda storage, loc: storage))

image = Image.open(IMAGE_NAME)
image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
if TEST_MODE:
    image = image.cuda()

out = model(image)
out = torch.clamp(out, min=0, max=1)
out_image = ToPILImage()(out[0].data.cpu())
out_image.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)