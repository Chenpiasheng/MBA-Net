from utils.Drive_loader import get_loader
from model.ccv4 import CCV4
import torch
import os
import torchvision

test_loader = get_loader(image_path="../data",
                         image_size=560,
                         val_image_size=560,
                         batch_size=1,
                         num_workers=0,
                         mode='test',
                         shuffle=False,
                         )
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = CCV4().to(device)
pth = unet_path = os.path.join("../pth", "CCV4")
net.load_state_dict(torch.load(unet_path))
net.eval()
for i, (images, GT) in enumerate(test_loader):
    images = images.to(device)
    GT = GT.to(device)
    SR = net(images)['predict3_2']
    torchvision.utils.save_image(SR.data.cpu(), os.path.join("../data", '%d.png') % (i+1))

    print(i)
