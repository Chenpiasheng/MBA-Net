from utils.Drive_loader import get_loader
from model.ccv4 import CCV4


import torch
import os
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.utils.multiclass import type_of_target

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
pth = unet_path = os.path.join("../pth", "Iternet")
net.load_state_dict(torch.load(unet_path))
net.eval()
for i, (images, GT) in enumerate(test_loader):
    images = images.to(device)
    GT = GT.to(device)
    SR = net(images)["final"]

    out = F.sigmoid(SR)
    out = out.cpu().data.numpy()
    y_pred = out.reshape([-1])
    tmp_gt = GT.reshape([-1])
    print(type_of_target(y_pred))
    print(type_of_target(tmp_gt))
    AUC_ROC_UNet = roc_auc_score(tmp_gt, y_pred)
    print("%.4f,   %d" % (AUC_ROC_UNet, i+1))