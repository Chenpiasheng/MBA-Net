import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve


# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    acc = float(corr) / float(tensor_size)

    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    SR = F.sigmoid(SR)

    SR_t = SR > threshold
    SR_f = SR < threshold
    GT_t = GT == torch.max(GT)
    GT_f = GT == torch.min(GT)

    SR_t = SR_t.type(torch.float)
    SR_f = SR_f.type(torch.float)
    GT_t = GT_t.type(torch.float)
    GT_f = GT_f.type(torch.float)

    TP = GT_t * SR_t
    TP = TP.type(torch.float)
    FN = GT_t * SR_f
    FN = FN.type(torch.float)

    SE = float(torch.sum(TP)) / (float(torch.sum(TP) + torch.sum(FN)) + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = F.sigmoid(SR)

    SR_t = SR > threshold
    SR_f = SR < threshold
    GT_t = GT == torch.max(GT)
    GT_f = GT == torch.min(GT)

    SR_t = SR_t.type(torch.float)
    SR_f = SR_f.type(torch.float)
    GT_t = GT_t.type(torch.float)
    GT_f = GT_f.type(torch.float)

    TN = SR_f * GT_f
    TN = TN.type(torch.float)
    FP = SR_t * GT_f
    FP = FP.type(torch.float)

    SP = float(torch.sum(TN)) / (float(torch.sum(TN) + torch.sum(FP)) + 1e-6)

    return SP


def get_precision(SR, GT, threshold=0.5):
    SR = F.sigmoid(SR)

    SR_t = SR > threshold
    SR_f = SR < threshold
    GT_t = GT == torch.max(GT)
    GT_f = GT == torch.min(GT)

    SR_t = SR_t.type(torch.float)
    SR_f = SR_f.type(torch.float)
    GT_t = GT_t.type(torch.float)
    GT_f = GT_f.type(torch.float)

    TP = SR_t * GT_t
    TP = TP.type(torch.float) * 2.0
    FP = SR_t * GT_f
    FP = FP.type(torch.float) * 2.0

    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)
    SR = SR.type(torch.float)
    GT = GT.type(torch.float)

    Inter = torch.sum((SR + GT) == 2)
    Union = torch.sum((SR + GT) >= 1)

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    SR = SR.type(torch.float)
    GT = GT.type(torch.float)

    Inter = torch.sum((SR + GT) == 2)
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC


def get_AUC(SR, GT, threshold=0.5):
    GT = GT.cpu().numpy().reshape(560, 560).astype(int)
    out = torch.sigmoid(SR)
    out = out.cpu().data.numpy()
    y_pred = out.reshape([-1])
    tmp_gt = GT.reshape([-1])

    # fpr1, tpr1, thresholds = roc_curve((tmp_gt), y_pred)
    AUC_ROC = roc_auc_score(tmp_gt, y_pred)

    return AUC_ROC


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):

        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
