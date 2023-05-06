import argparse
import os
from torch.backends import cudnn
import csv
import numpy as np
import torch
import random

from solver4 import Solver
from utils.Drive_loader import get_loader
from model.ccv4 import CCV4  # solver4


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(config, model, i):
    cudnn.benchmark = True

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    print(config)

    train_loader = get_loader(image_path=config.train_path,
                              image_size=config.image_size,
                              val_image_size=config.val_image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              shuffle=True,
                              )

    test_loader = get_loader(image_path=config.test_path,
                             image_size=config.image_size,
                             val_image_size=config.val_image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             mode='test',
                             shuffle=False,
                             )

    solver = Solver(config, model, train_loader, test_loader, i)
    return solver.train()


if __name__ == '__main__':
    model = CCV4
    model_name = "CCV4"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=model_name)
    parser.add_argument('--note', type=str,
                        default="")

    # 超参数
    parser.add_argument('--image_size', type=int, default=240)
    parser.add_argument('--val_image_size', type=int, default=560)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0002)

    # 路径
    parser.add_argument('--model_path', type=str, default=os.path.join("pth", model_name))
    parser.add_argument('--train_path', type=str, default=r".\data")
    parser.add_argument('--valid_path', type=str, default=r".\data")
    parser.add_argument('--test_path', type=str, default=r".\data")
    parser.add_argument('--result_path', type=str, default=os.path.join("results"))

    # 优化器
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()

    for i in range(10):
        setup_seed(i)
        a = np.array(main(config, model, i))
        if i == 0:
            c = np.stack(a, axis=0)
        else:
            c = np.row_stack((c, a))
        print(c)

    acc_mean = np.mean(c[:, 0])
    acc_std = np.std(c[:, 0], ddof=1)
    acc = '%.2f ± %.2f' % (acc_mean * 100, acc_std * 100)
    DC_mean = np.mean(c[:, 1])
    DC_std = np.std(c[:, 1], ddof=1)
    DC = '%.2f ± %.2f' % (DC_mean * 100, DC_std * 100)
    SE_mean = np.mean(c[:, 2])
    SE_std = np.std(c[:, 2], ddof=1)
    SE = '%.2f ± %.2f' % (SE_mean * 100, SE_std * 100)
    SP_mean = np.mean(c[:, 3])
    SP_std = np.std(c[:, 3], ddof=1)
    SP = '%.2f ± %.2f' % (SP_mean * 100, SP_std * 100)
    PC_mean = np.mean(c[:, 4])
    PC_std = np.std(c[:, 4], ddof=1)
    PC = '%.2f ± %.2f' % (PC_mean * 100, PC_std * 100)
    F1_mean = np.mean(c[:, 5])
    F1_std = np.std(c[:, 5], ddof=1)
    F1 = '%.2f ± %.2f' % (F1_mean * 100, F1_std * 100)
    JS_mean = np.mean(c[:, 6])
    JS_std = np.std(c[:, 6], ddof=1)
    JS = '%.2f ± %.2f' % (JS_mean * 100, JS_std * 100)
    AUC_mean = np.mean(c[:, 7])
    AUC_std = np.std(c[:, 7], ddof=1)
    AUC = '%.2f ± %.2f' % (AUC_mean * 100, AUC_std * 100)

    f = open(os.path.join(config.result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(
        [config.model_name, config.batch_size, config.val_image_size, config.num_epochs, acc, DC, SE, SP, PC, F1, JS,
         AUC, config.note
         ])
    f.close()
    print(acc)
