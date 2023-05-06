import os
import torch.nn.functional as F
from utils.evaluation import *
import csv
import torchvision
from torch import optim


class Solver(object):
    def __init__(self, config, model, train_loader, test_loader, i):

        self.model = model
        self.model_name = config.model_name
        self.val_image_size = config.val_image_size
        self.note = config.note
        self.seed = i

        # Data loader
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.criterion = torch.nn.BCELoss()
        self.lr = config.lr

        # Training settings
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.unet_path = os.path.join(self.model_path, '%d' % (
            i))

        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()
        self.parameter = self.print_network(self.unet, self.model_name)

    def build_model(self):
        """Build generator and discriminator."""
        self.unet = self.model()
        self.unet.to(self.device)
        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, [self.beta1, self.beta2])

    # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        # print(model)
        print(name)
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))

        return format(total)

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def train(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#
        best_unet_score = 0.

        for epoch in range(self.num_epochs):

            self.unet.train(True)
            epoch_loss = 0

            acc = 0.  # Accuracy
            SE = 0.  # Sensitivity (Recall)
            SP = 0.  # Specificity
            PC = 0.  # Precision
            F1 = 0.  # F1 Score
            JS = 0.  # Jaccard Similarity
            DC = 0.  # Dice Coefficient
            length = 0

            for i, (images, GT) in enumerate(self.train_loader):
                # GT : Ground Truth

                images = images.to(self.device)
                GT = GT.to(self.device)
                # edge = edge.to(self.device)

                # SR : Segmentation Result
                SR = self.unet(images)
                SR_probs = F.sigmoid(SR["final"])
                SR_predict1 = F.sigmoid(SR["predict1"])
                SR_predict2 = F.sigmoid(SR["predict2"])
                SR_predict3 = F.sigmoid(SR["predict3"])
                SR_predict4 = F.sigmoid(SR["predict4"])
                SR_predict1_2 = F.sigmoid(SR["predict1_2"])
                SR_predict2_2 = F.sigmoid(SR["predict2_2"])
                SR_predict3_2 = F.sigmoid(SR["predict3_2"])
                SR_predict4_2 = F.sigmoid(SR["predict4_2"])
                # SR_edge = SR["edge"]
                SR = SR["final"]

                SR_flat = SR_probs.view(SR_probs.size(0), -1).to(torch.float32)
                SR_predict1 = SR_predict1.view(SR_predict1.size(0), -1).to(torch.float32)
                SR_predict2 = SR_predict2.view(SR_predict2.size(0), -1).to(torch.float32)
                SR_predict3 = SR_predict3.view(SR_predict3.size(0), -1).to(torch.float32)
                SR_predict4 = SR_predict4.view(SR_predict4.size(0), -1).to(torch.float32)
                SR_predict1_2 = SR_predict1_2.view(SR_predict1_2.size(0), -1).to(torch.float32)
                SR_predict2_2 = SR_predict2_2.view(SR_predict2_2.size(0), -1).to(torch.float32)
                SR_predict3_2 = SR_predict3_2.view(SR_predict3_2.size(0), -1).to(torch.float32)
                SR_predict4_2 = SR_predict4_2.view(SR_predict4_2.size(0), -1).to(torch.float32)

                GT_flat = GT.view(GT.size(0), -1).to(torch.float32)

                # loss_edge = self.criterion(SR_edge_flat, GT_edge_flat) * self.aux_rate
                loss_out = self.criterion(SR_flat, GT_flat)
                loss_predict1 = self.criterion(SR_predict1, GT_flat)
                loss_predict2 = self.criterion(SR_predict2, GT_flat)
                loss_predict3 = self.criterion(SR_predict3, GT_flat)
                loss_predict4 = self.criterion(SR_predict4, GT_flat)
                loss_predict1_2 = self.criterion(SR_predict1_2, GT_flat)
                loss_predict2_2 = self.criterion(SR_predict2_2, GT_flat)
                loss_predict3_2 = self.criterion(SR_predict3_2, GT_flat)
                loss_predict4_2 = self.criterion(SR_predict4_2, GT_flat)
                loss = loss_out + loss_predict1 + loss_predict2 + loss_predict3 + loss_predict4 + loss_predict1_2 + loss_predict2_2 + loss_predict3_2 + loss_predict4_2
                epoch_loss += loss.item()

                # Backprop + optimize
                self.reset_grad()
                loss.backward()
                self.optimizer.step()

                lr = self.optimizer.param_groups[0]["lr"]

                acc += get_accuracy(SR, GT)
                SE += get_sensitivity(SR, GT)
                SP += get_specificity(SR, GT)
                PC += get_precision(SR, GT)
                F1 += get_F1(SR, GT)
                JS += get_JS(SR, GT)
                DC += get_DC(SR, GT)
                length += images.size(0)

            acc = acc / length
            SE = SE / length
            SP = SP / length
            PC = PC / length
            F1 = F1 / length
            JS = JS / length
            DC = DC / length

            # Print the log info
            print(
                'Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                    epoch + 1, self.num_epochs, \
                    epoch_loss, \
                    acc, SE, SP, PC, F1, JS, DC))
            # print('Decay learning rate to lr: {}.'.format(lr))

            # ===================================== Validation ====================================#
            self.unet.train(False)
            self.unet.eval()

            acc = 0.  # Accuracy
            SE = 0.  # Sensitivity (Recall)
            SP = 0.  # Specificity
            PC = 0.  # Precision
            F1 = 0.  # F1 Score
            JS = 0.  # Jaccard Similarity
            DC = 0.  # Dice Coefficient
            length = 0
            for i, (images, GT) in enumerate(self.test_loader):
                images = images.to(self.device)
                GT = GT.to(self.device)
                SR = self.unet(images)["final"]

                acc += get_accuracy(SR, GT)
                SE += get_sensitivity(SR, GT)
                SP += get_specificity(SR, GT)
                PC += get_precision(SR, GT)
                F1 += get_F1(SR, GT)
                JS += get_JS(SR, GT)
                DC += get_DC(SR, GT)

                length += images.size(0)

            acc = acc / length
            SE = SE / length
            SP = SP / length
            PC = PC / length
            F1 = F1 / length
            JS = JS / length
            DC = DC / length
            unet_score = JS + DC

            print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                acc, SE, SP, PC, F1, JS, DC))

            '''
            torchvision.utils.save_image(images.data.cpu(),
                                        os.path.join(self.result_path,
                                                    '%s_valid_%d_image.png'%(self.model_type,epoch+1)))
            torchvision.utils.save_image(SR.data.cpu(),
                                        os.path.join(self.result_path,
                                                    '%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
            torchvision.utils.save_image(GT.data.cpu(),
                                        os.path.join(self.result_path,
                                                    '%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
            '''

            # Save Best U-Net model
            if unet_score > best_unet_score:
                best_unet_score = unet_score
                best_epoch = epoch
                best_unet = self.unet.state_dict()
                print('Best model score : %.4f\n' % (best_unet_score))
                torch.save(best_unet, self.unet_path)

        # ===================================== Test ====================================#
        del self.unet
        del best_unet
        self.build_model()
        self.unet.load_state_dict(torch.load(self.unet_path))

        self.unet.train(False)
        self.unet.eval()

        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        F1 = 0.  # F1 Score
        JS = 0.  # Jaccard Similarity
        DC = 0.  # Dice Coefficient
        AUC = 0.
        length = 0
        for i, (images, GT) in enumerate(self.test_loader):
            images = images.to(self.device)
            GT = GT.to(self.device)
            SR = self.unet(images)["final"]

            acc += get_accuracy(SR, GT)
            SE += get_sensitivity(SR, GT)
            SP += get_specificity(SR, GT)
            PC += get_precision(SR, GT)
            F1 += get_F1(SR, GT)
            JS += get_JS(SR, GT)
            DC += get_DC(SR, GT)
            AUC += get_AUC(SR, GT)

            length += images.size(0)

        acc = acc / length
        SE = SE / length
        SP = SP / length
        PC = PC / length
        F1 = F1 / length
        JS = JS / length
        DC = DC / length
        AUC = AUC / length

        print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, AUC: %.4f' % (
            acc, SE, SP, PC, F1, JS, DC, AUC))

        # torchvision.utils.save_image(SR.data.cpu(),
        #                              os.path.join(self.result_path,
        #                                           'res.png'))

        f = open(os.path.join(self.result_path, self.model_name + '.csv'), 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
        wr.writerow(
            [self.seed, acc, DC, SE, SP, PC, F1, JS, AUC])
        f.close()

        return [acc, DC, SE, SP, PC, F1, JS, AUC]
