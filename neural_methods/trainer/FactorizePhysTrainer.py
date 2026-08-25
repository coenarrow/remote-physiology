"""
FactorizePhys: Matrix Factorization for Multidimensional Attention in Remote Physiological Sensing
NeurIPS 2024
Jitesh Joshi, Sos S. Agaian, and Youngjun Cho
"""

import os
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from evaluation.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.FactorizePhys.FactorizePhys import FactorizePhys
from neural_methods.model.FactorizePhys.FactorizePhysBig import FactorizePhysBig
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm


class FactorizePhysTrainer(BaseTrainer):

    def __init__(self, config, data_loader, **kwargs):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__(**kwargs)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.dropout_rate = config.MODEL.DROP_RATE
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        self.device = torch.device(f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu')

        frames = self.config.MODEL.FactorizePhys.FRAME_NUM
        in_channels = self.config.MODEL.FactorizePhys.CHANNELS
        model_type = self.config.MODEL.FactorizePhys.TYPE
        model_type = model_type.lower()

        md_config = {}
        md_config["FRAME_NUM"] = self.config.MODEL.FactorizePhys.FRAME_NUM
        md_config["MD_TYPE"] = self.config.MODEL.FactorizePhys.MD_TYPE
        md_config["MD_FSAM"] = self.config.MODEL.FactorizePhys.MD_FSAM
        md_config["MD_TRANSFORM"] = self.config.MODEL.FactorizePhys.MD_TRANSFORM
        md_config["MD_S"] = self.config.MODEL.FactorizePhys.MD_S
        md_config["MD_R"] = self.config.MODEL.FactorizePhys.MD_R
        md_config["MD_STEPS"] = self.config.MODEL.FactorizePhys.MD_STEPS
        md_config["MD_INFERENCE"] = self.config.MODEL.FactorizePhys.MD_INFERENCE
        md_config["MD_RESIDUAL"] = self.config.MODEL.FactorizePhys.MD_RESIDUAL

        self.md_infer = self.config.MODEL.FactorizePhys.MD_INFERENCE
        self.use_fsam = self.config.MODEL.FactorizePhys.MD_FSAM

        if model_type == "standard":
            self.model = FactorizePhys(frames=frames, md_config=md_config, in_channels=in_channels,
                                    dropout=self.dropout_rate, device=self.device)  # [3, T, 72,72]
        elif model_type == "big":
            self.model = FactorizePhysBig(frames=frames, md_config=md_config, in_channels=in_channels,
                                       dropout=self.dropout_rate, device=self.device)  # [3, T, 144,144]
        else:
            print("Unexpected model type specified. Should be standard or big, but specified:", model_type)
            exit()

        self.model = self.model.to(self.device)
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank)

        if self.config.TOOLBOX_MODE == "train_and_test" or self.config.TOOLBOX_MODE == "only_train":
            self.num_train_batches = len(data_loader["train"])
            self.criterion = Neg_Pearson()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.config.TRAIN.LR)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.config.TRAIN.LR, epochs=self.config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)

            if self.world_size > 1 and data_loader["train"] is not None:
                self.train_sampler = data_loader["train"].sampler
            else:
                self.train_sampler = None
        elif self.config.TOOLBOX_MODE == "only_test":
            self.train_sampler = None
        else:
            raise ValueError("FactorizePhys trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        if self.world_size > 1:
            dist.barrier()

        mean_training_losses = []
        mean_valid_losses = []
        mean_appx_error = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            if self.is_main:
                print('')
                print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            appx_error_list = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80) if self.is_main else data_loader["train"]
            for idx, batch in enumerate(tbar):
                if self.is_main:
                    tbar.set_description("Train epoch %s" % epoch)

                data = batch[0].to(self.device)
                labels = batch[1].to(self.device)

                if len(labels.shape) > 2:
                    labels = labels[..., 0]     # Compatibility wigth multi-signal labelled data
                labels = (labels - torch.mean(labels)) / torch.std(labels)  # normalize
                last_frame = torch.unsqueeze(data[:, :, -1, :, :], 2).repeat(1, 1, 1, 1, 1)
                data = torch.cat((data, last_frame), 2)

                self.optimizer.zero_grad()
                if self.model.training and self.use_fsam:
                    pred_ppg, vox_embed, factorized_embed, appx_error = self.model(data)
                else:
                    pred_ppg, vox_embed = self.model(data)

                pred_ppg = (pred_ppg - torch.mean(pred_ppg)) / torch.std(pred_ppg)  # normalize

                loss = self.criterion(pred_ppg, labels)

                loss.backward()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    if self.is_main:
                        print(
                            f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                if self.use_fsam:
                    appx_error_list.append(appx_error.item())

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()

                if self.is_main:
                    if self.use_fsam:
                        tbar.set_postfix({"appx_error": appx_error.item()}, loss=loss.item())
                    else:
                        tbar.set_postfix(loss=loss.item())

            # Aggregate training loss across ranks
            if self.world_size > 1:
                loss_tensor = torch.tensor([np.mean(train_loss)], device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                epoch_train_loss = loss_tensor.cpu().item()
            else:
                epoch_train_loss = np.mean(train_loss)
            mean_training_losses.append(epoch_train_loss)

            if self.use_fsam:
                mean_appx_error.append(np.mean(appx_error_list))
                if self.is_main:
                    print("Mean train loss: {}, Mean appx error: {}".format(
                        epoch_train_loss, np.mean(appx_error_list)))
            else:
                if self.is_main:
                    print("Mean train loss: {}".format(epoch_train_loss))

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                if self.is_main:
                    print('validation loss: ', valid_loss)
                    if self.min_valid_loss is None:
                        self.min_valid_loss = valid_loss
                        self.best_epoch = epoch
                        print("Update best model! Best epoch: {}".format(self.best_epoch))
                    elif (valid_loss < self.min_valid_loss):
                        self.min_valid_loss = valid_loss
                        self.best_epoch = epoch
                        print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH:
            if self.is_main:
                print("best trained epoch: {}, min_val_loss: {}".format(
                    self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR and self.is_main:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        if self.is_main:
            print('')
            print(" ====Validing===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80) if self.is_main else data_loader["valid"]
            for valid_idx, valid_batch in enumerate(vbar):
                if self.is_main:
                    vbar.set_description("Validation")

                data, labels = valid_batch[0].to(self.device), valid_batch[1].to(self.device)
                if len(labels.shape) > 2:
                    labels = labels[..., 0]     # Compatibility wigth multi-signal labelled data
                labels = (labels - torch.mean(labels)) / torch.std(labels)  # normalize

                last_frame = torch.unsqueeze(data[:, :, -1, :, :], 2).repeat(1, 1, 1, 1, 1)
                data = torch.cat((data, last_frame), 2)

                if self.md_infer and self.use_fsam:
                    pred_ppg, vox_embed, factorized_embed, appx_error = self.model(data)
                else:
                    pred_ppg, vox_embed = self.model(data)
                pred_ppg = (pred_ppg - torch.mean(pred_ppg)) / torch.std(pred_ppg)  # normalize
                loss = self.criterion(pred_ppg, labels)

                valid_loss.append(loss.item())
                valid_step += 1
                if self.is_main:
                    if self.md_infer and self.use_fsam:
                        vbar.set_postfix({"appx_error": appx_error.item()}, loss=loss.item())
                    else:
                        vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)

        # Aggregate validation loss across ranks
        if self.world_size > 1:
            loss_tensor = torch.tensor([np.mean(valid_loss)], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            return loss_tensor.cpu().item()
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Runs the model on test sets."""
        if not self.is_main:
            return

        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing===")

        # Unwrap DDP for loading and inference
        model_to_load = self._unwrap_model()

        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            model_to_load.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH, map_location=self.device), strict=False)
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                model_to_load.load_state_dict(torch.load(last_epoch_model_path, map_location=self.device), strict=False)
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                model_to_load.load_state_dict(torch.load(best_model_path, map_location=self.device), strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, labels_test = test_batch[0].to(self.device), test_batch[1].to(self.device)

                if len(labels_test.shape) > 2:
                    labels_test = labels_test[..., 0]     # Compatibility wigth multi-signal labelled data
                labels_test = (labels_test - torch.mean(labels_test)) / torch.std(labels_test)  # normalize

                last_frame = torch.unsqueeze(data[:, :, -1, :, :], 2).repeat(1, 1, 1, 1, 1)
                data = torch.cat((data, last_frame), 2)

                if self.md_infer and self.use_fsam:
                    pred_ppg_test, vox_embed, factorized_embed, appx_error = self.model(data)
                else:
                    pred_ppg_test, vox_embed = self.model(data)
                pred_ppg_test = (pred_ppg_test - torch.mean(pred_ppg_test)) / torch.std(pred_ppg_test)  # normalize

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    labels_test = labels_test.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = labels_test[idx]


        print('')
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        if not self.is_main:
            return
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        model_to_save = self._unwrap_model()
        torch.save(model_to_save.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
