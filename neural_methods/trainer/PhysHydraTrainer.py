"""PhysHydra Trainer."""
import os
from collections import OrderedDict

import math
import numpy as np
import torch
import torch.optim as optim
import random
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysHydraLoss import PhysHydraLoss
from neural_methods.model.PhysHydra import PhysHydra
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm
from scipy.signal import welch


class PhysHydraTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.diff_flag = 0
        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            self.diff_flag = 1
        self.num_channels = config.MODEL.PHYSHYDRA.NUM_CHANNELS
        self.num_labels = config.MODEL.PHYSHYDRA.NUM_LABELS
        self.frame_rate = config.TRAIN.DATA.FS
        self.num_frames = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH

        self.model = PhysHydra(
            in_channels=self.num_channels, 
            out_signals=self.num_labels,
            frames=self.num_frames
        ).to(self.device)  # [3, T, 128,128]
        if self.num_of_gpu > 0:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.loss_class = PhysHydraLoss()
            self.optimizer = optim.Adam(self.model.parameters(), 
                                        lr=config.TRAIN.LR, 
                                        weight_decay = 0.0005)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=config.TRAIN.LR, 
                epochs=config.TRAIN.EPOCHS, 
                steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            self.loss_class = PhysHydraLoss()
            pass
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            self.model.train()
            running_loss = 0.0
            tbar = tqdm(data_loader["train"], ncols=80)
            
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].float(), batch[1].float()
                N, D, C, H, W = data.shape
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                pred_ppg = self.model(data)  # [batch, num_signals, frames]

                if pred_ppg.dim() == 2:
                    pred_ppg = pred_ppg.unsqueeze(1)
                if labels.dim() == 2:
                    labels = labels.unsqueeze(1)

                # If labels came as [N,T,C], move to [N,C,T]
                if labels.shape[-1] < labels.shape[1]:  # e.g., [N, T, 1] -> likely wrong
                    labels = labels.transpose(1, 2)
                # If preds came as [N,T,C], move to [N,C,T]
                if pred_ppg.shape[-1] < pred_ppg.shape[1]:
                    pred_ppg = pred_ppg.transpose(1, 2)

                # Hard checks
                assert pred_ppg.shape[0] == labels.shape[0], "Batch mismatch"
                assert pred_ppg.shape[1] == labels.shape[1], f"Channel mismatch: {pred_ppg.shape[1]} vs {labels.shape[1]}"

                # One-time per epoch debug print
                if idx == 0:
                    Tp, Tl = pred_ppg.size(-1), labels.size(-1)
                    print(f"[epoch {epoch}] T_pred={Tp}, T_lab={Tl}, C={pred_ppg.shape[1]}")
                
                # Ensure labels have same shape as predictions
                if labels.dim() == 2 and self.num_labels == 1:
                    labels = labels.unsqueeze(1)  # [batch, 1, frames]
                
                loss = self.loss_class(pred_ppg, labels)
                loss.backward()
                running_loss += loss.item()
                
                if idx % 100 == 99:
                    print(f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                    
                self.optimizer.step()
                self.scheduler.step()
                tbar.set_postfix(loss=loss.item())
            
            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
            torch.cuda.empty_cache()

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")
        print('')
        print(" ====Validing===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                BVP_label = valid_batch[1].to(torch.float32).to(self.device)
                rPPG = self.model(valid_batch[0].to(torch.float32).to(self.device))
                signal_loss = self.loss_class(rPPG, BVP_label)
                valid_loss.append(signal_loss.item())
                valid_step += 1
                vbar.set_postfix(loss=signal_loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data = test_batch[0].to(self.config.DEVICE)
                label = test_batch[1].to(self.config.DEVICE)
                
                pred_ppg_test = self.model(data)  # [batch, num_signals, frames]
                
                # Ensure labels match prediction shape
                if label.dim() == 2 and self.num_labels == 1:
                    label = label.unsqueeze(1)
                
                if self.config.TEST.OUTPUT_SAVE_DIR:
                    label = label.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()
                
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    # Store all signals
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]  # [num_signals, frames]
                    labels[subj_index][sort_index] = label[idx]  # [num_signals, frames]
 

        print('')
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs 
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)