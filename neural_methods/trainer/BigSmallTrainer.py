"""Trainer for BigSmall Multitask Models"""

# Training / Eval Imports
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods import loss
from neural_methods.model.BigSmall import BigSmall
from evaluation.bigsmall_multitask_metrics import (calculate_bvp_metrics,
                                                   calculate_resp_metrics,
                                                   calculate_bp4d_au_metrics)

# Other Imports
from collections import OrderedDict
import numpy as np
import os
from tqdm import tqdm

class BigSmallTrainer(BaseTrainer):

    def define_model(self, config):

        # BigSmall Model
        model = BigSmall(n_segment=3)

        if self.using_TSM:
            self.frame_depth = config.MODEL.BIGSMALL.FRAME_DEPTH
            self.base_len = self.frame_depth

        return model

    def format_data_shape(self, data, labels):
        # reshape big data
        data_big = data[0]
        N, D, C, H, W = data_big.shape
        data_big = data_big.view(N * D, C, H, W)

        # reshape small data
        data_small = data[1]
        N, D, C, H, W = data_small.shape
        data_small = data_small.view(N * D, C, H, W)

        # reshape labels
        if len(labels.shape) != 3: # this training format requires labels that are of shape N_label, D_label, C_label
            labels = torch.unsqueeze(labels, dim=-1)
        N_label, D_label, C_label = labels.shape
        labels = labels.view(N_label * D_label, C_label)

        # If using temporal shift module
        if self.using_TSM:
            data_big = data_big[:(N * D) // self.base_len * self.base_len]
            data_small = data_small[:(N * D) // self.base_len * self.base_len]
            labels = labels[:(N * D) // self.base_len * self.base_len]

        data[0] = data_big
        data[1] = data_small
        labels = torch.unsqueeze(labels, dim=-1)

        return data, labels


    def send_data_to_device(self, data, labels):
        big_data = data[0].to(self.device)
        small_data = data[1].to(self.device)
        labels = labels.to(self.device)
        data = (big_data, small_data)
        return data, labels


    def get_label_idxs(self, label_list, used_labels):
        label_idxs = []
        for l in used_labels:
            idx = label_list.index(l)
            label_idxs.append(idx)
        return label_idxs


    def save_model(self, index):
        if not self.is_main:
            return
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        model_to_save = self._unwrap_model()
        torch.save(model_to_save.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
        print('')


    def __init__(self, config, data_loader, **kwargs):

        print('')
        print('Init BigSmall Multitask Trainer\n\n')

        super().__init__(**kwargs)

        self.config = config

        # Set up device via DDP rank
        self.device = torch.device(f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu')

        # Defining model
        self.using_TSM = True
        self.model = self.define_model(config)
        self.model = self.model.to(self.device)

        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank)

        # Training parameters
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.LR = config.TRAIN.LR

        # Set Loss and Optimizer
        AU_weights = torch.as_tensor([9.64, 11.74, 16.77, 1.05, 0.53, 0.56,
                                      0.75, 0.69, 8.51, 6.94, 5.03, 25.00]).to(self.device)

        self.criterionAU = torch.nn.BCEWithLogitsLoss(pos_weight=AU_weights).to(self.device)
        self.criterionBVP = torch.nn.MSELoss().to(self.device)
        self.criterionRESP = torch.nn.MSELoss().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.LR, weight_decay=0)

        # Scheduler (was missing — Bug 1 fix)
        self.num_train_batches = len(data_loader["train"])
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.LR, epochs=self.max_epoch_num,
            steps_per_epoch=self.num_train_batches)

        # Model info (saved more dir, chunk len, best epoch, etc.)
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH

        # Model selection state (Bug 2 fix — replaces undefined self.model_to_use)
        self.min_valid_loss = None
        self.best_epoch = 0

        # Track distributed sampler
        if self.world_size > 1 and data_loader["train"] is not None:
            self.train_sampler = data_loader["train"].sampler
        else:
            self.train_sampler = None

        # Indicies corresponding to used labels
        label_list = ['bp_wave', 'HR_bpm', 'systolic_bp', 'diastolic_bp', 'mean_bp',
                      'resp_wave', 'resp_bpm', 'eda',
                      'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU06int', 'AU07', 'AU09', 'AU10', 'AU10int',
                      'AU11', 'AU12', 'AU12int', 'AU13', 'AU14', 'AU14int', 'AU15', 'AU16', 'AU17', 'AU17int',
                      'AU18', 'AU19', 'AU20', 'AU22', 'AU23', 'AU24', 'AU27', 'AU28', 'AU29', 'AU30', 'AU31',
                      'AU32', 'AU33', 'AU34', 'AU35', 'AU36', 'AU37', 'AU38', 'AU39',
                      'pos_bvp','pos_env_norm_bvp']

        used_labels = ['bp_wave', 'AU01', 'AU02', 'AU04', 'AU06', 'AU07', 'AU10', 'AU12',
                       'AU14', 'AU15', 'AU17', 'AU23', 'AU24',
                        'pos_env_norm_bvp', 'resp_wave']

        # Get indicies for labels from npy array
        au_label_list = [label for label in used_labels if 'AU' in label]
        bvp_label_list_train = [label for label in used_labels if 'bvp' in label]
        bvp_label_list_test = [label for label in used_labels if 'bp_wave' in label]
        resp_label_list = [label for label in used_labels if 'resp' in label]

        self.label_idx_train_au = self.get_label_idxs(label_list, au_label_list)
        self.label_idx_valid_au = self.get_label_idxs(label_list, au_label_list)
        self.label_idx_test_au = self.get_label_idxs(label_list, au_label_list)

        self.label_idx_train_bvp = self.get_label_idxs(label_list, bvp_label_list_train)
        self.label_idx_valid_bvp = self.get_label_idxs(label_list, bvp_label_list_train)
        self.label_idx_test_bvp = self.get_label_idxs(label_list, bvp_label_list_test)

        self.label_idx_train_resp = self.get_label_idxs(label_list, resp_label_list)
        self.label_idx_valid_resp = self.get_label_idxs(label_list, resp_label_list)
        self.label_idx_test_resp = self.get_label_idxs(label_list, resp_label_list)


    def train(self, data_loader):
        """Model Training"""

        if data_loader["train"] is None:
            raise ValueError("No data for train")

        if self.world_size > 1:
            dist.barrier()

        if self.is_main:
            print('Starting Training Routine')
            print('')

        # ARRAYS TO SAVE (LOSS ARRAYS)
        train_loss_dict = dict()
        train_au_loss_dict = dict()
        train_bvp_loss_dict = dict()
        train_resp_loss_dict = dict()

        val_loss_dict = dict()
        val_au_loss_dict = dict()
        val_bvp_loss_dict = dict()
        val_resp_loss_dict = dict()

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []

        # ITERATE THROUGH EPOCHS
        for epoch in range(self.max_epoch_num):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            if self.is_main:
                print(f"====Training Epoch: {epoch}====")

            # INIT PARAMS FOR TRAINING
            running_loss = 0.0
            train_loss = []
            train_au_loss = []
            train_bvp_loss = []
            train_resp_loss = []
            self.model.train()

            # MODEL TRAINING
            tbar = tqdm(data_loader["train"], ncols=80) if self.is_main else data_loader["train"]
            for idx, batch in enumerate(tbar):
                if self.is_main:
                    tbar.set_description("Train epoch %s" % epoch)

                # GATHER AND FORMAT BATCH DATA
                data, labels = batch[0], batch[1]
                data, labels = self.format_data_shape(data, labels)
                data, labels = self.send_data_to_device(data, labels)

                # FOWARD AND BACK PROPOGATE THROUGH MODEL
                self.optimizer.zero_grad()
                au_out, bvp_out, resp_out = self.model(data)
                au_loss = self.criterionAU(au_out, labels[:, self.label_idx_train_au, 0])
                bvp_loss = self.criterionBVP(bvp_out, labels[:, self.label_idx_train_bvp, 0])
                resp_loss = self.criterionRESP(resp_out, labels[:, self.label_idx_train_resp, 0])
                loss = au_loss + bvp_loss + resp_loss
                loss.backward()

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()

                # UPDATE RUNNING LOSS AND PRINTED TERMINAL OUTPUT AND SAVED LOSSES
                train_loss.append(loss.item())
                train_au_loss.append(au_loss.item())
                train_bvp_loss.append(bvp_loss.item())
                train_resp_loss.append(resp_loss.item())

                running_loss += loss.item()
                if idx % 100 == 99:
                    if self.is_main:
                        print(f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

                if self.is_main:
                    tbar.set_postfix({"loss:": loss.item(), "lr:": self.optimizer.param_groups[0]["lr"]})

            # APPEND EPOCH LOSS LIST TO TRAINING LOSS DICTIONARY
            train_loss_dict[epoch] = train_loss
            train_au_loss_dict[epoch] = train_au_loss
            train_bvp_loss_dict[epoch] = train_bvp_loss
            train_resp_loss_dict[epoch] = train_resp_loss

            # Aggregate training loss across ranks
            if self.world_size > 1:
                loss_tensor = torch.tensor([np.mean(train_loss)], device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                epoch_train_loss = loss_tensor.cpu().item()
            else:
                epoch_train_loss = np.mean(train_loss)
            mean_training_losses.append(epoch_train_loss)

            # SAVE MODEL FOR THIS EPOCH
            self.save_model(epoch)

            # VALIDATION (IF ENABLED)
            if not self.config.TEST.USE_LAST_EPOCH:

                # Get validation losses
                valid_loss, valid_au_loss, valid_bvp_loss, valid_resp_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                val_loss_dict[epoch] = valid_loss
                val_au_loss_dict[epoch] = valid_au_loss
                val_bvp_loss_dict[epoch] = valid_bvp_loss
                val_resp_loss_dict[epoch] = valid_resp_loss

                if self.is_main:
                    print('validation loss: ', valid_loss)
                    if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                        self.min_valid_loss = valid_loss
                        self.best_epoch = epoch
                        print("Update best model! Best epoch: {}".format(self.best_epoch))

            # VALIDATION (NOT ENABLED)
            else:
                self.best_epoch = epoch

            torch.cuda.empty_cache()

            if self.is_main:
                print('')

        if self.config.TRAIN.PLOT_LOSSES_AND_LR and self.is_main:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

        # PRINT MODEL TO BE USED FOR TESTING
        if self.is_main:
            if not self.config.TEST.USE_LAST_EPOCH:
                print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))
            print('')



    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""

        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        if self.is_main:
            print("===Validating===")

        # INIT PARAMS FOR VALIDATION
        valid_loss = []
        valid_au_loss = []
        valid_bvp_loss = []
        valid_resp_loss = []
        self.model.eval()

        # MODEL VALIDATION
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80) if self.is_main else data_loader["valid"]
            for valid_idx, valid_batch in enumerate(vbar):
                if self.is_main:
                    vbar.set_description("Validation")

                # GATHER AND FORMAT BATCH DATA
                data, labels = valid_batch[0], valid_batch[1]
                data, labels = self.format_data_shape(data, labels)
                data, labels = self.send_data_to_device(data, labels)

                au_out, bvp_out, resp_out = self.model(data)
                au_loss = self.criterionAU(au_out, labels[:, self.label_idx_valid_au, 0])
                bvp_loss = self.criterionBVP(bvp_out, labels[:, self.label_idx_valid_bvp, 0])
                resp_loss = self.criterionRESP(resp_out, labels[:, self.label_idx_valid_resp, 0])
                loss = au_loss + bvp_loss + resp_loss

                # APPEND VAL LOSS
                valid_loss.append(loss.item())
                valid_au_loss.append(au_loss.item())
                valid_bvp_loss.append(bvp_loss.item())
                valid_resp_loss.append(resp_loss.item())
                if self.is_main:
                    vbar.set_postfix(loss=loss.item())

        valid_loss = np.asarray(valid_loss)
        valid_au_loss = np.asarray(valid_au_loss)
        valid_bvp_loss = np.asarray(valid_bvp_loss)
        valid_resp_loss = np.asarray(valid_resp_loss)

        # Aggregate validation losses across ranks
        if self.world_size > 1:
            losses = torch.tensor([np.mean(valid_loss), np.mean(valid_au_loss),
                                   np.mean(valid_bvp_loss), np.mean(valid_resp_loss)],
                                  device=self.device)
            dist.all_reduce(losses, op=dist.ReduceOp.AVG)
            vals = losses.cpu().numpy()
            return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
        return np.mean(valid_loss), np.mean(valid_au_loss), np.mean(valid_bvp_loss), np.mean(valid_resp_loss)



    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""

        if not self.is_main:
            return

        print("===Testing===")
        print('')

        # SETUP
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        # Unwrap DDP for loading and inference
        model_to_load = self._unwrap_model()

        # Change chunk length to be test chunk length
        self.chunk_len = self.config.TEST.DATA.PREPROCESS.CHUNK_LENGTH

        # ARRAYS TO SAVE (PREDICTIONS AND METRICS ARRAYS)
        preds_dict_au = dict()
        labels_dict_au = dict()
        preds_dict_bvp = dict()
        labels_dict_bvp = dict()
        preds_dict_resp = dict()
        labels_dict_resp = dict()

        # IF ONLY_TEST MODE LOAD PRETRAINED MODEL
        if self.config.TOOLBOX_MODE == "only_test":
            model_path = self.config.INFERENCE.MODEL_PATH
            print("Testing uses pretrained model!")
            print('Model path:', model_path)
            if not os.path.exists(model_path):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")

        # IF USING MODEL FROM TRAINING
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                model_path = os.path.join(self.model_dir,
                                          self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
            else:
                model_path = os.path.join(self.model_dir,
                                          self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
            print('Model path:', model_path)
            if not os.path.exists(model_path):
                raise ValueError("Something went wrong... cant find trained model...")
        print('')

        # LOAD ABOVE SPECIFIED MODEL FOR TESTING
        model_to_load.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        # MODEL TESTING
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):

                # PROCESSING - ANALYSIS, METRICS, SAVING OUT DATA
                batch_size = test_batch[1].shape[0]

                # GATHER AND FORMAT BATCH DATA
                data, labels = test_batch[0], test_batch[1]
                data, labels = self.format_data_shape(data, labels)
                data, labels = self.send_data_to_device(data, labels)

                # Weird dataloader bug is causing the final training batch to be of size 0...
                if labels.shape[0] == 0:
                    continue

                # GET MODEL PREDICTIONS
                au_out, bvp_out, resp_out = self.model(data)
                au_out = torch.sigmoid(au_out)

                # GATHER AND SLICE LABELS USED FOR TEST DATASET
                TEST_AU = False
                if len(self.label_idx_test_au) > 0:
                    TEST_AU = True
                    labels_au = labels[:, self.label_idx_test_au]
                else:
                    labels_au = np.ones((batch_size, len(self.label_idx_train_au)))
                    labels_au = -1 * labels_au

                TEST_BVP = False
                if len(self.label_idx_test_bvp) > 0:
                    TEST_BVP = True
                    labels_bvp = labels[:, self.label_idx_test_bvp]
                else:
                    labels_bvp = np.ones((batch_size, len(self.label_idx_train_bvp)))
                    labels_bvp = -1 * labels_bvp

                TEST_RESP = False
                if len(self.label_idx_test_resp) > 0:
                    TEST_RESP = True
                    labels_resp = labels[:, self.label_idx_test_resp]
                else:
                    labels_resp = np.ones((batch_size, len(self.label_idx_train_resp)))
                    labels_resp = -1 * labels_resp

                # ITERATE THROUGH BATCH, SORT, AND ADD TO CORRECT DICTIONARY
                for idx in range(batch_size):

                    # if the labels are cut off due to TSM dataformating
                    if idx * self.chunk_len >= labels.shape[0] and self.using_TSM:
                        continue

                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])

                    # add subject to prediction / label arrays
                    if subj_index not in preds_dict_bvp.keys():
                        preds_dict_au[subj_index] = dict()
                        labels_dict_au[subj_index] = dict()
                        preds_dict_bvp[subj_index] = dict()
                        labels_dict_bvp[subj_index] = dict()
                        preds_dict_resp[subj_index] = dict()
                        labels_dict_resp[subj_index] = dict()

                    # append predictions and labels to subject dict
                    preds_dict_au[subj_index][sort_index] = au_out[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels_dict_au[subj_index][sort_index] = labels_au[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    preds_dict_bvp[subj_index][sort_index] = bvp_out[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels_dict_bvp[subj_index][sort_index] = labels_bvp[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    preds_dict_resp[subj_index][sort_index] = resp_out[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels_dict_resp[subj_index][sort_index] = labels_resp[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        # Calculate Eval Metrics
        bvp_metric_dict = calculate_bvp_metrics(preds_dict_bvp, labels_dict_bvp, self.config)
        resp_metric_dict = calculate_resp_metrics(preds_dict_resp, labels_dict_resp, self.config)
        au_metric_dict = calculate_bp4d_au_metrics(preds_dict_au, labels_dict_au, self.config)
