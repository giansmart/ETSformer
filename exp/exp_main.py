from models import ETSformer
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.Adam import Adam

import numpy as np
import torch
import torch.nn as nn

import os
import time
import json
from datetime import datetime

import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'ETSformer': ETSformer,
        }
        model = model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if 'warmup' in self.args.lradj:
            lr = self.args.min_lr
        else:
            lr = self.args.learning_rate

        if self.args.smoothing_learning_rate > 0:
            smoothing_lr = self.args.smoothing_learning_rate
        else:
            smoothing_lr = 100 * self.args.learning_rate

        if self.args.damping_learning_rate > 0:
            damping_lr = self.args.damping_learning_rate
        else:
            damping_lr = 100 * self.args.learning_rate

        nn_params = []
        smoothing_params = []
        damping_params = []
        for k, v in self.model.named_parameters():
            if k[-len('_smoothing_weight'):] == '_smoothing_weight':
                smoothing_params.append(v)
            elif k[-len('_damping_factor'):] == '_damping_factor':
                damping_params.append(v)
            else:
                nn_params.append(v)

        model_optim = Adam([
            {'params': nn_params, 'lr': lr, 'name': 'nn'},
            {'params': smoothing_params, 'lr': smoothing_lr, 'name': 'smoothing'},
            {'params': damping_params, 'lr': damping_lr, 'name': 'damping'},
        ])

        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, folder_name=None):
        if folder_name is None:
            folder_name = setting
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Create results folder for logs
        results_path = os.path.join('./results', folder_name)
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        # Initialize logging
        log_data = {
            'experiment_config': vars(self.args),
            'setting': setting,
            'start_time': datetime.now().isoformat(),
            'train_samples': len(train_data),
            'val_samples': len(vali_data),
            'test_samples': len(test_data),
            'epochs': []
        }

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            # Log epoch metrics
            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'val_loss': float(vali_loss),
                'test_loss': float(test_loss),
                'epoch_time': time.time() - epoch_time,
                'learning_rate': model_optim.param_groups[0]['lr']
            }
            log_data['epochs'].append(epoch_log)
            
            # Save training log after each epoch for monitoring
            log_data['last_epoch'] = epoch + 1
            log_data['current_time'] = datetime.now().isoformat()
            log_data['total_training_time_so_far'] = time.time() - time_now
            
            log_path = os.path.join(results_path, 'training_log.json')
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        # Save training log
        log_data['end_time'] = datetime.now().isoformat()
        log_data['total_training_time'] = time.time() - time_now
        log_data['early_stopped'] = early_stopping.early_stop
        log_data['best_val_loss'] = float(early_stopping.best_score) if early_stopping.best_score is not None else None
        
        log_path = os.path.join(results_path, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"Training log saved to {log_path}")

        return self.model

    def test(self, setting, data, save_vals=False, folder_name=None):
        if folder_name is None:
            folder_name = setting
        """data - 'val' or 'test' """
        test_data, test_loader = self._get_data(flag=data)

        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + folder_name, 'checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + folder_name + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        
        # Save test results to JSON
        test_results = {
            'dataset': data,
            'setting': setting,
            'test_time': datetime.now().isoformat(),
            'metrics': {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'mspe': float(mspe)
            },
            'prediction_shape': list(preds.shape),
            'true_shape': list(trues.shape)
        }
        
        result_path = os.path.join(folder_path, f'test_results_{data}.json')
        with open(result_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"Test results saved to {result_path}")

        np.save(folder_path + f'{data}_metrics.npy', np.array([mae, mse, rmse, mape, mspe]))

        if save_vals:
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

        return
