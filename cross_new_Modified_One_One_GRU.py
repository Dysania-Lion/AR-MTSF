import csv
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import TensorDataset
from tqdm import tqdm
from pandas import read_csv

t1 = time.time()
def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


class Config():
    timestep = 40  
    batch_size = 8  
    feature_size = 1  
    hidden_size = 64  
    output_size = 1  
    num_layers = 2  
    epochs = 400  
    best_loss = 0  
    learning_rate = 0.003  
    model_name = 'gru'  
    
    base_dir = 'GRU_PTH'
    save_path = os.path.join(base_dir, '{}.pth'.format(model_name))

pred_length = [12, 24]

for p_l in pred_length:
    dataset = read_csv(
        './多元data/all_data_weekly.csv',
        header=0)
    df1 = dataset
    if p_l == 24:
        train, test = df1.loc[:, '2021-week1':'2022-week52'], df1.loc[:, '2023-week1':'2023-week25']
    elif p_l == 12:
        train, test = df1.loc[:, '2021-week1':'2022-week52'], df1.loc[:, '2023-week1':'2023-week13']
    
    predict_days = test.shape[1]
    device = df1.index   
    predict = test.copy()
    config = Config()
    
    rmse = []
    mse = []
    mae = []
    mape_array = []
    smape_array = []
    df1 = df1.set_index('date')
    
    count = 0
    for Partnumber in device:  
        count = count + 1
        print(f'Partnumber count: {count}')
        
        ts = df1.iloc[[Partnumber]]        
        df = ts        
        def split_data(data, timestep, feature_size):
            dataX = []  
            dataY = []        
            
            data = data.loc[:, '2021-week1':'2022-week52'].copy()
            data = data.values.reshape(data.shape[1], 1)
            windows_data_len = len(data) - timestep

            for index in range(windows_data_len):
                dataX.append(data[index: index + timestep][:, 0])  
                dataY.append(data[index + timestep][0])

            dataX = np.array(dataX)
            dataY = np.array(dataY)
            
            train_size = windows_data_len - predict_days

            x_train = dataX[: train_size, :].reshape(-1, timestep, feature_size)
            y_train = dataY[: train_size].reshape(-1, 1)
            x_test = dataX[train_size:, :].reshape(-1, timestep, feature_size)
            y_test = dataY[train_size:].reshape(-1, 1)
            return [x_train, y_train, x_test, y_test]

        x_train, y_train, x_test, y_test = split_data(df, config.timestep, config.feature_size)

        
        x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
        y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
        x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
        y_test_tensor = torch.from_numpy(y_test).to(torch.float32)
        
        train_data = TensorDataset(x_train_tensor, y_train_tensor)  
        test_data = TensorDataset(x_test_tensor, y_test_tensor)  

        class GRU(nn.Module):
            def __init__(self, feature_size, hidden_size, num_layers, output_size):
                super(GRU, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.gru = nn.GRU(feature_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x, hidden=None):
                batch_size = x.shape[0]

                if hidden is None:
                    h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
                else:
                    h_0 = hidden

                output, h_0 = self.gru(x, h_0)
                batch_size, timestep, hidden_size = output.shape
                output = output.reshape(-1, hidden_size)
                output = self.fc(output)  
                output = output.reshape(timestep, batch_size, -1)

                return output[-1]

        param_grid = {
            'hidden_size': [32, 64, 128],  
            'num_layers': [1, 2, 3],  
            'learning_rate': [1e-4, 1e-3, 1e-2]  
        }

        tscv = TimeSeriesSplit(n_splits=5)
        
        best_loss = float('inf')
        best_params = None

        for hidden_size in param_grid['hidden_size']:
            for num_layers in param_grid['num_layers']:
                for lr in param_grid['learning_rate']:
                    
                    model = GRU(config.feature_size, hidden_size, num_layers, config.output_size)
                    model.cuda()
                    loss_function = nn.MSELoss()
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)                    
                    fold_losses = []

                    for train_idx, test_idx in tscv.split(train_data):
                        subset_train_data = [tensors[train_idx] for tensors in train_data.tensors]
                        train_data_with_tensors = TensorDataset(subset_train_data[0], subset_train_data[1])

                        train_loader = torch.utils.data.DataLoader(train_data_with_tensors,
                                                                   batch_size=config.batch_size, shuffle=False)

                        subset_test_data = [tensors[test_idx] for tensors in train_data.tensors]
                        train_data_with_tensors = TensorDataset(subset_test_data[0], subset_test_data[1])

                        test_loader = torch.utils.data.DataLoader(train_data_with_tensors, batch_size=config.batch_size,
                                                                  shuffle=False)

                        
                        for epoch in range(400):
                            model.train()
                            running_loss = 0
                            train_bar = tqdm(train_loader)
                            for data_train in train_bar:
                                x_train_c, y_train_c = data_train
                                x_train_c, y_train_c = x_train_c.cuda(), y_train_c.cuda()
                                optimizer.zero_grad()
                                y_train_pred = model(x_train_c)
                                loss = loss_function(y_train_pred.reshape(-1, 1), y_train_c.reshape(-1, 1))
                                loss.backward()
                                optimizer.step()
                                running_loss += loss.item()
                            avg_train_loss = running_loss / len(train_loader)
                        
                        model.eval()
                        test_loss = 0
                        with torch.no_grad():
                            test_bar = tqdm(test_loader)
                            for data_test in test_bar:
                                x_test_c, y_test_c = data_test
                                x_test_c, y_test_c = x_test_c.cuda(), y_test_c.cuda()
                                y_test_pred = model(x_test_c)
                                test_loss += loss_function(y_test_pred, y_test_c.reshape(-1, 1)).item()
                        avg_test_loss = test_loss / len(test_loader)
                        fold_losses.append(avg_test_loss)
                    
                    avg_fold_loss = np.mean(fold_losses)
                    print(
                        f"Params: hidden_size={hidden_size}, num_layers={num_layers}, lr={lr} => Loss: {avg_fold_loss}")

                    
                    if avg_fold_loss < best_loss:
                        best_loss = avg_fold_loss
                        best_params = {'hidden_size': hidden_size, 'num_layers': num_layers, 'learning_rate': lr}

        print(f"Best parameters found: {best_params} with loss {best_loss}")

        
        model = GRU(config.feature_size, best_params['hidden_size'], best_params['num_layers'], config.output_size)
        model.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['learning_rate'])       
        
        loss_function = nn.MSELoss()  
                
        x_train, y_train, x_test, y_test = split_data(df, config.timestep, config.feature_size)
       
        x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
        y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
        x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
        y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

        
        train_data = TensorDataset(x_train_tensor, y_train_tensor)  
        test_data = TensorDataset(x_test_tensor, y_test_tensor)          
        train_loader = torch.utils.data.DataLoader(train_data,
                                                   config.batch_size,
                                                   False)

        test_loader = torch.utils.data.DataLoader(test_data,
                                                  config.batch_size,
                                                  False)
        epoch_losses = []
        for epoch in range(config.epochs):
            model.train()
            running_loss = 0
            train_bar = tqdm(train_loader)
            for data_train in train_bar:
                x_train, y_train = data_train
                x_train, y_train = x_train.cuda(), y_train.cuda()

                optimizer.zero_grad()
                y_train_pred = model(x_train)
                loss = loss_function(y_train_pred.reshape(-1, 1), y_train.reshape(-1, 1))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                train_bar.desc = "train epoch[{}/{}] loss:{:.6f}".format(epoch + 1,
                                                                         config.epochs,
                                                                         loss)
            avg_train_loss = running_loss / len(train_loader)
            
            model.eval()
            test_loss = 0
            with torch.no_grad():
                test_bar = tqdm(test_loader)
                for data_test in test_bar:
                    x_test, y_test = data_test
                    x_test, y_test = x_test.cuda(), y_test.cuda()
                    y_test_pred = model(x_test)
                    test_loss += loss_function(y_test_pred, y_test.reshape(-1, 1)).item()  

            avg_test_loss = test_loss / len(test_loader)
            epoch_losses.append((epoch, avg_train_loss, avg_test_loss))

            if config.best_loss == 0:
                config.best_loss = avg_test_loss
                torch.save(model.state_dict(), config.save_path)
            elif avg_test_loss < config.best_loss:
                config.best_loss = avg_test_loss
                torch.save(model.state_dict(), config.save_path)

        print('Finished Training')

        with open(f'Results/一元/pred{p_l}/loss_records{count}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Average Training Loss', 'Average Testing Loss'])
            writer.writerows(epoch_losses)
        
        predictions = []  
        current_input = x_test_tensor[-1].unsqueeze(0).cuda()  
        
        for i in range(len(x_test_tensor)):
            y_test_pred = model(current_input)
            print(y_test_pred)
            predictions.append(y_test_pred.item())
            current_input = torch.cat([current_input[:, 1:, :], y_test_pred.unsqueeze(0)], dim=1)
            print(current_input)

        
        predictions = np.array(predictions)

        
        end_col_index = -1 * len(predictions)        
        part_row = predict.loc[Partnumber]        
        part_row.iloc[end_col_index:] = predictions        
        predict.loc[Partnumber] = part_row

        hidden_size_value = best_params['hidden_size']
        num_layers_value = best_params['num_layers']
        learning_rate_value = best_params['learning_rate']

        with open(f'./Progress-一元/GRU/{p_l}-{Partnumber}-{hidden_size_value}-{num_layers_value}-{learning_rate_value}.txt', 'w') as file:
            pass  
    

    result = pd.concat([train, predict], axis=1)
    result.to_csv(
        path_or_buf=f'Results/一元/pred{p_l}/0One_One_result.csv',
        header=True)

t2 = time.time()

print(f'Time = {(t2 - t1) / 60} minutes')
