import csv
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from tqdm import tqdm
from pandas import read_csv

t1 = time.time()


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
    model_name = 'lstm'

    base_dir = 'LSTM_PTH/'
    save_path = os.path.join(base_dir, '{}.pth'.format(model_name))


pred_length = [12, 24]

for p_l in pred_length:

    dataset = read_csv(
        'all_data_weekly.csv',
        header=0)
    df1 = dataset

    if p_l == 24:
        train, test = df1.loc[:, '2021-week1':'2022-week52'], df1.loc[:, '2023-week1':'2023-week24']
    elif p_l == 12:
        train, test = df1.loc[:, '2021-week1':'2022-week52'], df1.loc[:, '2023-week1':'2023-week12']

    predict_days = test.shape[1]

    device = df1.index
    print(device)

    predict = test.copy()
    print(predict)

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
        print(count)

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
            print(dataX)
            print(dataX.shape)

            train_size = windows_data_len - predict_days

            print(train_size)

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

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   config.batch_size,
                                                   False)

        test_loader = torch.utils.data.DataLoader(test_data,
                                                  config.batch_size,
                                                  False)


        class LSTM(nn.Module):
            def __init__(self, feature_size, hidden_size, num_layers, output_size):
                super(LSTM, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers

                self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x, hidden=None):
                batch_size = x.shape[0]

                if hidden is None:
                    h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
                    c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
                else:
                    h_0, c_0 = hidden

                output, (h_0, c_0) = self.lstm(x, (h_0, c_0))

                batch_size, timestep, hidden_size = output.shape

                output = output.reshape(-1, hidden_size)

                output = self.fc(output)

                output = output.reshape(timestep, batch_size, -1)

                return output[-1], (h_0, c_0)


        model = LSTM(config.feature_size, config.hidden_size, config.num_layers, config.output_size)
        model.cuda()
        loss_function = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        epoch_losses = []
        for epoch in range(config.epochs):
            model.train()
            running_loss = 0
            train_bar = tqdm(train_loader)
            for data_train in train_bar:
                x_train, y_train = data_train
                x_train, y_train = x_train.cuda(), y_train.cuda()

                optimizer.zero_grad()
                y_train_pred, _ = model(x_train)
                loss = loss_function(y_train_pred, y_train.reshape(-1, 1))
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
                    y_test_pred, _ = model(x_test)
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
        print(current_input)
        for i in range(len(x_test_tensor)):
            y_test_pred, _ = model(current_input)
            print(y_test_pred)

            predictions.append(y_test_pred.item())

            current_input = torch.cat([current_input[:, 1:, :], y_test_pred.unsqueeze(0)], dim=1)
            print(current_input)

        predictions = np.array(predictions)
        print(predictions)

        end_col_index = -1 * len(predictions)

        part_row = predict.loc[Partnumber]

        part_row.iloc[end_col_index:] = predictions

        predict.loc[Partnumber] = part_row

    result = pd.concat([train, predict], axis=1)

    result.to_csv(
        path_or_buf=f'Results/一元/pred{p_l}/0One_One_result.csv',
        header=True)

    print("预测结果已保存")

t2 = time.time()

print(f'Time = {(t2 - t1) / 60} minutes')
