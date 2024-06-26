import csv
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset
from tqdm import tqdm
import time

t1 = time.time()


def extract_data(cluster, cluster_csv_path, data_csv_path, output_csv_path):
    cluster_df = pd.read_csv(cluster_csv_path)

    matched_rows = cluster_df[cluster_df['Cluster'] == cluster]

    pn_list = matched_rows['PN'].tolist()

    data_df = pd.read_csv(data_csv_path)

    matched_data = data_df[data_df['date'].isin(pn_list)]

    matched_data.to_csv(output_csv_path, index=False)

    return matched_data


TH_list = ['TH=20', 'TH=17', 'TH=14']
pred_length = [12, 24]

for TH_path in TH_list:

    if TH_path == 'TH=20':
        PN_file = '多元data/PN-detail hierarchy cluster result for weekly 60 para-20.csv'
    elif TH_path == 'TH=14':
        PN_file = '多元data/PN-detail hierarchy cluster result for weekly 60 para-14.csv'
    elif TH_path == 'TH=17':
        PN_file = '多元data/PN-detail hierarchy cluster result for weekly 60 para-17.csv'
    else:
        print('路径有误')
        break

    for p_l in pred_length:

        data_file = f'多元data/all_data_weekly_selected_60_windowed_T_normalized-pred{p_l}.csv'

        cluster_nums = 0
        if TH_path == 'TH=20':
            cluster_nums = 42
        elif TH_path == 'TH=17':
            cluster_nums = 80
        elif TH_path == 'TH=14':
            cluster_nums = 139

        for i_cluster in range(1, cluster_nums):
            output_csv_path = f'多元data/{TH_path}/pred{p_l}/多对一/matched_data{i_cluster}.csv'

            cluster = i_cluster
            matched_data = extract_data(cluster, PN_file, data_file, output_csv_path)
            matched_data_i = pd.read_csv(output_csv_path)
            feature_size_i = len(matched_data_i)


            class Config():
                timestep = 40
                batch_size = 8
                feature_size = feature_size_i
                hidden_size = 64
                output_size = p_l + 1
                num_layers = 2
                epochs = 400
                best_loss = 0
                learning_rate = 0.003
                model_name = 'gru'
                base_dir = 'GRU_PTH/'
                save_path = os.path.join(base_dir, '{}.pth'.format(model_name))


            if p_l == 12:
                train, test = matched_data.loc[:, '2021-week1':'2022-week52'], matched_data.loc[:,
                                                                               '2023-week1':'2023-week12']
            elif p_l == 24:
                train, test = matched_data.loc[:, '2021-week1':'2022-week52'], matched_data.loc[:,
                                                                               '2023-week1':'2023-week24']

            predict_days = test.shape[1]

            all_days = train.shape[1] + test.shape[1]

            predict = test.copy()

            config = Config()

            rmse = []
            mse = []
            mae = []
            mape_array = []
            smape_array = []

            matched_data.set_index('date', inplace=True)
            device = matched_data.index

            count = 0
            for Partnumber in device:

                count = count + 1

                print('第%d个零件，零件号为：%s' % (count, Partnumber))

                ts = matched_data

                df = ts


                def split_data(data, timestep, feature_size, count):
                    dataX = []
                    dataY = []

                    data_df = pd.read_csv(output_csv_path)

                    data_df.set_index('date', inplace=True)
                    data_df = data_df.loc[:, '2021-week1':'2022-week52'].copy()

                    windows_data_len = data_df.shape[1] - timestep

                    df_transposed = data_df.T

                    for index in range(windows_data_len - predict_days + 1):
                        X_data = df_transposed.iloc[index:index + timestep].values

                        y_data = df_transposed.iloc[index + timestep:index + timestep + predict_days, count - 1].values

                        dataX.append(X_data)
                        dataY.append(y_data)

                    dataX = np.array(dataX)
                    dataY = np.array(dataY)

                    train_size = windows_data_len - predict_days

                    x_train = dataX[: train_size, :].reshape(-1, timestep, feature_size)

                    y_train = dataY[: train_size].reshape(-1, predict_days, 1)

                    x_test = dataX[train_size:, :].reshape(-1, timestep, feature_size)

                    y_test = dataY[train_size:].reshape(-1, predict_days, 1)

                    x_val = df_transposed.iloc[data_df.shape[1] - timestep:data_df.shape[1]].values.reshape(
                        (-1, timestep, feature_size))

                    return [x_train, y_train, x_test, y_test, x_val]


                x_train, y_train, x_test, y_test, x_val = split_data(df, config.timestep, config.feature_size, count)

                x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
                y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
                x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
                y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

                x_val_tensor = torch.from_numpy(x_val).to(torch.float32)

                train_data = TensorDataset(x_train_tensor, y_train_tensor)

                test_data = TensorDataset(x_test_tensor, y_test_tensor)

                train_loader = torch.utils.data.DataLoader(train_data,
                                                           config.batch_size,
                                                           False)

                test_loader = torch.utils.data.DataLoader(test_data,
                                                          config.batch_size,
                                                          False)


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


                model = GRU(config.feature_size, config.hidden_size, config.num_layers, config.output_size)
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
                        y_train_pred = model(x_train)

                        loss = loss_function(y_train_pred.reshape(-1, Config.output_size),
                                             y_train.reshape(-1, Config.output_size))
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
                            test_loss += loss_function(y_test_pred,
                                                       y_test.reshape(-1, Config.output_size)).item()

                    avg_test_loss = test_loss / len(test_loader)

                    epoch_losses.append((epoch, avg_train_loss, avg_test_loss))

                    if config.best_loss == 0:
                        config.best_loss = avg_test_loss
                        torch.save(model.state_dict(), config.save_path)
                    elif avg_test_loss < config.best_loss:
                        config.best_loss = avg_test_loss
                        torch.save(model.state_dict(), config.save_path)

                print('Finished Training')

                with open(f'Results/{TH_path}/pred{p_l}/多对一/loss_records{i_cluster}_{count}.csv', 'w',
                          newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Epoch', 'Average Training Loss', 'Average Testing Loss'])
                    writer.writerows(epoch_losses)

                predictions = []

                current_input = x_val_tensor[0].unsqueeze(0).cuda()
                y_test_pred = model(current_input)
                predictions.append(y_test_pred.tolist())

                predictions = np.array(predictions)
                print("预测结果如下：")
                print(predictions)

                predict.iloc[count - 1] = predictions

            multivariant_One_result = pd.concat([train, predict], axis=1)
            multivariant_One_result.set_index(matched_data.index, inplace=True)
            multivariant_One_result.to_csv(
                path_or_buf=f'Results/{TH_path}/pred{p_l}/多对一/multivariant_one_result' + str(
                    i_cluster) + '.csv', header=True)

        t2 = time.time()

        print(f'Time = {(t2 - t1) / 60} minutes')
