import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import random
from torch.utils.tensorboard import SummaryWriter
import xarray as xr
from collections import deque
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

import assimilation as assi

def data_preprocess(dataset_name="2022"):
    # Load the specified ERA5 dataset
    data = xr.open_dataset('E:/Era5-Global-0.5/' + dataset_name + '.nc')

    # Extract longitude and latitude data
    lon = data['longitude'].data[::]
    lat = data['latitude'].data[::]

    # Extract wind speed (u and v components) and wave height data
    wind_u = data['u10'].data
    wind_v = data['v10'].data
    wave_height = data['swh'].data

    return wind_u[:, :, :], wind_v[:, :, :], wave_height[:, :, :], lat, lon


class DynamicDataset(Dataset):
    def __init__(self, wind_u, wind_v, wave_height, time=1):
        self.wind_u = wind_u
        self.wind_v = wind_v
        self.wave_height = wave_height
        self.time = time


        self.dataset_size = len(wind_u) - time
        print(self.dataset_size)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        indices = idx

        # Extract wind data for the current time step
        wind_u_data = self.wind_u[indices + self.time]
        wind_v_data = self.wind_v[indices + self.time]

        # Extract wave height data for the current and future time steps
        wave_height_data_thistime = self.wave_height[idx]
        wave_height_data = self.wave_height[idx + self.time]

        # Create an array to store wind and wave height data with extended edges
        extend_edge = 20
        data = np.empty((3, lat_wind, lon_wind + extend_edge * 2), dtype=np.float32)

        # Populate the data array with wind and wave height information
        data[0, :, extend_edge:-extend_edge] = wind_u_data
        data[1, :, extend_edge:-extend_edge] = wind_v_data
        data[0, :, 0:extend_edge] = wind_u_data[:, -extend_edge:]
        data[1, :, 0:extend_edge] = wind_v_data[:, -extend_edge:]
        data[0, :, -extend_edge:] = wind_u_data[:, :extend_edge]
        data[1, :, -extend_edge:] = wind_v_data[:, :extend_edge]

        wave_height_data_thistime = np.nan_to_num(wave_height_data_thistime, nan=0, copy=False)
        data[2, :, extend_edge:-extend_edge] = wave_height_data_thistime
        data[2, :, 0:extend_edge] = wave_height_data_thistime[:, -extend_edge:]
        data[2, :, -extend_edge:] = wave_height_data_thistime[:, :extend_edge]

        # Convert data and labels to PyTorch tensors
        data = torch.Tensor(data).float()
        label = torch.Tensor(np.nan_to_num(wave_height_data, nan=0, copy=False)).float()

        return data, label




class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),


        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Channelfold(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.timesfold = nn.Sequential(

             nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
             nn.SiLU(inplace=True),
             nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
             nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.timesfold(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.final_resize = nn.AdaptiveAvgPool2d((281, 720))

    def forward(self, x):
        x=self.conv(x)
        x=self.final_resize(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)



    def forward(self, x1, x2):
        x1 = self.up(x1)


        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class RollingModel(nn.Module):
    def __init__(self, n_vars, n_times, n_classes, bilinear=True):
        super(RollingModel, self).__init__()
        self.n_vars = n_vars
        self.n_times = n_times
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.cfold = Channelfold(3, 32)
        self.inc = DoubleConv(32, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.silu=nn.SiLU

    def forward(self, x):

        x = self.cfold(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = torch.squeeze(logits)

        return logits

def custom_loss(outputs, labels, lat_cos):

    lat_cos = lat_cos.reshape(1, lat,1)
    mse_loss = F.mse_loss(outputs, labels, reduction='none')

    corrected_loss = mse_loss * lat_cos
    corrected_loss_mean = corrected_loss.mean()

    return corrected_loss_mean

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state_dict keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            # Remove prefix
            new_key = k[7:]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

def NetTrain():

    learning_rate=0.0001
    epochs=100
    batch=4
    sequence_length=1

    model = RollingModel(n_vars=2, n_times=sequence_length, n_classes=1)


    writer = SummaryWriter(log_dir="./logs/" + datetime.now().strftime("%Y%m%d%H%M%S") + '_Unet' + comment)
    fake_img = torch.zeros((1, 3, lat, lon+40))
    writer.add_graph(model, fake_img)



    latest_checkpoint_path = os.path.join(model_path, "latest_checkpoint.pt")


    if os.path.exists(latest_checkpoint_path):
        print("Loading from latest checkpoint:", latest_checkpoint_path)
    else:
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        latest_epoch = -1
        for file_name in os.listdir(model_path):
            if not file_name.endswith(".pt"):
                continue
            try:
                epoch, loss = file_name.rsplit("_", 1)[0], file_name[:-3].split("_")[-1]
                epoch = int(epoch)
                loss = float(loss)
            except ValueError:
                continue
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_checkpoint_path = os.path.join(model_path, file_name)

        if latest_epoch == -1:
            print("No valid checkpoint found in", model_path)
            latest_checkpoint_path = None
        else:
            print("Loading from checkpoint:", latest_checkpoint_path)

    if latest_checkpoint_path is not None:
        checkpoint = torch.load(latest_checkpoint_path)
        print(latest_checkpoint_path)
        if 'model' in checkpoint:
            model = checkpoint['model'].to(device)
        else:

            model = model.to(device)
            print("Error: cannot find model in checkpoint!")



        model.load_state_dict( remove_module_prefix(checkpoint['model_state_dict']))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print("Loaded checkpoint {}: epoch={}, loss={}".format(latest_checkpoint_path, start_epoch, loss  ))

    else:
        start_epoch = 0
        net = model.to(device)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)



    mse = nn.MSELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)




    model.train()
    years = list(range(2000, 2018))
    random.shuffle(years)



    loaded = np.load('mask_land.npz')
    mask= loaded['mask']
    mask= np.logical_not(mask)
    mask = mask[:, :]



    extend_edge = 20

    mask_extend = np.empty(( lat_wind, lon_wind + extend_edge * 2), dtype=bool)

    mask_extend[ :, extend_edge:-extend_edge] = mask
    mask_extend[ :, 0:extend_edge] = mask[:, -extend_edge:]
    mask_extend[ :, -extend_edge:] = mask[:, :extend_edge]

    mask = torch.tensor(mask).to(device)

    mask_extend=torch.tensor(mask_extend).to(device)


    lat_start, lat_end = -70, 70
    lat_size = 281
    lat_cos_fix = np.cos(np.deg2rad(np.linspace(lat_start, lat_end, lat_size))).reshape(-1, 1)
    lat_cos_fix = torch.tensor(lat_cos_fix).to(device)


    for epoch in range(0,epochs):
        for year in years:

            running_loss = 0.0
            true_loss=0.0
            model = model.to(device)

            wind_u, wind_v, wave_height, _, _ = data_preprocess(str(year))
            dataset = DynamicDataset(wind_u, wind_v, wave_height)

            dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True)
            dataloader_length = len(dataloader)

            print(f"Epoch:{epoch},Year:{year},DataShape{dataloader_length }")


            del wind_u, wind_v, wave_height

            for data, labels in tqdm(dataloader, desc=f"Epoch {epoch }", leave=False):
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()

                data[:,2,:,:] = torch.masked_fill(data[:,2,:,:],mask_extend, 0)
                outputs = model(data)


                outputs = torch.masked_fill(outputs, mask, 0)
                labels = torch.masked_fill(labels, mask, 0)

                loss = custom_loss(outputs, labels, lat_cos_fix)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tqdm.write(f"MSE: {loss.item():.4f}", end="")
                true_loss += (torch.mean((outputs - labels) ** 2))

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch }/{epochs}, Rmse Loss: {np.sqrt(running_loss /  dataloader_length)},True Loss:{true_loss /  dataloader_length}, Current Learning Rate: {current_lr}")
            writer.add_scalar('Learning Rate:', current_lr, global_step=epoch)
            writer.add_scalar('Train Rmse Loss:', np.sqrt(running_loss /  dataloader_length), global_step=epoch)

            del dataset,dataloader

            save_dict = {}
            save_dict['epoch'] = epoch
            save_dict['model_state_dict'] = model.state_dict()
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
            save_dict['loss'] = np.sqrt(running_loss / dataloader_length)

            checkpoint_path = os.path.join(model_path, "latest_checkpoint.pt")

            torch.save(save_dict, checkpoint_path)

            print("Model saved to {}".format(checkpoint_path))

        if epoch % 1 == 0:

            print("Validation")
            wind_u, wind_v, wave_height, _, _ = data_preprocess('2022')

            test_dataset = DynamicDataset(wind_u, wind_v, wave_height)
            test_dataloader = DataLoader(test_dataset, batch_size=batch)

            del  wind_u, wind_v, wave_height

            test_loss = 0
            real_loss = 0
            test_step = 0
            rmse = np.zeros([lat, lon])



            model.eval()
            with torch.no_grad():


                for test_data, target in test_dataloader:
                    test_data, target=test_data.to(device), target.to(device)

                    test_data[:, 2, :, :] = torch.masked_fill(test_data[:, 2, :, :], mask_extend, 0)
                    logits = model(test_data)
                    logits=torch.masked_fill(logits, mask, 0)
                    target= torch.masked_fill( target, mask, 0)

                    msevalue = mse(logits, target).item()


                    realvalue = torch.mean(torch.abs(logits - target))
                    test_loss += msevalue
                    real_loss += realvalue
                    rmse += torch.sqrt(torch.mean(torch.square(logits - target), 0)).detach().cpu().numpy()

                    test_step += 1




                test_loss /= test_step
                real_loss /= test_step
                rmse  /= test_step

                print(
                    '\n  Epoch: {} Validation set: Average Mse loss: {:.6f},Average RMSE loss: {:.6f}, Abs loss: {:.6f}'.format(
                         epoch, test_loss, np.sqrt(test_loss), real_loss))
                writer.add_scalar('Validation RMSE Loss:', np.sqrt(test_loss), global_step=epoch)


                plt.figure()
                cax = plt.matshow(rmse, cmap='viridis')
                plt.colorbar(cax)

                writer.add_figure('RMSE Validation Heatmap', plt.gcf(), global_step=epoch)
                plt.close()


        if epoch%1==0 :
            save_dict = {}
            save_dict[f'epoch'] =  epoch
            save_dict[f'model_state_dict'] =model.state_dict()
            save_dict[f'optimizer_state_dict'] = optimizer.state_dict()
            save_dict[f'loss'] = np.sqrt(test_loss)

            checkpoint_path = os.path.join(model_path, "{}_{}_mid_".format(epoch, (running_loss /  dataloader_length)) + comment + ".pt")
            torch.save(save_dict, checkpoint_path)
            print("Save cp")




    save_dict = {}
    save_dict[f'epoch'] =  epoch
    save_dict[f'model_state_dict'] =model.state_dict()
    save_dict[f'optimizer_state_dict'] = optimizer.state_dict()
    save_dict[f'loss'] = np.sqrt(test_loss)

    checkpoint_path = os.path.join(model_path, "{}_{}.pt".format(epoch, (running_loss /  dataloader_length)))
    torch.save(save_dict, checkpoint_path)
    print("Finish")


def forecast_curver(forecast_steps, RMSE, CC):

    time = np.arange(1, forecast_steps + 1, int(forecast_steps/len(RMSE)))

    plt.figure(figsize=(12, 6), dpi=100)
    plt.rcParams.update({'font.size': 16})

    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=100)

    ax1.plot(time, RMSE, label='RMSE', color='#4D85BD', linestyle='-', linewidth=2)
    ax1.set_xlabel('Forecast Hours', fontsize=18, fontname='Arial')
    ax1.set_ylabel('RMSE ', fontsize=18, fontname='Arial', color='#4D85BD')
    ax1.tick_params(axis='y', labelcolor='#4D85BD')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax2 = ax1.twinx()
    ax2.plot(time, CC, label='CC', color='#F7903D', linestyle='-', linewidth=2)
    ax2.set_ylabel('CC', fontsize=18, fontname='Arial', color='#F7903D')
    ax2.tick_params(axis='y', labelcolor='#F7903D')


    plt.title('Forecast Error Curve ', fontsize=20, fontweight='bold',
              fontname='Arial')

    plt.savefig('Forecast_Error_Curve.jpg', bbox_inches='tight')
    plt.show()


def continuous_inference(models, test_dataloader,forecast_steps=0,Enable_Assi=True):

    cci_data = np.load(".\CCI2020.npz")
    cci_swhlist = cci_data["swh"]
    cci_lonlist = cci_data["lonlist"]
    cci_latlist = cci_data["latlist"]
    cci_timelist = cci_data["timelist"]



    model_predict = deque(maxlen=48)
    model_label = deque(maxlen=48)

    if forecast_steps <= 0:
        total_time = int(len(test_dataloader))
    else:
        total_time=forecast_steps


    running_data = np.zeros([2, total_time, lat, lon],dtype=np.float32)


    RMSE=[]
    RRMSE=[]
    CC=[]


    count = 0
    hour=0


    loaded = np.load('mask_land.npz')
    mask = loaded['mask']
    mask = np.logical_not(mask)
    mask = mask[:, :]

    mask = torch.tensor(mask).to(device)
    extend_edge = 20
    out_expend = torch.empty((1, lat_wind, lon_wind + extend_edge * 2)).to(device)

    start_time=0
    with torch.no_grad():
        for test_data, target in test_dataloader:
            batch_predictions = torch.zeros([lat,lon]).to(device)
            test_data, target = test_data.to(device), target.to(device)



            if hour ==  start_time:
                for index, model in enumerate(models):
                    out = model(test_data)
                    batch_predictions=batch_predictions+out

            elif hour >start_time:
                out_expend[:, :, extend_edge:-extend_edge] = out
                out_expend[:, :, 0:extend_edge] = out[:,:, -extend_edge:]
                out_expend[:, :, -extend_edge:] = out[:,:, :extend_edge]
                out =out_expend
                test_data[:,2,:,:]=out
                for index, model in enumerate(models):
                     batch_predictions = batch_predictions + model(test_data)


            out=torch.unsqueeze(batch_predictions/len(models),0)

            out = torch.masked_fill(out, mask, 0)
            target = torch.masked_fill(target, mask, 0)

            out[out<0] = 0




            if hour % 1 == 0:
                x = out.detach().cpu().numpy().flatten()
                y = target.detach().cpu().numpy().flatten()
                x_filtered = [x[i] for i in range(len(y)) if y[i] != 0]
                y_filtered = [val for val in y if val != 0]

                x = np.array(x_filtered)
                y = np.array(y_filtered)
                loss = np.mean(np.square(x - y))

                running_data[0, count, :, :] = np.reshape(out.detach().cpu().numpy(), [lat, lon])  # predict:0
                running_data[1, count, :, :] = np.reshape(target.detach().cpu().numpy(), [lat, lon])  # label:1


                RMSE.append(np.sqrt(loss))
                RRMSE.append(np.sqrt(loss / np.mean(y)))
                CC.append(np.corrcoef(x, y)[0, 1])

                print(hour,RMSE[count])
                count+=1


            model_predict.append(out.squeeze().cpu().numpy())
            model_label.append(target.squeeze().cpu().numpy())


            if (hour>20 and hour % 6==0 and Enable_Assi):

                Fcref = assi.oa_assimilation_multi_thread(cci_swhlist, cci_latlist, cci_lonlist, cci_timelist, np.array(model_predict),
                                                          hour + 1, window_size=30)
                print(assi.rmse(model_predict[-1], model_label[-1]), '^', assi.rmse(Fcref, model_label[-1]))

                out = torch.unsqueeze(torch.tensor(Fcref), 0).to(device)

            hour += 1
            if hour == total_time: break;

    print(hour, count)

    forecast_curver(hour, RMSE, CC)

    del  RMSE, RRMSE, CC,model_predict,model_label


    return  running_data




def NetInference():

    #########Epoch Ensemble################

    model_names = ['rolling_model_cp1.pt',
                   'rolling_model_cp2.pt',
                   'rolling_model_cp3.pt',
                   'rolling_model_cp4.pt',
                   'rolling_model_cp5.pt',
                   ]

    models_list = []


    for model_name in model_names:
        model = RollingModel(n_vars=2, n_times=1, n_classes=1)
        model = model.to(device)
        checkpoint= torch.load(model_path +'/'+ model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        models_list.append(model)

#######################################################


    for year in range(2020,2021):
        print(year)

        wind_u, wind_v, wave_height, axis_lat, axis_lon = data_preprocess(str(year))
        dataset = DynamicDataset(wind_u, wind_v, wave_height)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)



        model_data = continuous_inference(models_list, dataloader,forecast_steps=2000,Enable_Assi=True)

        print("save")
        np.savez( 'model_data.npz',
                 model_data=model_data,
                 axis_lat=axis_lat,
                 axis_lon=axis_lon)




os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

in_channels = 2
lat_wind =281
lon_wind =720
lat = 281
lon =720

model_path=r"./model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

comment= "rolling_model"

#NetTrain()
NetInference()
