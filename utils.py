import os
import random

import numpy as np
import pandas as pd
import torch
import json
import math
from torch.utils.data import DataLoader
import datetime as dt

from architecture.conv1med_dp import get_conv_model
from architecture.mlp import MLPDay, MLPYear, MLPSpace


def create_directories(list_structure, region, is_ft, split_type, my_date):
    save_dir = os.getcwd()
    for folder in list_structure:
        save_dir = save_dir + "/" + str(folder) 
        os.makedirs(save_dir, exist_ok=True)
    folders_name = os.listdir(save_dir)
    if is_ft:
        training_type = "_ft_" + region + '_' + split_type
    elif region:
        training_type = "_local_" + region + '_' + split_type
    else:
        training_type = "global" + '_' + split_type
    date_save_dir = save_dir + "/" + str(my_date) + training_type
    if not os.path.exists(date_save_dir):
        os.makedirs(date_save_dir, exist_ok=True)
    else:
        m = sum((str(my_date) + training_type) in item for item in folders_name) 
        date_save_dir = save_dir + "/" + str(my_date) + training_type + "_" + str(m)
        os.makedirs(date_save_dir, exist_ok=True)
    return date_save_dir


def shuffle_dict(my_dict):
    items = list(my_dict.items())  
    random.shuffle(items)
    return dict(items)


def from_day_rad_to_day(day_rad):
    day = (day_rad * 365) / (2 * np.pi)
    return day

def decode_year(year_offset):
        """
        Reverse:
            year_offset = year - 2010
        """
        return int((year_offset + 2010).item())

def decode_day(day_sin, day_cos):
        """
        Recover day angle from sin/cos encoding.

        atan2 returns angle in [-pi, pi]
        so we wrap it into [0, 2*pi].
        """
        angle = torch.atan2(day_sin, day_cos)

        angle = torch.where(
            angle < 0,
            angle + 2 * math.pi,
            angle
        )

        # convert to day-of-year
        day_of_year = angle / (2 * math.pi) * 365.0

        return day_of_year

def decode_space(x, y, z):
        """
        Reverse spherical Cartesian encoding.

        Forward:
            x = cos(lat) cos(lon)
            y = cos(lat) sin(lon)
            z = sin(lat)
        """

        lat_rad = torch.asin(z)
        lon_rad = torch.atan2(y, x)

        lat = torch.rad2deg(lat_rad)
        lon = torch.rad2deg(lon_rad)
        
        return lat, lon


def get_date(day_enc, year_offset):

    year = int((year_offset + 2010).item())

    day_rad = torch.atan2(day_enc[0][0], day_enc[0][1])
    day_rad = torch.where(day_rad < 0, day_rad + 2 * math.pi, day_rad).item()
    day_total = round(day_rad * 365 / (2 * np.pi))

    pseudo_month = day_total // 30
    pseudo_day = day_total % 30

    if pseudo_day == 0:
        pseudo_day = 30
        pseudo_month -= 1

    pseudo_month = max(0, min(pseudo_month, 11))
    date = dt.datetime(year, pseudo_month + 1, 1) + dt.timedelta(days=pseudo_day - 1)

    return date.strftime("%Y-%m-%d")


def make_ds(variable, input_float_directory, flag_complete=1, flag_toy=0):
    from make_ds.make_superfloat_ds import make_pandas_df, make_pandas_toy_df
    if not os.path.exists(os.getcwd() + f"/ds/{variable}/"):
        os.mkdir(os.getcwd() + f"/ds/{variable}/")
    if flag_complete and not os.path.exists(os.getcwd() + f"/ds/{variable}/float_ds_sf_train_full.csv"):
        print("hello making complete ds...")
        make_pandas_df(input_float_directory, variable=variable)
        print("superfloat complete ds created")
    if flag_toy and not os.path.exists(os.getcwd() + f"/ds/{variable}/toy_ds_sf.csv"):
        print("hello making toy ds...")
        make_pandas_toy_df(input_float_directory, variable=variable)
        print("superfloat toy ds created")
    return


def save_ds_info(training_folder, flag_toy, batch_size, epochs, lr, dp_rate, lambda_l2_reg, save_dir, alpha_smooth_reg, alpha_shape, loss_type,  weighted_loss, cnn, optim):
    dict_info = {'train_ds': [training_folder],
                 'is_toy': [flag_toy],
                 'batch_size': [batch_size],
                 'epoch': [epochs],
                 'lr': [lr],
                 'dp_rate': [dp_rate],
                 'lambda_l2_reg': [lambda_l2_reg],
                 'alpha_smooth_reg': alpha_smooth_reg, 
                 'alpha_shape': alpha_shape, 
                 'loss_type' : loss_type, 
                 'weighted_loss': weighted_loss, 
                 'cnn' : cnn, 
                 'optim' : optim}
    pd_ds = pd.DataFrame(dict_info)
    pd_ds.to_csv(save_dir + '/info.csv')
    return


def save_ds_info_ddp(variable, flag_toy, batch_size, epochs, lr, dp_rate, save_dir, alpha_smooth_reg):
    dict_info = {'variable': [variable], 
                 'is_toy': [flag_toy],
                 'batch_size': [batch_size],
                 'epoch': [epochs],
                 'lr': [lr],
                 'dp_rate': [dp_rate],
                 'alpha_smooth_reg': alpha_smooth_reg}
    pd_ds = pd.DataFrame(dict_info)
    pd_ds.to_csv(save_dir + '/info.csv')
    return


def get_output(sample_dict, model_day, model_year, model_space, model):
    year = sample_dict["year"]
    day = sample_dict["day"]
    space = sample_dict["space"]
    temp = sample_dict["temp"]
    psal = sample_dict["psal"]
    doxy = sample_dict["doxy"]

    output_day = model_day(day)
    output_year = model_year(year.unsqueeze(1))
    output_space = model_space(space)
    output_day = torch.transpose(output_day.unsqueeze(0), 0, 1)
    output_year = torch.transpose(output_year.unsqueeze(0), 0, 1)
    output_space = torch.transpose(output_space.unsqueeze(0), 0, 1)
    temp = torch.transpose(temp.unsqueeze(0), 0, 1)
    psal = torch.transpose(psal.unsqueeze(0), 0, 1)
    doxy = torch.transpose(doxy.unsqueeze(0), 0, 1)
    x = torch.cat((output_day, output_year, output_space, temp, psal, doxy), 1)
    output_model = model(x.float())
    return output_model


def compute_and_save_normalization_stats(loader, save_path):
    all_temps, all_psals, all_doxys = [], [], []
    for sample_dict in loader:
        temp = sample_dict["temp"]
        psal = sample_dict["psal"]
        doxy = sample_dict["doxy"]
        all_temps.append(temp.flatten())
        all_psals.append(psal.flatten())
        all_doxys.append(doxy.flatten())
    def stack_and_stats(tensors):
        cat = torch.cat(tensors)
        return float(cat.mean()), float(cat.std())
    stats = {
        "temp": stack_and_stats(all_temps),
        "psal": stack_and_stats(all_psals),
        "doxy": stack_and_stats(all_doxys)
    }
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"[INFO] Saved normalization stats to {save_path}", flush=True)
    return stats



def compute_denormalization_stats(list_data, variable, save_path):
    normalized_data = list_data
    with open(save_path, "r") as f:
        stats = json.load(f)
    var_mean, var_std = float(stats[variable][0]), float(stats[variable][1])
    original_data = [n_var * var_std + var_mean for n_var in normalized_data]   
    return original_data

def get_season(day_of_year, lat):
    """
    Restituisce:
    winter, spring, summer, autumn
    """

    if day_of_year < 80:
        season_north = "winter"
    elif day_of_year < 172:
        season_north = "spring"
    elif day_of_year < 264:
        season_north = "summer"
    elif day_of_year < 355:
        season_north = "autumn"
    else:
        season_north = "winter"

    if lat >= 0:
        return season_north

    inversion = {
        "winter": "summer",
        "spring": "autumn",
        "summer": "winter",
        "autumn": "spring"
    }
    return inversion[season_north]


def load_models(variable, checkpoint_path, device):
    model = get_conv_model(variable=variable, dp_rate=0.2).to(device)
    model_mlp_day = MLPDay().to(device)
    model_mlp_year = MLPYear().to(device)
    model_mlp_space = MLPSpace().to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt["model"])
    model_mlp_day.load_state_dict(ckpt["mlp_day"])
    model_mlp_year.load_state_dict(ckpt["mlp_year"])
    model_mlp_space.load_state_dict(ckpt["mlp_space"])

    model.eval()
    model_mlp_day.eval()
    model_mlp_year.eval()
    model_mlp_space.eval()

    return {
        "model": model,
        "day": model_mlp_day,
        "year": model_mlp_year,
        "space": model_mlp_space
    }

def prepare_dataloader(dataset, batch_size=1):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)