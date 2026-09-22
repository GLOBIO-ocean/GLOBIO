import torch
import numpy as np

from canyonbpy import canyonb

import datetime as dt

from discretization import dict_max_pressure, dict_interval
from utils import decode_day, decode_space, get_date, decode_year


index_depths_SOCA = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 125, 150, 175, 199]


def compute_error_metadata(
    dataloader,
    model,
    model_mlp_day,
    model_mlp_year,
    model_mlp_space,
    device
):
    model.eval()
    model_mlp_day.eval()
    model_mlp_year.eval()
    model_mlp_space.eval()

    records = []

    with torch.no_grad():

        for idx, data_dict in enumerate(dataloader):

            training_year = data_dict["year"].to(device)
            training_day = data_dict["day"].to(device)
            training_space = data_dict["space"].to(device)

            training_temp = data_dict["temp"].to(device)
            training_psal = data_dict["psal"].to(device)
            training_doxy = data_dict["doxy"].to(device)

            targets = data_dict["target_variable"].to(device)

            output_day = model_mlp_day(training_day)

            training_year = training_year.unsqueeze(1)
            output_year = model_mlp_year(training_year.float())

            output_space = model_mlp_space(training_space.float())

            output_day = torch.transpose(output_day.unsqueeze(0), 0, 1)
            output_year = torch.transpose(output_year.unsqueeze(0), 0, 1)
            output_space = torch.transpose(output_space.unsqueeze(0), 0, 1)

            training_temp = torch.transpose(training_temp.unsqueeze(0), 0, 1)
            training_psal = torch.transpose(training_psal.unsqueeze(0), 0, 1)
            training_doxy = torch.transpose(training_doxy.unsqueeze(0), 0, 1)

            source = torch.cat(
                (
                    output_day,
                    output_year,
                    output_space,
                    training_temp,
                    training_psal,
                    training_doxy
                ),
                dim=1
            )

            targets = torch.transpose(targets.unsqueeze(0), 0, 1)

            output = model(source)
            output = torch.unsqueeze(output, 1)

            mse = ((output - targets) ** 2).mean()
            rmse = torch.sqrt(mse).item()

            # coordinate geografiche
            lat, lon = decode_space(
                training_space[0][0],
                training_space[0][1],
                training_space[0][2]
            )

            day_of_year = decode_day(
                training_day[0][0],
                training_day[0][1]
            )

            records.append({
                "rmse": rmse,
                "lat": lat.item(),
                "lon": lon.item(),
                "day": day_of_year.item()
            })

    return records


def compute_sample_errors(
        dataloader, 
        model, 
        model_mlp_day, 
        model_mlp_year, 
        model_mlp_space, 
        device, 
        baseline_name = None,
        max_samples = np.inf
    ):

    model.eval()
    model_mlp_day.eval()
    model_mlp_year.eval()
    model_mlp_space.eval()

    errors = []

    with torch.no_grad():

        for idx, data_dict in enumerate(dataloader):

            if idx > max_samples:
                break

            training_year = data_dict["year"].to(device)
            training_day = data_dict["day"].to(device)
            training_space = data_dict["space"].to(device)
            training_temp = data_dict["temp"].to(device)
            training_psal = data_dict["psal"].to(device)
            training_doxy = data_dict["doxy"].to(device)
            targets = data_dict["target_variable"].to(device)

            output_day = model_mlp_day(training_day)
            training_year = training_year.unsqueeze(1)
            output_year = model_mlp_year(training_year.float())
            output_space = model_mlp_space(training_space.float())

            output_day = torch.transpose(output_day.unsqueeze(0), 0, 1)
            output_year = torch.transpose(output_year.unsqueeze(0), 0, 1)
            output_space = torch.transpose(output_space.unsqueeze(0), 0, 1)

            training_temp = torch.transpose(training_temp.unsqueeze(0), 0, 1)
            training_psal = torch.transpose(training_psal.unsqueeze(0), 0, 1)
            training_doxy = torch.transpose(training_doxy.unsqueeze(0), 0, 1)

            source = torch.cat(
                (output_day, output_year, output_space,
                 training_temp, training_psal, training_doxy),
                dim=1
            )

            targets = torch.transpose(targets.unsqueeze(0), 0, 1)

            output = model(source)
            output = torch.unsqueeze(output, 1)

            if baseline_name == "soca":
                output = output[:,:, index_depths_SOCA]
                targets  = targets[:,:, index_depths_SOCA]

            # RMSE per singolo profilo
            mse = ((output - targets) ** 2).mean()
            rmse = torch.sqrt(mse)

            errors.append(rmse.item())

    return np.array(errors)

def get_suazade_profile(date, lat, lon, temp, psal, doxy, measured_var):

    n = len(measured_var.squeeze())
    output_suazade = np.zeros(n)
    pres = np.arange(0, dict_max_pressure['NITRATE'], dict_interval['NITRATE'])

    for index in range(n):
        single_data = {
            'gtime': dt.datetime.strptime(date, "%Y-%m-%d"), 
            'lat': lat, 
            'lon': lon, 
            'pres': pres[index], 
            'temp': temp.squeeze()[index], 
            'psal': psal.squeeze()[index], 
            'doxy': doxy.squeeze()[index]
        }
        output_suazade[index] = canyonb(**single_data)['NO3']
    return output_suazade

def compute_canyonb_errors(dataloader, denorm_function, max_samples = np.inf):

    errors = []

    for idx, sample_dict in enumerate(dataloader):

        if idx > max_samples:
            break

        year = sample_dict["year"]
        day = sample_dict["day"]
        space = sample_dict["space"]
        temp = denorm_function(sample_dict["temp"].numpy(), "temp")
        psal = denorm_function(sample_dict["psal"].numpy(), "psal")
        doxy = denorm_function(sample_dict["doxy"].numpy(), "doxy")
        target = sample_dict["target_variable"].numpy()

        lat, lon = decode_space(space[0][0], space[0][1], space[0][2])
        lat = lat.item()
        lon = lon.item()
        date = get_date(day, year)

        pred = get_suazade_profile(date, lat, lon, temp, psal, doxy, target)

        pred = np.array(pred)

        rmse = np.sqrt(np.mean((pred - target) ** 2))
        errors.append(rmse)

    return np.array(errors)


def compute_soca_errors(dataloader, soca_ds, soca_indexes):

    errors = []
    valid_indexes = []

    for idx, sample_dict in enumerate(dataloader):


        # Original index in the unfiltered SOCA dataset
        original_idx = soca_indexes[idx]

        target = (
            sample_dict["target_variable"]
            .squeeze()
            .detach()
            .cpu()
            .numpy()[index_depths_SOCA]
        )

        year = decode_year(sample_dict["year"])

        lat, lon = decode_space(
            sample_dict["space"][0][0],
            sample_dict["space"][0][1],
            sample_dict["space"][0][2]
        )

        lat = lat.item()
        lon = lon.item()

        pred = soca_ds.iloc[-1, original_idx + 1]

        if isinstance(pred, str):
            pred = pred.strip("[]").split()
            pred = np.array([float(x) for x in pred], dtype=float)

        valid_chla = np.all(pred < 100.0) and not np.all(pred == 0.0)

        if year < 2024 and valid_chla:
            valid_indexes.append(idx)

        rmse = np.sqrt(np.mean((pred - target) ** 2))
        errors.append(rmse)

    return np.array(errors), valid_indexes
