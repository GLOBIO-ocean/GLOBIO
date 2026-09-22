import os
import torch

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch.nn.functional as F
import pandas as pd

from discretization import dict_max_pressure
from utils import decode_year, decode_day, decode_space, get_season, get_date
from evaluation.metrics import index_depths_SOCA, get_suazade_profile


cartopy.config["data_dir"] = '/leonardo/home/userexternal/lbonin00/cartopy_data'


def torch_moving_average(x, window_size):
    kernel = torch.ones(1, 1, window_size, device=x.device) / window_size
    padding = window_size // 2
    input_padded = F.pad(x, (padding, padding), mode='replicate')
    output_ma = F.conv1d(input_padded, kernel, padding=0)
    return output_ma

def get_soca_profile(soca_ds, idx):
    pred = soca_ds.iloc[-1, idx + 1]
    if isinstance(pred, str):
        pred = pred.strip("[]").split()
        pred = np.array([float(x) for x in pred], dtype=float)
    return pred

def get_canyonb_profile():
    pass

def plot_nitrate_profiles(
        ds,
        save_dir,
        models_dict,
        denorm_function,
        mov_average=False,
        compute_baseline = False
    ):
    print(save_dir)

    dir_profile = os.path.join(save_dir, "profile")
    os.makedirs(dir_profile, exist_ok=True)

    max_pres = dict_max_pressure["NITRATE"]

    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="sans-serif",
        font_scale=4.5,
        color_codes=True,
        rc=None
    )

    plot_counter = 0

    for sample_dict in ds:

        if plot_counter > 50:
            break

        plot_counter += 1

        year = sample_dict["year"]
        day = sample_dict["day"]
        space = sample_dict["space"]
        temp = sample_dict["temp"]
        psal = sample_dict["psal"]
        doxy = sample_dict["doxy"]
        output_variable = sample_dict["target_variable"]
        device_name = sample_dict["device_name"]

        temp = torch.transpose(temp.unsqueeze(0), 0, 1)
        psal = torch.transpose(psal.unsqueeze(0), 0, 1)
        doxy = torch.transpose(doxy.unsqueeze(0), 0, 1)
        output_variable = torch.transpose(output_variable.unsqueeze(0), 0, 1)

        plt.figure(figsize=(9, 12))

        for name, models in models_dict.items():
            model = models["model"]
            model_day = models["day"]
            model_year = models["year"]
            model_space = models["space"]

            output_day = model_day(day)
            output_year = model_year(year.unsqueeze(1))
            output_space = model_space(space)

            output_day = torch.transpose(output_day.unsqueeze(0), 0, 1)
            output_year = torch.transpose(output_year.unsqueeze(0), 0, 1)
            output_space = torch.transpose(output_space.unsqueeze(0), 0, 1)
        
            x = torch.cat((output_day, output_year, output_space, temp, psal, doxy), 1)

            output_test = model(x.float())
            output_test = torch.unsqueeze(output_test, 1)

            depth_output = np.linspace(0, max_pres, len(output_test[0, 0, :].detach().numpy()))  #prima era [0,0,:]

            if mov_average == True:
                output_test = torch_moving_average(output_test, 5)

            if name == "global":
                model_color = plt.cm.tab10.colors[0]
            elif name == "global_fine_tuned":
                model_color = plt.cm.tab10.colors[1]
            elif name == "local":
                model_color = plt.cm.tab10.colors[2]
            else:
                raise ValueError("Unrecognized model name")

            plt.plot(
                output_test[0, 0, :].detach().numpy(), 
                depth_output, 
                color=model_color,
                linewidth=4.0,
                label=f"{name}"
            )

        if mov_average == True:
            output_variable = torch_moving_average(output_variable, 5)
        
        depth_variable = np.linspace(0, max_pres, len(output_variable[0, 0, :].detach().numpy()))    #prima era [0,0,:]
        plt.plot(
            output_variable[0, 0, :].detach().numpy(),
            depth_variable,
            color=plt.cm.tab10.colors[3],
            linewidth=4.0,
            label="measured"
        )

        year = decode_year(year)
        day = decode_day(day[0][0], day[0][1])
        lat, lon = decode_space(space[0][0], space[0][1], space[0][2])
        
        if compute_baseline:
            baseline_profile = get_suazade_profile(
                get_date(sample_dict["day"], sample_dict["year"]),
                lat.item(),
                lon.item(),
                denorm_function(sample_dict["temp"].numpy(), "temp"),
                denorm_function(sample_dict["psal"].numpy(), "psal"),
                denorm_function(sample_dict["doxy"].numpy(), "doxy"),
                sample_dict["target_variable"].numpy()
            )
            depth_baseline = np.linspace(
                0,
                max_pres,
                len(baseline_profile)
            )

            plt.plot(
                baseline_profile, 
                depth_baseline, 
                color=plt.cm.tab10.colors[4], 
                linewidth=4.0, 
                linestyle="--"
            )

        plt.gca().invert_yaxis()

        my_yticks = np.array([i for i in [0 ,250, 500, 750, 999]])   #np.array(index_depths_SOCA[::4])  #np.array(index_depths_SOCA[:-1] + [200])
        my_yticks_label = np.array([5*i for i in [0 ,50, 100, 150]] + [1000])   #np.array(index_depths_SOCA[:-1] + [200])
        plt.yticks(my_yticks, my_yticks_label)
        plt.xlabel(r"Nitrate [$mmol \ m^{-3}$]")
        plt.ylabel(r"Depth [$m$]")
        plt.tight_layout()

        plt.savefig(dir_profile + f"/profile_{year}_{int(day.item())}_{round(lat.item(), 2)}_{round(lon.item(), 2)}_{device_name[0]}.png")
        # plt.show()
        plt.close()

def plot_chla_profiles(
        ds,
        save_dir,
        models_dict,
        mov_average=False,
        soca_path = None,
        soca_indexes = None,
        valid_indexes = [],
        compute_baseline = False
    ):
    print(save_dir)

    dir_profile = os.path.join(save_dir, "profiles")
    if not os.path.exists(dir_profile):
        os.mkdir(path=dir_profile)

    max_pres = dict_max_pressure["CHLA"]

    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="sans-serif",
        font_scale=4.5,
        color_codes=True,
        rc=None
    )

    plot_counter = 0

    for idx, sample_dict in enumerate(ds):

        if plot_counter > 300:
            break

        if idx in valid_indexes or not compute_baseline:
            plot_counter += 1

            plt.figure(figsize=(9, 12))

            year = sample_dict["year"]
            day = sample_dict["day"]
            space = sample_dict["space"]
            temp = sample_dict["temp"]
            psal = sample_dict["psal"]
            doxy = sample_dict["doxy"]
            output_variable = sample_dict["target_variable"]
            device_name = sample_dict["device_name"]

            temp = torch.transpose(temp.unsqueeze(0), 0, 1)
            psal = torch.transpose(psal.unsqueeze(0), 0, 1)
            doxy = torch.transpose(doxy.unsqueeze(0), 0, 1)
            output_variable = torch.transpose(output_variable.unsqueeze(0), 0, 1)

            for name, models in models_dict.items():
                model = models["model"]
                model_day = models["day"]
                model_year = models["year"]
                model_space = models["space"]

                output_day = model_day(day)
                output_year = model_year(year.unsqueeze(1))
                output_space = model_space(space)

                output_day = torch.transpose(output_day.unsqueeze(0), 0, 1)
                output_year = torch.transpose(output_year.unsqueeze(0), 0, 1)
                output_space = torch.transpose(output_space.unsqueeze(0), 0, 1)
            
                x = torch.cat((output_day, output_year, output_space, temp, psal, doxy), 1)

                output_test = model(x.float())
                output_test = torch.unsqueeze(output_test, 1)

                depth_output = np.linspace(0, max_pres, len(output_test[0, 0, :].detach().numpy()))  #prima era [0,0,:]

                if mov_average == True:
                    output_test = torch_moving_average(output_test, 5)

                if name == "global":
                    model_color = plt.cm.tab10.colors[0]
                elif name == "global_fine_tuned":
                    model_color = plt.cm.tab10.colors[1]
                elif name == "local":
                    model_color = plt.cm.tab10.colors[2]
                else:
                    raise ValueError("Unrecognized model name")

                plt.plot(
                    output_test[0, 0, :].detach().numpy(),
                    depth_output,
                    color=model_color,
                    linewidth=4.0,
                    label=f"{name}"
                )

            if mov_average == True:
                output_variable = torch_moving_average(output_variable, 5)
            
            depth_variable = np.linspace(0, max_pres, len(output_variable[0, 0, :].detach().numpy()))    #prima era [0,0,:]
            plt.plot(
                output_variable[0, 0, :].detach().numpy(),
                depth_variable,
                color=plt.cm.tab10.colors[3],
                linewidth=4.0,
                label="measured"
            )

            year = decode_year(year)
            day = decode_day(day[0][0], day[0][1])
            lat, lon = decode_space(space[0][0], space[0][1], space[0][2])
            
            if compute_baseline:
                soca_ds = pd.read_csv(soca_path, header=None)
                original_idx = soca_indexes[idx]
                baseline_profile = get_soca_profile(soca_ds, original_idx)
                if not(np.all(baseline_profile < 100.0) and not np.all(baseline_profile == 0.0)):
                    plot_counter -= 1
                    continue
                plt.plot(
                    baseline_profile,
                    index_depths_SOCA,
                    color= plt.cm.tab10.colors[4],
                    linewidth=4.0,
                    linestyle="--",
                    marker="o",
                    markersize=8,
                    label="SOCA",
                    zorder=5
                )

            plt.gca().invert_yaxis()

            my_yticks = np.array([0, 50, 100, 150, 199])
            my_yticks_label = np.array([0, 50, 100, 150, 200])
            plt.yticks(my_yticks, my_yticks_label)

            plt.xlabel(r"Chlorophyll [$mg \ m^{-3}$]")
            plt.ylabel(r"Depth [$m$]")

            plt.tight_layout()

            plt.savefig(dir_profile + f"/profile_{year}_{int(day.item())}_{round(lat.item(), 2)}_{round(lon.item(), 2)}_{device_name[0]}.png")
            # plt.show()
            plt.close()


def plot_seasonal_error(records, save_dir):

    seasons = {
        "winter": [],
        "spring": [],
        "summer": [],
        "autumn": []
    }

    for r in records:
        season = get_season(r["day"], r["lat"])
        seasons[season].append(r["rmse"])

    order = ["winter", "spring", "summer", "autumn"]
    data = [seasons[s] for s in order]

    plt.figure(figsize=(7, 5))

    bp = plt.boxplot(
        data,
        labels=["Winter", "Spring", "Summer", "Autumn"],
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2)
    )

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    plt.ylabel("RMSE")
    plt.title("Seasonal RMSE distribution")

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "seasonal_rmse.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\n=== Seasonal RMSE ===")

    for s in order:
        arr = np.array(seasons[s])
        print(
            f"{s:10s} "
            f"n={len(arr):5d} "
            f"mean={arr.mean():.6f} "
            f"median={np.median(arr):.6f}"
        )

def plot_global_uncertainty_map(records, save_dir, variable):

    lats = [r["lat"] for r in records]
    lons = [r["lon"] for r in records]
    unc = [r["uncertainty"] for r in records]

    fig = plt.figure(figsize=(14, 7))

    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    sc = ax.scatter(
        lons,
        lats,
        c=unc,
        s=1,
        cmap="turbo",
        alpha=0.6,
        transform=ccrs.PlateCarree()
    )

    plt.colorbar(sc, label="Predictive variance")

    plt.title("Global model Uncertainty")

    plt.savefig(
        os.path.join(save_dir, f"global_uncertainty_map_{variable}.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def plot_global_error_map(records, save_dir, variable):

    lats = [r["lat"] for r in records]
    lons = [r["lon"] for r in records]
    rmses = [r["rmse"] for r in records]

    sns.set_theme(
        context='paper',
        style='whitegrid',
        font='sans-serif',
        font_scale=2.0,
        color_codes=True,
        rc=None
    )

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=1,
        color='gray',
        alpha=0.5,
        linestyle='--'
    )

    gl.top_labels = False
    gl.right_labels = False

    rmses_array = np.array(rmses)

    vmin = np.percentile(rmses_array, 2)
    vmax = np.percentile(rmses_array, 96)

    scatter = ax.scatter(
        lons,
        lats,
        c=rmses,
        cmap='plasma',
        s=15,
        edgecolor='none',
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree()
    )

    # Colorbar
    plt.colorbar(
        scatter
    )

    plt.savefig(
        os.path.join(save_dir, f"global_error_map_{variable}.png"),
        dpi=600,
        bbox_inches="tight"
    )
    plt.close()