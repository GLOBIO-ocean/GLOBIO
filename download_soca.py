import os
import argparse
import copernicusmarine
import numpy as np
import netCDF4 as nc
from copernicusmarine.core_functions.exceptions import CoordinatesOutOfDatasetBounds
from utils import prepare_dataloader, decode_space, get_date
from dataset import FloatDataset
import pandas as pd

#FOR SOCA -- CHLA comparison

COPERNICUS_DATASET = "cmems_obs-mob_glo_bgc-chl-poc_my_0.25deg_P7D-m"
VARIABLES = ["chl"]

def is_valid_nc(path):
    try:
        with nc.Dataset(path) as ds:

            if "chl" not in ds.variables:
                return False

            chl = ds.variables["chl"]

            # Deve avere dati
            if chl.size == 0:
                return False

            # Controllo shape attesa
            if len(chl.shape) != 4:
                return False

            # Il download che stiamo facendo dovrebbe produrre
            # una struttura tipo:
            # (time, depth, lat, lon)

            if chl.shape[0] < 1:
                return False

            if chl.shape[1] < 22:
                return False

            if chl.shape[2] < 1:
                return False

            if chl.shape[3] < 1:
                return False

            # Prova effettivamente a leggere i dati
            data = np.asarray(chl[:])

            if not np.isfinite(data).any():
                return False

        return True

    except Exception as e:
        print(f"Invalid/corrupted NetCDF: {path} -> {e}")
        return False


def find_matching_nc(directory, lat, lon):
    """
    Cerca il file NetCDF associato alla posizione corrente,
    replicando la logica utilizzata nello script della ragazza.

    Copernicus normalmente inserisce lat/lon nel nome del file.
    """

    nc_files = [f for f in os.listdir(directory) if f.endswith(".nc")]

    matching_files = []

    for f in nc_files:
        f_lower = f.lower()
        lat_match = str(abs(int(lat))) in f_lower
        lon_match = str(abs(int(lon))) in f_lower

        if lat_match and lon_match:
            matching_files.append(f)

    if len(matching_files) == 0:
        return None
    
    return os.path.join(directory, matching_files[0])


def download_copernicus_dataset(
        list_lons, 
        list_lats, 
        list_times, 
        out_chl_directory, 
        max_depth=200, 
        time_step="weekly"
    ):
    chla = []
    wrong_p_index = []

    for i in range(len(list_times)):
        
        out_chl_directory_i = os.path.join(
            out_chl_directory,
            str(list_times[i][:4]),
            str(list_times[i][5:7])
        )
        if time_step in ["daily", "weekly"]:
            out_chl_directory_i = os.path.join(
                out_chl_directory_i,
                str(list_times[i][8:10])
            )
        os.makedirs(out_chl_directory_i, exist_ok=True)

        try:

            matching_file = find_matching_nc(
                            out_chl_directory_i,
                            list_lats[i],
                            list_lons[i]
                        )
            
            if matching_file is not None:
                print(f"[{i}] Already downloaded")
                if is_valid_nc(matching_file):
                    print(f"[{i}] Already downloaded and valid")
                else:
                    print(f"[{i}] Existing file is invalid -> redownloading")
                    os.remove(matching_file)
                    matching_file = None

            if matching_file is None:
                print(f"[{i}] Downloading...")

                copernicusmarine.subset(
                    dataset_id=COPERNICUS_DATASET,
                    variables=VARIABLES,
                    minimum_longitude=list_lons[i],
                    maximum_longitude=list_lons[i],
                    minimum_latitude=list_lats[i],
                    maximum_latitude=list_lats[i],
                    start_datetime=list_times[i],
                    end_datetime=list_times[i],
                    minimum_depth=0,
                    maximum_depth=max_depth,
                    output_directory=out_chl_directory_i
                )

                matching_file = find_matching_nc(
                    out_chl_directory_i,
                    list_lats[i],
                    list_lons[i]
                )

                if matching_file is None or not is_valid_nc(matching_file):
                    print(f"[{i}] ERROR: downloaded file is invalid")
                    wrong_p_index.append(i)
                    chla.append(np.zeros([22]))
                    continue

            # ------------------------------------------------
            # Read CHL
            # ------------------------------------------------

            ds_chl = nc.Dataset(matching_file)
            chl = np.array(ds_chl.variables["chl"][:])
            ds_chl.close()
            chl = np.squeeze(chl, axis=(0, 2, 3))
            chla.append(chl)

        except CoordinatesOutOfDatasetBounds:
            print(f"WARNING: coordinates out of bounds for profile {i}")
            wrong_p_index.append(i)
            chla.append(np.zeros([22]))

    return chla, wrong_p_index


def main(
    data_dir, 
    ds_name, 
    path_ds_test,
    split_type
    ):

    save_dir = os.path.join(data_dir, ds_name, "soca_profiles", split_type)
    os.makedirs(save_dir,exist_ok=True)

    dataset = FloatDataset(
        path_df=path_ds_test,
        stats_path=os.path.join(os.getcwd(), f"normalization_stats_{ds_name}.json")
    )
    dataloader = prepare_dataloader(dataset, batch_size=1)

    # SOCA INPUT LISTS (aligned with dataloader)
    list_lats = []
    list_lons = []
    list_times = []

    for i, sample_dict in enumerate(dataloader):

        year = sample_dict["year"]
        day = sample_dict["day"]
        space = sample_dict["space"]

        date = get_date(day, year)
        lat, lon = decode_space(space[0][0], space[0][1], space[0][2])

        list_lats.append(float(lat.item()))
        list_lons.append(float(lon.item()))
        list_times.append(date + "T00:00:00")

    print()
    print("Number of profiles:", len(list_times))
    print()

    chla, wrong_p_index = download_copernicus_dataset(
        list_lons=list_lons,
        list_lats=list_lats,
        list_times=list_times,
        out_chl_directory=save_dir,
        max_depth=200,
        time_step="weekly"
    )

    print()
    print("Profiles without valid Copernicus file: ", len(wrong_p_index))

    if len(wrong_p_index) > 0:
        print(
            "Wrong profile indexes:",
            wrong_p_index
        )

    ds = pd.read_csv(path_ds_test, header=None)

    df = pd.DataFrame({
        "year": [float(ds.iloc[:, l].iloc[1].strip()) for l in range(1, ds.shape[1])],
        "day_rad": [float(ds.iloc[:, l].iloc[2].strip()) for l in range(1, ds.shape[1])],
        "lat": list_lats,
        "lon": list_lons,
        "chla_SOCA": chla,
    })

    df_T = df.T
    df_T.columns = [f"sample_{i}" for i in range(df_T.shape[1])]
    SOCA_csv_dir = f"SOCA_test_{split_type}_{ds_name}.csv"
    df_T.to_csv(SOCA_csv_dir)

    print(f"SOCA download completed: {len(list_times)} profiles saved")

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--ds_name", type=str, required=True)
    parser.add_argument("--path_ds_test", type=str, required=True)
    parser.add_argument("--split_type", type=str, required=True, choices=['float', 'profile'])

    args = parser.parse_args()
    
    main(args.out_dir, args.ds_name, args.path_ds_test, args.split_type)