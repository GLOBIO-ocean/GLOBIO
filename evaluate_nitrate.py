import torch
import os

import numpy as np

from utils import prepare_dataloader, load_models
from evaluation.metrics import compute_sample_errors, compute_canyonb_errors
from evaluation.plot import plot_nitrate_profiles
from evaluation.statistical_tests import wilcoxon_best_vs_all
from dataset import FloatDataset



def main(
    global_model_path:str,
    global_ft_model_path:str,
    local_model_path:str,
    main_dir,
    data_dir,
    exp_name,
    region,
    ds_time,
    split_type,
    ds_name,
    compute_baseline=False
    ):

    max_samples = 700

    save_dir = os.path.join(main_dir, "experiments", exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    path_ds_test = os.path.join(
        data_dir, 
        ds_time, 
        f"NITRATE_datasets/",
        f"test_{split_type}_wise_{ds_name}.csv"
    )

    print(f"Test Path: {path_ds_test}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = FloatDataset(
        path_df=path_ds_test, 
        stats_path=os.path.join(os.getcwd(), f"normalization_stats_{ds_name}.json"), 
        region=region
        )

    print(f"Test profiles after region filtering: {len(test_dataset)}")

    test_dataloader = prepare_dataloader(test_dataset, batch_size=1)

    models_dict = {
        "global": load_models("NITRATE", global_model_path, device),
        "global_fine_tuned": load_models("NITRATE", global_ft_model_path, device),
        "local": load_models("NITRATE", local_model_path, device)
    }

    print(f"\n=== Per-sample RMSE for region {region} ===")

    errors_dict = {}

    if compute_baseline:
        baseline = "canyonb"
        errors_dict[baseline] = compute_canyonb_errors(test_dataloader, test_dataset.denormalize, max_samples=max_samples)
    else:
        baseline = None
    
    for name, models in models_dict.items():
    
            test_errors = compute_sample_errors(
                test_dataloader, 
                models["model"], 
                models["day"], 
                models["year"], 
                models["space"], 
                device,
                max_samples=max_samples
            )
            test_errors = np.array(test_errors)

            errors_dict[name] = test_errors
            
            print("\nTEST")
            print(
                f"{name:15s} | "
                f"n_rmse={test_errors.size} | "
                f"finite={np.isfinite(test_errors).sum()} | "
                f"non_finite={(~np.isfinite(test_errors)).sum()} | "
                f"mean={test_errors.mean():.6f} | "
                f"median={np.median(test_errors):.6f} |"
                f"std={np.std(test_errors):.6f}"
            )

    if compute_baseline:
        print(
            f"{baseline:15s} | "
            f"n_rmse={errors_dict[baseline].size} | "
            f"finite={np.isfinite(errors_dict[baseline]).sum()} | "
            f"non_finite={(~np.isfinite(errors_dict[baseline])).sum()} | "
            f"mean={errors_dict[baseline].mean():.6f} | "
            f"median={np.median(errors_dict[baseline]):.6f} | "
            f"std={np.std(errors_dict[baseline]):.6f}"
        )

    # ----------------------------
    # 2. RANKING (DESCRIPTIVE)
    # ----------------------------

    print("\n=== Ranking by median RMSE ===")

    ranking = sorted(errors_dict.items(), key=lambda x: np.median(x[1]))

    for i, (name, err) in enumerate(ranking):
        print(f"{i+1}. {name:15s} median={np.median(err):.6f}")

    best_model = ranking[0][0]

    print(f"\nBest (by median RMSE): {best_model}")

    # ----------------------------
    # 3. STATISTICAL TEST (GLOBAL BEST)
    # ----------------------------

    print("\n=== Wilcoxon + Holm (best vs all) ===")

    statistical_errors = {
        name: np.round(errors, decimals=2)
        for name, errors in errors_dict.items()
    }
    best_res, _ = wilcoxon_best_vs_all(statistical_errors, alpha=0.05)

    for m, v in best_res.items():
        print(f"{m:15s} -> {'BEST' if v else 'not best'}")

    # ----------------------------
    # 5. FINAL CONCLUSION
    # ----------------------------

    is_best_significant = best_res[best_model]

    print("\n=== FINAL CONCLUSION ===")

    if is_best_significant:
        print(f" {best_model} is SIGNIFICANTLY better than all others")
    else:
        print(f" {best_model} is NOT significantly better than all others")

    plot_nitrate_profiles(
        test_dataloader,
        save_dir=save_dir,
        models_dict=models_dict,
        mov_average=True,
        denorm_function = test_dataset.denormalize,
        compute_baseline = compute_baseline
    )
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_global_model", type=str, required=True)
    parser.add_argument("--path_ft_model", type=str, required=True)
    parser.add_argument("--path_local_model", type=str, required=True)
    parser.add_argument("--main_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--ds_time", type=str, default="old")
    parser.add_argument("--ds_name", type=str, required=True)
    parser.add_argument("--split_type", type=str, required=True, choices=['float', 'profile'])
    parser.add_argument('--region', type=str, default=None,
                        choices=[
                            'mediterranean', 
                            'indian_ocean', 
                            'bay_of_bengal', 
                            'southern_ocean', 
                            'north_atlantic', 
                            'north_pacific'
                            ]
                            )
    parser.add_argument("--compute_baseline", type=bool, default=False)

    args = parser.parse_args()

    main(
        args.path_global_model,
        args.path_ft_model,
        args.path_local_model,
        args.main_dir,
        args.data_dir, 
        args.exp_name, 
        args.region,
        args.ds_time,
        args.split_type,
        args.ds_name,
        args.compute_baseline
    )