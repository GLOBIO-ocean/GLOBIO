"""Unit and end-to-end tests on synthetic data (CPU only).

Run from the repository root:
    python -m unittest discover -s tests -v
"""
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from globio.constants import REGIONS, SOCA_LEVEL_INDICES, pressure_levels  # noqa: E402
from globio.data import (ProfileDataset, compute_normalization_stats, day_rad_from_date,  # noqa: E402
                         decode_space, float_id, parse_profile, profile_date, read_profile_csv,
                         region_mask, season)
from globio.inference import enable_mc_dropout, predict, predict_mc, recalibrate_batchnorm  # noqa: E402
from globio.metrics import (compare_methods, depth_statistics, holm, profile_rmse, summarize,  # noqa: E402
                            uncertainty_metrics, wilcoxon_best_vs_all)
from globio.models import GlobioNet, load_checkpoint, resolve_norm_stats, save_checkpoint  # noqa: E402
from globio.training import TrainConfig, apply_ft_strategy, loss_function, train  # noqa: E402
from tests.synthetic import make_profiles, write_csv  # noqa: E402

STATS = {"temp": [15.0, 6.0], "psal": [35.0, 0.5], "doxy": [200.0, 30.0]}


def run(*args):
    env = {**os.environ, "OMP_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(f"{args} failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class DataTests(TempDirTest):
    def test_parse_profile(self):
        values = parse_profile("tensor([1.0, 2.5,\n 3e-2])", n_levels=3)
        np.testing.assert_allclose(values, [1, 2.5, 0.03], rtol=1e-6)
        self.assertEqual(values.dtype, np.float32)
        np.testing.assert_allclose(parse_profile("[1, 2]", None), [1, 2])
        with self.assertRaises(ValueError):
            parse_profile("tensor([1.0, 2.0])")  # not 200 levels
        with self.assertRaises(ValueError):
            parse_profile("tensor([1.0, x])", None)

    def test_read_csv_labelled_and_unlabelled(self):
        data = make_profiles(5)
        for labelled in (True, False):
            path = write_csv(self.tmp / f"d{labelled}.csv", data, labelled=labelled)
            out = read_profile_csv(path, "CHLA")
            self.assertEqual(out["variable"], "CHLA")
            self.assertEqual(list(out["profile_ids"]), data["profile_ids"])
            np.testing.assert_allclose(out["lat"], data["lat"])
            np.testing.assert_allclose(out["temp"], data["temp"], atol=1e-4)
        with self.assertRaises(ValueError):
            read_profile_csv(write_csv(self.tmp / "n.csv", data, variable="NITRATE"), "CHLA")

    def test_dataset_items_and_encodings(self):
        data = make_profiles(6)
        dataset = ProfileDataset(write_csv(self.tmp / "d.csv", data), "CHLA", STATS)
        item = dataset[2]
        self.assertEqual(item["temp"].shape, (200,))
        self.assertEqual(item["space"].shape, (3,))
        self.assertAlmostEqual(item["year"].item(), data["year"][2] - 2010)
        self.assertAlmostEqual(item["temp"][0].item(), (round(data["temp"][2, 0], 4) - 15) / 6, places=4)
        np.testing.assert_allclose(item["day"].numpy(), [np.sin(data["day_rad"][2]), np.cos(data["day_rad"][2])],
                                   atol=1e-6)
        lat, lon = decode_space(item["space"].numpy())
        self.assertAlmostEqual(float(lat), data["lat"][2], places=4)
        self.assertAlmostEqual(float(lon), data["lon"][2], places=4)
        # renormalization
        dataset.set_normalization(None)
        self.assertAlmostEqual(dataset[2]["temp"][0].item(), round(data["temp"][2, 0], 4), places=4)

    def test_invalid_profiles_are_dropped(self):
        data = make_profiles(5)
        data["psal"][1, :] = 0.1          # broken salinity sensor
        data["target"][3, 10] = 99999.0   # fill value
        dataset = ProfileDataset(write_csv(self.tmp / "d.csv", data), "CHLA", STATS, verbose=False)
        self.assertEqual(len(dataset), 3)
        self.assertNotIn(data["profile_ids"][1], dataset.profile_ids)
        self.assertEqual(len(ProfileDataset(self.tmp / "d.csv", "CHLA", drop_invalid=False)), 5)

    def test_dataset_without_target(self):
        path = write_csv(self.tmp / "d.csv", make_profiles(4), with_target=False)
        dataset = ProfileDataset(path, "CHLA", STATS)
        self.assertFalse(dataset.has_target)
        self.assertTrue(torch.isnan(dataset[0]["target"]).all())

    def test_regions(self):
        lat = np.array([40, 40, 50, -70, 10, 10])
        lon = np.array([10, -20, -30, 0, 150, -100])
        np.testing.assert_array_equal(region_mask(lat, lon, "mediterranean"), [1, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(region_mask(lat, lon, "north_atlantic"), [0, 0, 1, 0, 0, 0])
        np.testing.assert_array_equal(region_mask(lat, lon, "pacific"), [0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(region_mask(lat, lon, "r6"), [0, 0, 0, 1, 0, 0])
        with self.assertRaises(ValueError):
            region_mask(lat, lon, "atlantis")
        self.assertIn("bay_of_bengal", REGIONS)

    def test_dates_and_seasons(self):
        for date in (dt.datetime(2020, 1, 1), dt.datetime(2019, 6, 15), dt.datetime(2021, 12, 30)):
            self.assertEqual(profile_date(date.year, day_rad_from_date(date)), date)
        self.assertEqual(profile_date(2020, day_rad_from_date(dt.datetime(2020, 1, 31))), dt.datetime(2020, 2, 1))
        july = day_rad_from_date(dt.datetime(2020, 7, 15))
        self.assertEqual(season(july, 45), "summer")
        self.assertEqual(season(july, -45), "winter")

    def test_float_id_and_stats(self):
        self.assertEqual(float_id("o_scratch/OGS/5906529/SD5906529_081"), "5906529")
        dataset = ProfileDataset(write_csv(self.tmp / "d.csv", make_profiles(8)), "CHLA")
        stats = compute_normalization_stats(dataset)
        self.assertAlmostEqual(stats["temp"][0], float(dataset.temp.astype(float).mean()), places=5)
        normalized = ProfileDataset(self.tmp / "d.csv", "CHLA", stats)
        self.assertAlmostEqual(float(normalized._encoded["psal"].mean()), 0, places=4)

    def test_from_arrays_and_subset(self):
        data = make_profiles(4)
        dataset = ProfileDataset.from_arrays(data["year"], data["day_rad"], data["lat"], data["lon"],
                                             data["temp"], data["psal"], data["doxy"], variable="CHLA",
                                             norm_stats=STATS)
        self.assertEqual(len(dataset), 4)
        sub = dataset.subset([3, 1])
        np.testing.assert_allclose(sub.lat, data["lat"][[3, 1]])
        self.assertEqual(sub.norm_stats, dataset.norm_stats)


class ModelTests(TempDirTest):
    def test_shapes_and_positivity(self):
        for variable in ("CHLA", "NITRATE", "BBP700"):
            model = GlobioNet(variable).eval()
            batch = {"year": torch.zeros(3), "day": torch.zeros(3, 2), "space": torch.zeros(3, 3),
                     "temp": torch.randn(3, 200), "psal": torch.randn(3, 200), "doxy": torch.randn(3, 200)}
            out = model.forward_batch(batch)
            self.assertEqual(out.shape, (3, 200))
            self.assertTrue((out > 0).all())

    def test_checkpoint_round_trip(self):
        model = GlobioNet("NITRATE", 0.3)
        save_checkpoint(self.tmp / "c.pt", model, STATS, epoch=4)
        loaded, info = load_checkpoint(self.tmp / "c.pt")
        self.assertEqual(info["variable"], "NITRATE")
        self.assertEqual(info["dp_rate"], 0.3)
        self.assertEqual(info["norm_stats"], STATS)
        for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
            self.assertTrue(torch.equal(a, b))
        with self.assertRaises(ValueError):
            load_checkpoint(self.tmp / "c.pt", "CHLA")

    def test_legacy_checkpoint(self):
        """Reference checkpoints: only the four state dicts."""
        model = GlobioNet("CHLA")
        torch.save(model.subnet_state_dicts() | {"epoch": 1}, self.tmp / "old.pt")
        with self.assertRaises(ValueError):
            load_checkpoint(self.tmp / "old.pt")  # CHLA or BBP700?
        loaded, info = load_checkpoint(self.tmp / "old.pt", "CHLA")
        self.assertIsNone(info["norm_stats"])
        with self.assertRaises(ValueError):
            resolve_norm_stats(info)
        path = self.tmp / "stats.json"
        path.write_text(json.dumps(STATS))
        self.assertEqual(resolve_norm_stats(info, path), STATS)

    def test_ft_strategies(self):
        model = GlobioNet("NITRATE")
        apply_ft_strategy(model, "freeze_backbone")
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertFalse(any(name.startswith("mlp_") for name in trainable))
        self.assertNotIn("cnn.deconv1.weight", trainable)
        self.assertNotIn("cnn.bn5.weight", trainable)
        self.assertIn("cnn.conv10.weight", trainable)  # not an early layer
        self.assertIn("cnn.fc11.weight", trainable)

    def test_loss(self):
        prediction = torch.tensor([[1.0, 2.0, 4.0]])
        target = torch.tensor([[1.0, 1.0, 1.0]])
        expected = (0 + 1 + 9) / 3 + 0.5 * (1 + 2) / 2
        self.assertAlmostEqual(loss_function(prediction, target, 0.5).item(), expected, places=6)
        # Default weight 1.99e-2 reproduces the loss of the original code: 1e-4 * sum of |differences|.
        torch.manual_seed(0)
        p, t = torch.rand(32, 200), torch.rand(32, 200)
        old = torch.nn.functional.mse_loss(p, t) + 1e-4 * torch.abs(p[:, 1:] - p[:, :-1]).sum()
        self.assertAlmostEqual(loss_function(p, t, 1.99e-2).item(), old.item(), places=6)


class InferenceTests(TempDirTest):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        self.model = GlobioNet("CHLA")
        self.dataset = ProfileDataset(write_csv(self.tmp / "d.csv", make_profiles(10)), "CHLA", STATS)

    def test_predict_is_deterministic_and_batch_independent(self):
        a = predict(self.model, self.dataset, batch_size=3, device="cpu")
        b = predict(self.model, self.dataset, batch_size=10, device="cpu")
        self.assertEqual(a.shape, (10, 200))
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
        self.assertTrue(self.model.training)  # mode restored

    def test_mc_dropout(self):
        bn = self.model.cnn.bn1
        running_mean = bn.running_mean.clone()
        out = predict_mc(self.model, self.dataset, n_samples=20, batch_size=4, device="cpu", seed=1,
                         return_samples=True)
        self.assertEqual(out["samples"].shape, (20, 10, 200))
        self.assertTrue((out["std"] > 0).all())
        np.testing.assert_allclose(out["mean"], out["samples"].mean(0), rtol=1e-5)
        np.testing.assert_allclose(out["deterministic"], predict(self.model, self.dataset, device="cpu"),
                                   rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.equal(running_mean, bn.running_mean))  # BatchNorm untouched
        again = predict_mc(self.model, self.dataset, n_samples=20, batch_size=4, device="cpu", seed=1)
        np.testing.assert_array_equal(out["mean"], again["mean"])  # reproducible

    def test_enable_mc_dropout(self):
        count = enable_mc_dropout(self.model)
        self.assertEqual(count, 6)
        self.assertFalse(self.model.cnn.bn1.training)
        self.assertTrue(self.model.cnn.do1.training)
        with self.assertRaises(ValueError):
            enable_mc_dropout(GlobioNet("CHLA", dp_rate=0.0))

    def test_recalibrate_batchnorm(self):
        from globio.inference import make_loader
        self.model.train()
        recalibrate_batchnorm(self.model, make_loader(self.dataset, 5), "cpu")
        self.assertTrue(self.model.training)
        self.assertEqual(self.model.cnn.bn1.momentum, 0.1)
        self.assertFalse(torch.equal(self.model.cnn.bn1.running_mean, torch.zeros(53)))


class MetricTests(unittest.TestCase):
    def test_profile_rmse(self):
        prediction = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        target = np.array([[1.0, 2.0, 5.0], [1.0, 1.0, 1.0]])
        np.testing.assert_allclose(profile_rmse(prediction, target), [np.sqrt(4 / 3), 1])
        np.testing.assert_allclose(profile_rmse(prediction, target, levels=[0, 1]), [0, 1])
        mask = np.array([[True, True, False], [True, False, False]])
        np.testing.assert_allclose(profile_rmse(prediction, target, mask=mask), [0, 1])

    def test_holm_matches_statsmodels(self):
        try:
            from statsmodels.stats.multitest import multipletests
        except ImportError:
            self.skipTest("statsmodels not installed")
        rng = np.random.default_rng(0)
        for _ in range(50):
            p = rng.uniform(0, 0.1, rng.integers(1, 6))
            np.testing.assert_array_equal(holm(p), multipletests(p, alpha=0.05, method="holm")[0])

    def test_wilcoxon(self):
        rng = np.random.default_rng(0)
        base = rng.uniform(0.5, 1.5, 200)
        errors = {"good": base * 0.7, "medium": base * 0.9 + 0.01, "bad": base}
        best, pvalues = wilcoxon_best_vs_all(errors)
        self.assertEqual(best, {"good": True, "medium": False, "bad": False})
        self.assertLess(pvalues["good"]["bad"], 1e-6)
        table = compare_methods(errors)
        self.assertEqual(list(table.method), ["good", "medium", "bad"])
        same = wilcoxon_best_vs_all({"a": base, "b": base})[0]
        self.assertEqual(same, {"a": False, "b": False})

    def test_summaries(self):
        import pandas as pd
        meta = pd.DataFrame({"lat": [70.0, 40.0, -70.0], "lon": [0.0, 10.0, 0.0],
                             "season": ["summer", "winter", "winter"]})
        summary = summarize({"m": np.array([1.0, 2.0, 3.0])}, meta)
        row = summary.set_index("group").loc
        self.assertEqual(row["all", "median"], 2)
        self.assertEqual(row["winter", "n"], 2)
        self.assertEqual(row["r1", "mean"], 1)
        self.assertEqual(row["r6", "mean"], 3)
        prediction = np.ones((3, 4))
        depth = depth_statistics(prediction, np.zeros((3, 4)), meta)
        self.assertTrue((depth.query("group == 'r2' and season == 'all'").rmse == 1).all())

    def test_uncertainty_metrics(self):
        target = np.array([[0.0, 2.0, 99999.0], [0.0, 0.0, 0.0]])
        mean, std = np.zeros_like(target), np.ones_like(target)
        mask = np.array([[True, True, False], [True, True, True]])
        profiles, coverage, vertical = uncertainty_metrics(target, mean, std, mask, np.arange(3))
        self.assertEqual(coverage.iloc[0].entire_profile_percent, 50)
        self.assertEqual(coverage.iloc[0].pointwise_percent, 75)
        self.assertEqual(coverage.iloc[1].entire_profile_percent, 100)
        self.assertAlmostEqual(profiles.iloc[0].rmse, np.sqrt(2))
        self.assertEqual(vertical.iloc[2].n_profiles, 1)
        with self.assertRaises(ValueError):
            uncertainty_metrics(target, mean, std, np.zeros_like(mask))


class EndToEndTests(TempDirTest):
    """Training, fine-tuning, resume and all the command-line scripts."""

    @classmethod
    def setUpClass(cls):
        cls._shared = tempfile.TemporaryDirectory()
        root = Path(cls._shared.name)
        for name, n, seed in (("train", 90, 0), ("val", 30, 1), ("test", 30, 2)):
            write_csv(root / f"{name}.csv", make_profiles(n, seed))
        cls.data = root
        config = TrainConfig(variable="CHLA", train_csv=str(root / "train.csv"), val_csv=str(root / "val.csv"),
                             output_dir=str(root / "runs"), run_name="global", epochs=2, batch_size=16,
                             device="cpu", save_every=1)
        cls.run_dir = train(config)
        cls.checkpoint = cls.run_dir / "checkpoints" / "best.pt"

    @classmethod
    def tearDownClass(cls):
        cls._shared.cleanup()

    def test_training_outputs(self):
        history = json.loads((self.run_dir / "history.json").read_text())
        self.assertEqual([e["epoch"] for e in history["epochs"]], [1, 2])
        for name in ("best.pt", "last.pt", "epoch_0001.pt", "epoch_0002.pt"):
            self.assertTrue((self.run_dir / "checkpoints" / name).is_file())
        _, info = load_checkpoint(self.checkpoint)
        stats = json.loads((self.run_dir / "normalization_stats.json").read_text())
        self.assertEqual(info["norm_stats"], stats)
        self.assertEqual(info["epoch"], history["best_epoch"])

    def test_fine_tuning_and_resume(self):
        out = self.tmp / "runs"
        run(ROOT / "train.py", "--variable", "CHLA", "--train_csv", self.data / "train.csv",
            "--val_csv", self.data / "val.csv", "--init_checkpoint", self.checkpoint, "--epochs", "1",
            "--output_dir", out, "--run_name", "ft", "--device", "cpu", "--batch_size", "8")
        ft_dir = out / "CHLA" / "ft"
        _, info = load_checkpoint(ft_dir / "checkpoints" / "last.pt")
        _, base = load_checkpoint(self.checkpoint)
        self.assertEqual(info["norm_stats"], base["norm_stats"])  # statistics of the pretrained model
        self.assertEqual(info["config"]["ft_strategy"], "aggressive")
        run(ROOT / "train.py", "--resume", ft_dir, "--epochs", "2")
        history = json.loads((ft_dir / "history.json").read_text())
        self.assertEqual([e["epoch"] for e in history["epochs"]], [1, 2])

    def test_multiprocess_training(self):
        config = TrainConfig(variable="CHLA", train_csv=str(self.data / "train.csv"),
                             val_csv=str(self.data / "val.csv"), output_dir=str(self.tmp), run_name="ddp",
                             epochs=1, batch_size=8, device="cpu", n_procs=2)
        run_dir = train(config)
        self.assertTrue((run_dir / "checkpoints" / "best.pt").is_file())

    def test_predict_script(self):
        out = self.tmp / "pred"
        run(ROOT / "predict.py", "--checkpoint", self.checkpoint, "--csv", self.data / "test.csv",
            "--output", out, "--mc_samples", "5", "--device", "cpu")
        arrays = np.load(f"{out}.npz")
        self.assertEqual(arrays["mc_mean"].shape, (30, 200))
        self.assertEqual(arrays["prediction"].shape, (30, 200))
        np.testing.assert_allclose(arrays["pressure"], pressure_levels("CHLA"))
        import pandas as pd
        table = pd.read_csv(f"{out}.csv")
        self.assertIn("rmse", table)
        self.assertIn("uncertainty_rms", table)
        no_target = write_csv(self.tmp / "new.csv", make_profiles(4, 9), with_target=False)
        run(ROOT / "predict.py", "--checkpoint", self.checkpoint, "--csv", no_target, "--output", self.tmp / "p2",
            "--device", "cpu")
        self.assertNotIn("target", np.load(self.tmp / "p2.npz"))

    def test_evaluate_global_script(self):
        out = self.tmp / "global"
        run(ROOT / "evaluate_global.py", "--checkpoint", self.checkpoint, "--data", f"val={self.data / 'val.csv'}",
            "--data", f"test={self.data / 'test.csv'}", "--depth_stats", "--output_dir", out, "--device", "cpu")
        import pandas as pd
        summary = pd.read_csv(out / "summary.csv")
        self.assertEqual(set(summary.set), {"val", "test"})
        self.assertEqual(summary.query("set == 'test' and group == 'all'").n.item(), 30)
        self.assertTrue((out / "depth_test.csv").is_file())

    def test_evaluate_regional_with_soca(self):
        # SOCA file aligned by profile id, in reversed column order.
        import pandas as pd
        test = ProfileDataset(self.data / "test.csv", "CHLA")
        values = test.target[:, SOCA_LEVEL_INDICES] * 1.1
        values[0] = 0.0  # failed download -> invalid
        order = np.arange(len(test))[::-1]
        soca = pd.DataFrame([test.year[order], test.day_rad[order], test.lat[order], test.lon[order],
                             [np.array2string(v) for v in values[order]]],
                            index=["year", "day_rad", "lat", "lon", "chla_SOCA"], columns=test.profile_ids[order])
        soca.to_csv(self.tmp / "soca.csv")
        out = self.tmp / "regional"
        stdout = run(ROOT / "evaluate_regional.py", "--test_csv", self.data / "test.csv",
                     "--model", f"global={self.checkpoint}", "--model", f"other={self.checkpoint}",
                     "--baseline", "soca", "--soca_csv", self.tmp / "soca.csv", "--output_dir", out,
                     "--device", "cpu")
        self.assertIn("Profiles with a valid SOCA estimate: 29 of 30", stdout)
        profiles = pd.read_csv(out / "profiles.csv")
        np.testing.assert_allclose(profiles.rmse_soca[1:], 0.1 * np.sqrt(
            (test.target[1:, SOCA_LEVEL_INDICES] ** 2).mean(1)), rtol=1e-4)
        table = pd.read_csv(out / "comparison_with_baseline.csv")
        self.assertEqual(set(table.method), {"global", "other", "soca"})
        self.assertTrue((table.n == 29).all())

    def test_evaluate_uncertainty_script(self):
        out = self.tmp / "unc"
        run(ROOT / "evaluate_uncertainty.py", "--model", f"global={self.checkpoint}", "--test_csv",
            self.data / "test.csv", "--n_samples", "5", "--output_dir", out, "--device", "cpu")
        import pandas as pd
        first = pd.read_csv(out / "coverage.csv")
        self.assertEqual(list(first.k), [1, 2, 3])
        run(ROOT / "evaluate_uncertainty.py", "--results", out / "predictions.npz", "--output_dir", self.tmp / "u2")
        second = pd.read_csv(self.tmp / "u2" / "coverage.csv")
        pd.testing.assert_frame_equal(first, second)

    def test_data_tools(self):
        out = self.tmp
        run("-m", "tools.split_dataset", self.data / "train.csv", self.data / "val.csv", "--split", "float",
            "--fraction", "0.25", "--out_rest", out / "a.csv", "--out_fraction", out / "b.csv")
        a, b = ProfileDataset(out / "a.csv", "CHLA"), ProfileDataset(out / "b.csv", "CHLA")
        self.assertEqual(len(a) + len(b), 120)
        self.assertTrue({float_id(p) for p in a.profile_ids}.isdisjoint({float_id(p) for p in b.profile_ids}))
        data = make_profiles(6)
        data["psal"][2] = 0.1
        data["target"][4] = 0.3  # constant profile
        write_csv(out / "dirty.csv", data)
        run("-m", "tools.clean_dataset", out / "dirty.csv")
        self.assertEqual(len(ProfileDataset(out / "dirty_clean.csv", "CHLA", drop_invalid=False)), 4)
        years = sorted(set(int(y) for y in ProfileDataset(self.data / "test.csv", "CHLA").year))
        run("-m", "tools.select_profiles", self.data / "test.csv", "--year", str(years[0]), "--out", out / "y.csv")
        self.assertTrue((ProfileDataset(out / "y.csv", "CHLA").year == years[0]).all())
        legacy = out / "legacy.pt"
        torch.save({k: v for k, v in torch.load(self.checkpoint, weights_only=True).items()
                    if k in ("model", "mlp_day", "mlp_year", "mlp_space")}, legacy)
        (out / "stats.json").write_text(json.dumps(STATS))
        run("-m", "tools.upgrade_checkpoint", "--checkpoint", legacy, "--variable", "CHLA",
            "--stats_path", out / "stats.json", "--output", out / "upgraded.pt")
        _, info = load_checkpoint(out / "upgraded.pt")
        self.assertEqual(info["norm_stats"], STATS)


if __name__ == "__main__":
    unittest.main()
