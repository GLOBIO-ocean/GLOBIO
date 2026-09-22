import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
import os
from datetime import date
import time

from architecture.mlp import MLPDay, MLPYear, MLPSpace
from architecture.conv1med_dp import get_conv_model
from dataset import FloatDataset
from utils import compute_and_save_normalization_stats , create_directories, save_ds_info_ddp

print("end installation")



def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)



class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        model_mlp_day: torch.nn.Module, 
        model_mlp_year: torch.nn.Module,
        model_mlp_space: torch.nn.Module, 
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        alpha_smooth_reg: float,
        ft: bool = False
    ) -> None:
        self.gpu_id = gpu_id
        model = model.to(gpu_id)
        model_mlp_day = model_mlp_day.to(gpu_id)
        model_mlp_year = model_mlp_year.to(gpu_id)
        model_mlp_space = model_mlp_space.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.alpha_smooth_reg = alpha_smooth_reg
        self.model = DDP(model, device_ids=[gpu_id])
        if ft:
            self.model_mlp_day = model_mlp_day
            self.model_mlp_year = model_mlp_year
            self.model_mlp_space = model_mlp_space
        else:
            self.model_mlp_day = DDP(model_mlp_day, device_ids=[gpu_id])
            self.model_mlp_year = DDP(model_mlp_year, device_ids=[gpu_id])
            self.model_mlp_space = DDP(model_mlp_space, device_ids=[gpu_id])
        self.train_loss_history = []
        self.val_loss_history = []
        self.best_val_loss = float("inf")
        self.best_epoch = -1

    def _save_losses(self, save_dir):
        path = os.path.join(save_dir, "losses.json")
        with open(path, "w") as f:
            json.dump({
                "train": self.train_loss_history,
                "val": self.val_loss_history
            }, f)

    def _data_train_preprocess(self, input_data_dict, list_mlp_models):
        training_year = input_data_dict["year"].to(self.gpu_id)
        training_day = input_data_dict["day"].to(self.gpu_id)
        training_space = input_data_dict["space"].to(self.gpu_id)
        training_temp = input_data_dict["temp"].to(self.gpu_id)
        training_psal = input_data_dict["psal"].to(self.gpu_id)
        training_doxy = input_data_dict["doxy"].to(self.gpu_id)
        
        output_day = list_mlp_models[0](training_day)
        training_year = training_year.unsqueeze(1)
        output_year = list_mlp_models[1](training_year.float())
        output_space = list_mlp_models[2](training_space.float())
        output_day = torch.transpose(output_day.unsqueeze(0), 0, 1)
        output_year = torch.transpose(output_year.unsqueeze(0), 0, 1)
        
        output_space = torch.transpose(output_space.unsqueeze(0), 0, 1)
        training_temp = torch.transpose(training_temp.unsqueeze(0), 0, 1)
        training_psal = torch.transpose(training_psal.unsqueeze(0), 0, 1)
        training_doxy = torch.transpose(training_doxy.unsqueeze(0), 0, 1)
        training_x = torch.cat(
            (output_day, output_year, output_space, training_temp, training_psal, training_doxy), 1)
        return training_x

    def _run_batch(self, source, targets, losses):
        self.optimizer.zero_grad()
        output = self.model(source)
        output = torch.unsqueeze(output, 1)
        loss = torch.nn.functional.mse_loss(output, targets) + self.alpha_smooth_reg * torch.abs(
            output[:, 0, 1:] - output[:, 0, :-1]
            ).sum() 
        #Start part for GPUs loss synchronization
        loss_over_GPUs = loss.detach().clone()
        all_reduce(loss_over_GPUs, op=ReduceOp.AVG)
        #End part for GPUs loss synchronization
        loss.backward()
        #losses.append(loss.item())
        losses.append(loss_over_GPUs.item())
        self.optimizer.step()

    def _run_validation(self):
        self.model.eval()
        self.model_mlp_day.eval()
        self.model_mlp_year.eval()
        self.model_mlp_space.eval()
        val_losses = []
        with torch.no_grad():
            for data_dict in self.val_data:
                targets = data_dict["target_variable"]
                source = self._data_train_preprocess(
                    data_dict,
                    list((self.model_mlp_day, self.model_mlp_year, self.model_mlp_space))
                )
                source = source.to(self.gpu_id)
                targets = torch.transpose(targets.unsqueeze(0), 0, 1).to(self.gpu_id)
                output = self.model(source)
                output = torch.unsqueeze(output, 1)
                loss = torch.nn.functional.mse_loss(output, targets) + self.alpha_smooth_reg * torch.abs(
                    output[:, 0, 1:] - output[:, 0, :-1]
                ).sum()
                # sync across GPUs
                loss_detached = loss.clone()
                all_reduce(loss_detached, op=ReduceOp.AVG)
                val_losses.append(loss_detached.item())
        self.model.train()
        self.model_mlp_day.train()
        self.model_mlp_year.train()
        self.model_mlp_space.train()
        return sum(val_losses) / len(val_losses)

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))["year"])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        self.val_data.sampler.set_epoch(epoch)
        #for source, targets in self.train_data:
        losses = []
        for data_dict in self.train_data:
            #print(f"[GPU {self.gpu_id}] Processing year {year}")
            targets = data_dict["target_variable"]
            source = self._data_train_preprocess(data_dict,
                                                 list((self.model_mlp_day, self.model_mlp_year, self.model_mlp_space)))
            source = source.to(self.gpu_id)
            targets = torch.transpose(targets.unsqueeze(0), 0, 1)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets, losses)
        assert len(losses) == len(self.train_data)
        #train_loss
        train_loss = sum(losses)/len(losses)
        #val_loss
        val_loss = self._run_validation()
        if self.gpu_id == 0:
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            print(
                f"[Epoch {epoch}] train loss: {train_loss:.6f} | val loss: {val_loss:.6f}"
            )
        return val_loss

    def _save_best_model(self, save_dir):
        save_dir_model = os.path.join(save_dir, "model_best")
        os.makedirs(save_dir_model, exist_ok=True)

        torch.save(get_state_dict(self.model),
                os.path.join(save_dir_model, "model_best.pt"))
        torch.save(get_state_dict(self.model_mlp_day),
                os.path.join(save_dir_model, "model_day_best.pt"))
        torch.save(get_state_dict(self.model_mlp_year),
                os.path.join(save_dir_model, "model_year_best.pt"))
        torch.save(get_state_dict(self.model_mlp_space),
                os.path.join(save_dir_model, "model_space_best.pt"))

        print(f"New best model saved (val_loss={self.best_val_loss:.6f})")

    def _save_checkpoint(self, epoch, save_dir_model):
        if not os.path.exists(save_dir_model):
            os.makedirs(save_dir_model)
        unified_path = os.path.join(save_dir_model, f"checkpoint_{epoch}.pt")
        torch.save({
            "model": get_state_dict(self.model),
            "mlp_day": get_state_dict(self.model_mlp_day),
            "mlp_year": get_state_dict(self.model_mlp_year),
            "mlp_space": get_state_dict(self.model_mlp_space),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
        }, unified_path)
        print(f"Epoch {epoch} | Training checkpoint saved at {unified_path}")

    def _remove_previous_best(self, epoch, save_dir):
        full_path = os.path.join(save_dir, "model_best", f"checkpoint_{epoch}.pt")
        if os.path.exists(full_path):
            os.remove(full_path)

    def train(self, max_epochs: int, save_dir: str):
        print(f"[GPU {self.gpu_id}] Starting training with sampler size: {len(self.train_data.sampler)}")
        for epoch in range(max_epochs + 1):
            val_loss = self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch, os.path.join(save_dir, "model"))
            # ---- BEST MODEL CHECK ----
            if self.gpu_id == 0 and val_loss < self.best_val_loss:
                # remove previous best model files (if any)
                if self.best_epoch != -1:
                    self._remove_previous_best(self.best_epoch, os.path.join(save_dir, "model_best"))
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self._save_checkpoint(epoch, os.path.join(save_dir, "model_best"))
        if self.gpu_id == 0:
            self._save_losses(save_dir)

def get_state_dict(model):
        return model.module.state_dict() if hasattr(model, "module") else model.state_dict()

def load_train_objs(variable, path_ds_train, stats_path, dp_rate, my_lr, checkpoint_path=None, ft_strategy = None):
    if not os.path.exists(stats_path):
        compute_and_save_normalization_stats(DataLoader(FloatDataset(path_ds_train, stats_path)), stats_path)
    
    model_mlp_day = MLPDay()
    model_mlp_year = MLPYear()
    model_mlp_space = MLPSpace()
    model = get_conv_model(variable=variable, dp_rate=dp_rate)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model_mlp_day.load_state_dict(ckpt["mlp_day"])
        model_mlp_year.load_state_dict(ckpt["mlp_year"])
        model_mlp_space.load_state_dict(ckpt["mlp_space"])

    if ft_strategy == "freeze_backbone":
        freeze_mlp(model_mlp_day)
        freeze_mlp(model_mlp_year)
        freeze_mlp(model_mlp_space)
        freeze_early_layers_chla(model)
        model.apply(freeze_bn)

    params = [p for p in model_mlp_day.parameters() if p.requires_grad]
    params += [p for p in model_mlp_year.parameters() if p.requires_grad]
    params += [p for p in model_mlp_space.parameters() if p.requires_grad]
    params += [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam(params=params,lr=my_lr)

    return model_mlp_day, model_mlp_year, model_mlp_space, model, optimizer



def prepare_dataloader(dataset: Dataset, batch_size: int, val = False):
    if val:
        return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def freeze_mlp(model):
    for param in model.parameters():
        param.requires_grad = False

def freeze_early_layers_chla(model):
    for name, param in model.named_parameters():
        if any(layer in name for layer in ["conv1", "bn1", "conv2", "bn2"]):
            param.requires_grad = False

def freeze_bn(module):
    if isinstance(module, torch.nn.BatchNorm1d):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

def main(
        rank: int,
        world_size: int,
        variable:str,
        save_every: int, 
        total_epochs: int, 
        batch_size: int, 
        path_ds_train: str, 
        path_ds_val: str,
        stats_path:str,
        save_dir: str,
        dp_rate: float, 
        my_lr: float, 
        alpha_smooth_reg: float,
        ft_strategy:str = None,
        region:str = None,
        checkpoint_path:str = None
    ):
    if (ft_strategy is None) != (checkpoint_path is None):
        raise ValueError("checkpoint_path and ft_strategy must either both be set or both be None")
    ddp_setup(rank, world_size)

    model_mlp_day, model_mlp_year, model_mlp_space, model, optimizer = load_train_objs(
        variable, 
        path_ds_train, 
        stats_path, 
        dp_rate, 
        my_lr, 
        checkpoint_path, 
        ft_strategy
    )
    
    train_dataset = FloatDataset(
        path_df=path_ds_train, 
        stats_path=stats_path,
        region = region
    )
    print(f"Train dataset size: {len(train_dataset)}")
    train_dataloader = prepare_dataloader(train_dataset, batch_size)
    
    val_dataset = FloatDataset(
        path_df = path_ds_val, 
        stats_path = stats_path, 
        region = region
    )
    print(f"Validation dataset size: {len(val_dataset)}")
    val_dataloader = prepare_dataloader(val_dataset, batch_size, val=True)
    trainer = Trainer(model,
                      model_mlp_day, 
                      model_mlp_year, 
                      model_mlp_space, 
                      train_dataloader,
                      val_dataloader,
                      optimizer,
                      rank, 
                      save_every,
                      alpha_smooth_reg,
                      ft= (ft_strategy is not None))
    trainer.train(total_epochs, save_dir)

    destroy_process_group()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--dp_rate', default=0.2, type=float, help='Dropout rate of CNN')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate of Adam optimizer')
    parser.add_argument('--variable', type=str, default="CHLA", choices=["NITRATE", "CHLA", "BBP700"])
    parser.add_argument('--alpha_smooth_reg', default=0.0001, type=float, help='Profiles smoothing coefficient')
    parser.add_argument('--ft_strategy', type=str, default=None,
                    choices=['aggressive', 'freeze_backbone'])
    parser.add_argument('--checkpoint_path', type=str, default=None)
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
    parser.add_argument('--split', type=str, default=None,
                        choices=['profile', 'float'])
    parser.add_argument('--ds_name', type=str, help='Name of the dataset used for training')

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print("world_size", world_size)
    path_ds_train = f"/leonardo_work/IscrC_SISCoDEM/globio/old/{args.variable}_datasets/train_{args.split}_wise_{args.ds_name}.csv"
    path_ds_val = f"/leonardo_work/IscrC_SISCoDEM/globio/old/{args.variable}_datasets/test_{args.split}_wise_{args.ds_name}.csv"
    stats_path = os.path.join(os.getcwd(), f"normalization_stats_{args.ds_name}.json")
    print("path ds train", path_ds_train)
    print("path ds val", path_ds_val)
    is_ft = args.checkpoint_path is not None
    save_dir = create_directories(["results_trial_ddp", args.variable], args.region, is_ft, args.split, date.today())
    save_ds_info_ddp(variable=args.variable,
                 flag_toy=False,
                 batch_size=args.batch_size,
                 epochs=args.total_epochs,
                 lr=args.lr,
                 dp_rate=args.dp_rate,
                 save_dir=save_dir,
                 alpha_smooth_reg=args.alpha_smooth_reg)
    start_time = time.time()
    mp.spawn(main, args=(world_size, args.variable, args.save_every, args.total_epochs, args.batch_size, path_ds_train, path_ds_val, stats_path, save_dir, args.dp_rate, args.lr, args.alpha_smooth_reg, args.ft_strategy, args.region, args.checkpoint_path), nprocs=world_size)
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"[INFO] Total training time: {elapsed_minutes:.2f} minutes", flush=True)
