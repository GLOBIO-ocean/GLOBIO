"""GlobIO: prediction of biogeochemical profiles (CHLA, NITRATE, BBP700) from
temperature, salinity, oxygen, position and date with 1D convolutional networks."""
from .constants import LATITUDE_BANDS, REGIONS, VARIABLES, pressure_levels
from .data import ProfileDataset, day_rad_from_date, load_normalization_stats, profile_date
from .inference import predict, predict_mc
from .models import GlobioNet, load_checkpoint, resolve_norm_stats, save_checkpoint

__version__ = "1.0.0"

__all__ = ["GlobioNet", "LATITUDE_BANDS", "ProfileDataset", "REGIONS", "VARIABLES", "day_rad_from_date",
           "load_checkpoint", "load_normalization_stats", "predict", "predict_mc", "pressure_levels",
           "profile_date", "resolve_norm_stats", "save_checkpoint"]
