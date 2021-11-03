from pathlib import Path
from datetime import timedelta
import pandas as pd
import torch
from tqdm import tqdm
from src.data.datasets import CarDataset
from src.data.datamodules import CarDataModule

def test_data():
    indices = pd.MultiIndex.from_frame(pd.read_csv((Path('.') / 'data' / 'processed' / 'locations.csv'), usecols=['Time', 'Vehicle_Number_Plate'], parse_dates=['Time'])).reorder_levels([1,0])
    ds = CarDataset(indices, timedelta(minutes=30))
    for i in tqdm(range(len(ds))):
        assert ds[i][1].shape == torch.Size([50])

if __name__=='__main__':
    test_data()