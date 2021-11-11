from pathlib import Path
from datetime import timedelta
import pandas as pd
import torch
from tqdm import tqdm
from src.data.datasets import CarDataset_s1
from src.data.datamodules import CarDataModule_s1

def test_data():
    dm = CarDataModule_s1(shuffle=False, batch_size=128)
    dm.setup('fit')
    dl = enumerate(dm.train_dataloader())
    for i in tqdm(range(100)):
        next(dl)

if __name__=='__main__':
    test_data()

# python -m cProfile -o reports/profiles/dataloader.prof -m tests.test_data
# snakeviz reports/profiles/dataloader.prof  