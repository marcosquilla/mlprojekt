from ..data.load_data import DataModule
import pytorch_lightning as pl

if __name__ == "__main__":
    pl.seed_everything(seed=42)
    dm = DataModule(shuffle_time=True)
    dm.setup(stage='fit')
    loader = dm.train_dataloader()
    print(next(iter(loader)))
    # for batch in dm.train_dataloader():
    #     s, a, s1, r = batch
    #     print(s)