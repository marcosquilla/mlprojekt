from ..data.load_data import DataModule
from sklearn.linear_model import SGDRegressor
import pytorch_lightning as pl


if __name__ == "__main__":
    pl.seed_everything(seed=42)
    dm = DataModule(shuffle_time=True)
    linear = SGDRegressor(shuffle=False)
    dm.setup(stage='fit')
    loader = dm.train_dataloader()
    print(next(iter(loader)))
    # for batch in dm.train_dataloader():
    #     s, a, s1, r = batch
    #     print(s)