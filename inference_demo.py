# inference_demo.py
import time
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler



def stream_simulator(n=1000, anomaly_prob=0.03):
    import random
    for i in range(n):
        label = 1 if random.random() < anomaly_prob else 0
        x = np.random.normal(size=(64,))
        yield x, label


class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dim=64, latent_dim=32):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, latent_dim),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, input_dim)
        )

    def forward(self,x):
        return self.decoder(self.encoder(x))




def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()


    model = AutoEncoder(input_dim=64)
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()


    for x,label in stream_simulator():
        with torch.no_grad():
            xt = torch.from_numpy(x).float().unsqueeze(0)
            recon = model(xt).numpy()
            err = ((recon - x)**2).mean()
        if err > 0.5:
            print('[ALERT]', 'anomaly', 'score=', err)
        time.sleep(0.01)


if __name__=='__main__':
    main()