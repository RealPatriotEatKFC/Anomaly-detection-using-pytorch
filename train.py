# train.py

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# 78번째 행 checkpoints 폴더가 없으면 만듦
os.makedirs("checkpoints", exist_ok=True)

#  autoencoder 모델 정의
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
)

    def forward(self,x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out



# 1 epoch 학습 루프
def train_loop(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    for x,_ in loader:
        x = x.to(device).float()
        opt.zero_grad()
        recon = model(x)
        loss = ((recon - x)**2).mean()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)



# 메인 함수
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--labels', required=False)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=256)
    args = parser.parse_args()


    X = np.load(args.features)
    device = torch.device('cpu')
    dataset = TensorDataset(torch.from_numpy(X), torch.zeros(len(X)))
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)


    model = AutoEncoder(input_dim=X.shape[1], latent_dim=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)


    for epoch in range(args.epochs):
        loss = train_loop(model, loader, opt, device)
        print(f"Epoch {epoch+1}/{args.epochs} loss={loss:.6f}")


    torch.save(model.state_dict(), 'checkpoints/ae.pth')
    print('saved checkpoints/ae.pth')


if __name__=='__main__':
    main()