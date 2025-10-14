# evaluate.py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt


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
        return self.decoder(self.encoder(x))




def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--features', required=True)
    parser.add_argument('--labels', required=True)
    args = parser.parse_args()


    X = np.load(args.features)
    y = np.load(args.labels)


    model = AutoEncoder(input_dim=X.shape[1])
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()


    X_t = torch.from_numpy(X).float()
    with torch.no_grad():
        recon = model(X_t).numpy()
    errors = ((recon - X)**2).mean(axis=1)


    auc = roc_auc_score(y, errors)
    print('AUC:', auc)


    # 간단 시각화, png파일은 evaluate.py가 있는 폴더에 저장됨
    plt.figure(figsize=(8,4))
    plt.hist(errors[y==0], bins=100, alpha=0.6, label='normal')
    plt.hist(errors[y==1], bins=100, alpha=0.6, label='anomaly')
    plt.legend()
    plt.title('Reconstruction error distribution')
    plt.savefig('error_history.png')
    print('saved error_history.png')


if __name__=='__main__':
    main()