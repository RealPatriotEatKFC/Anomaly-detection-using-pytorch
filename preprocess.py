# preprcess.py

import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler




def featurize(df):
    vec = CountVectorizer()
    X_action = vec.fit_transform(df['action']).toarray()


    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    X_port = df[['dst_port','hour']].values


    # user별 최근 실패 횟수 같은 거
    user_idx = pd.factorize(df['user'])[0]
    user_onehot = np.zeros((len(df), min(50, len(df['user'].unique()))))
    for i,u in enumerate(user_idx):
        if u < user_onehot.shape[1]:
            user_onehot[i,u] = 1


    X = np.hstack([X_action, X_port, user_onehot])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, scaler




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--out-prefix', required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.infile)
    X, scaler = featurize(df)
    np.save(args.out_prefix + '.npy', X)
    np.save(args.out_prefix + '_labels.npy', df['label'].values)
    print('saved', args.out_prefix)


if __name__=='__main__':
    main()