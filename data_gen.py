# data_gen.py
import csv
import random
import argparse
from datetime import datetime, timedelta


NORMAL_USERS = [f'user{i}' for i in range(1,51)]
SCANNER_USERS = ['attacker']

# 로그인 성공, 실패 등의 정상 행위
NORMAL_ACTIONS = [
    'ssh_login_success', 'ssh_login_fail', 'sudo_success', 'sudo_fail',
    'file_open', 'file_write', 'cron_job', 'service_restart'
]

# 공격 관련 의심 행위
ATTACK_ACTIONS = [
    'ssh_bruteforce', 'port_scan', 'suspicious_exec', 'payload_download'
]

# 지수 분포 기반 시간 생성기 (AI가 주석을 달아주네요...)
def gen_time_series(n, start=None):
    if start is None:
        start = datetime.now()
    t = start
    for _ in range(n):
        t += timedelta(seconds=random.expovariate(1/30))
        yield t.isoformat()



def gen_logs(n=10000, anomaly_ratio=0.02):
    times = list(gen_time_series(n))
    rows = []
    for i,t in enumerate(times):
        if random.random() < anomaly_ratio:
            user = random.choice(SCANNER_USERS)
            action = random.choice(ATTACK_ACTIONS)
            label = 1
        else:
            user = random.choice(NORMAL_USERS)
            action = random.choice(NORMAL_ACTIONS)
            label = 0
        src_ip = f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
        dst_port = random.choice([22,80,443,25,3306,8080])
        rows.append([t, user, src_ip, dst_port, action, label])
    return rows



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='logs.csv')
    parser.add_argument('--n', type=int, default=10000)
    args = parser.parse_args()
    rows = gen_logs(args.n)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestamp','user','src_ip','dst_port','action','label'])
        w.writerows(rows)
    print('written', args.out)


if __name__=='__main__':
    main()