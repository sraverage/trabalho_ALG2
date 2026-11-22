
from pathlib import Path

import pandas as pd

df = pd.read_csv(Path(__file__).with_name("metrics_log.csv"))

print(df.shape)

cold = df[df['strategy'] == 'cold_start']
cold_fri26 = cold[cold['instance'] == 'fri26']
print(cold_fri26['total_pipeline_time'])