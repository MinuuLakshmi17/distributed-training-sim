import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtrain_sim import LinearModel, synthetic_data, train

if __name__ == '__main__':
    data=synthetic_data(120)
    for strategy in ('ring','parameter-server'):
        result=train(data,LinearModel([0.0],0.0),steps=30,learning_rate=0.08,
                     workers=4,strategy=strategy,bucket_size=2 if strategy=='ring' else None)
        print(result.summary())
