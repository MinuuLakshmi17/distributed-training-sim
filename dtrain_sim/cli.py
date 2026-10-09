from __future__ import annotations
import argparse, json
from .model import LinearModel
from .training import synthetic_data, train, load_checkpoint

def main():
    p=argparse.ArgumentParser(description="CPU-only distributed training simulator")
    sub=p.add_subparsers(dest="cmd", required=True)
    t=sub.add_parser("train", help="train the tiny linear model")
    t.add_argument("--strategy", choices=["ring","parameter-server"], default="ring")
    t.add_argument("--workers", type=int, default=4); t.add_argument("--steps", type=int, default=30)
    t.add_argument("--learning-rate", type=float, default=0.08); t.add_argument("--bucket-size", type=int)
    t.add_argument("--straggler", type=int, action="append", default=[])
    t.add_argument("--straggler-delay", type=float, default=0.0)
    t.add_argument("--fail-worker", type=int); t.add_argument("--fail-step", type=int, default=5)
    t.add_argument("--checkpoint", default="checkpoint.json")
    t.add_argument("--no-checkpoint", action="store_true")
    c=sub.add_parser("checkpoint", help="inspect a saved checkpoint")
    c.add_argument("path")
    a=p.parse_args()
    if a.cmd=="checkpoint":
        m,step,losses=load_checkpoint(a.path)
        print(json.dumps({"step":step,"parameters":m.parameters(),"losses":losses},indent=2)); return
    failures={a.fail_worker:a.fail_step} if a.fail_worker is not None else {}
    result=train(synthetic_data(), LinearModel([0.0],0.0), steps=a.steps, learning_rate=a.learning_rate,
        workers=a.workers,strategy=a.strategy,bucket_size=a.bucket_size,
        straggler_workers=set(a.straggler),straggler_delay=a.straggler_delay,
        failed_worker_steps=failures,checkpoint_path=None if a.no_checkpoint else a.checkpoint)
    print(json.dumps(result.summary(),indent=2))
if __name__=="__main__": main()
