import argparse, torch
from models.model import build_model
from training.utils import seed_everything,count_parameters

def parse_mixture(s):
    d={k:float(v) for k,v in (x.split(":") for x in s.split(","))}
    z=sum(d.values()); return {k:v/z for k,v in d.items()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="50M",choices=["50M","100M"])
    p.add_argument("--mixture",required=True)
    p.add_argument("--train-tokens",type=int,default=300_000_000)
    p.add_argument("--seq-len",type=int,default=2048)
    p.add_argument("--batch-size",type=int,default=1)
    p.add_argument("--grad-accum",type=int,default=32)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--tokenizer",default="gpt2")
    p.add_argument("--output-dir",default="experiments/proxy/run")
    a=p.parse_args()
    seed_everything(123)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=build_model(a.model).to(device)
    print("Device:",device)
    print("Parameters:",count_parameters(model))
    print("Mixture:",parse_mixture(a.mixture))
    print("Dataset hookup is intentionally explicit: configure Korean corpus in configs/datasets.yaml.")
    if device.type=="cuda": print("GPU:",torch.cuda.get_device_name(0))
if __name__=="__main__": main()
