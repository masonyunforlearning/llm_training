import argparse, torch
from models.model import build_model
from training.utils import seed_everything,count_parameters

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="1.2B",choices=["1.2B"])
    p.add_argument("--mixture",required=True)
    p.add_argument("--train-tokens",type=int,default=100_000_000_000)
    p.add_argument("--seq-len",type=int,default=2048)
    p.add_argument("--batch-size",type=int,default=1)
    p.add_argument("--grad-accum",type=int,default=32)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--tokenizer",default="gpt2")
    p.add_argument("--output-dir",default="experiments/final_1.2B")
    a=p.parse_args()
    seed_everything(123)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=build_model(a.model).to(device)
    print("Device:",device,"Parameters:",count_parameters(model))
    print("Target tokens:",a.train_tokens,"Mixture:",a.mixture)
    if device.type!="cuda": return
    print("GPU:",torch.cuda.get_device_name(0))
    print("Model initialized. Connect streaming data before long run.")
if __name__=="__main__": main()
