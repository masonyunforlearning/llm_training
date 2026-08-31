import torch
from models.model import build_model

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=build_model("50M").to(device)
x=torch.randint(0,50257,(2,128),device=device)
y=torch.randint(0,50257,(2,128),device=device)
logits,loss=model(x,y)
loss.backward()
print("PASS",device,tuple(logits.shape),float(loss))
