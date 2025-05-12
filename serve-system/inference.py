from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {
  "lightgcn": torch.jit.load("lightgcn_opt.pt").to(device).eval(),
  "din":      torch.jit.load("din_opt.pt").to(device).eval(),
  "ranknet":  torch.jit.load("ranknet_opt.pt").to(device).eval(),
}

class SingleReq(BaseModel):
    model: str
    data: dict

class BatchReq(BaseModel):
    model: str
    batch: list[dict]

@app.post("/infer/")
async def infer(req: SingleReq):
    if req.model not in models: raise HTTPException(404, "Model not found")
    m, d = models[req.model], req.data
    inp = parse_input(req.model, d, device)
    with torch.no_grad(): out = m(*inp) if isinstance(inp, tuple) else m(inp)
    return {"result": out.cpu().tolist()}

@app.post("/infer_batch/")
async def infer_batch(req: BatchReq):
    if req.model not in models: raise HTTPException(404, "Model not found")
    m, batch = models[req.model], req.batch
    # 构造 batch 张量（示例以 DIN 为例）
    if req.model == "lightgcn":
        inp = torch.tensor(batch[0]["edge_index"], dtype=torch.long, device=device)
    elif req.model == "din":
        cand = torch.stack([torch.tensor(d["candidate"], device=device) for d in batch])
        hist = torch.stack([torch.tensor(d["history"],   device=device) for d in batch])
        leng = torch.tensor([d["length"] for d in batch], device=device)
        inp = (cand, hist, leng)
    else:  # ranknet
        u = torch.stack([torch.tensor(d["user"],     device=device) for d in batch])
        p = torch.stack([torch.tensor(d["pos_item"], device=device) for d in batch])
        n = torch.stack([torch.tensor(d["neg_item"], device=device) for d in batch])
        inp = (u, p, n)
    with torch.no_grad(): out = m(*inp) if isinstance(inp, tuple) else m(inp)
    return {"results": out.cpu().tolist()}

def parse_input(model, data, device):
    if model=="lightgcn":
        return torch.tensor(data["edge_index"], dtype=torch.long, device=device)
    if model=="din":
        return (
          torch.tensor(data["candidate"], dtype=torch.float32, device=device),
          torch.tensor(data["history"],   dtype=torch.float32, device=device),
          torch.tensor(data["length"],    dtype=torch.long,    device=device)
        )
    # ranknet
    return (
      torch.tensor(data["user"],     dtype=torch.float32, device=device),
      torch.tensor(data["pos_item"], dtype=torch.float32, device=device),
      torch.tensor(data["neg_item"], dtype=torch.float32, device=device)
    )
