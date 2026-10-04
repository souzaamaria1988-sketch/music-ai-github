#!/usr/bin/env python3
import json,logging,argparse
from pathlib import Path
from dataclasses import dataclass
import torch,numpy as np,soundfile as sf
from scipy.signal import resample_poly
from torch.utils.data import Dataset,DataLoader
from transformers import MusicgenForConditionalGeneration,AutoProcessor
logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")
log=logging.getLogger(__name__)

@dataclass
class Cfg:
    dataset_dir:str="."
    jsonl_name:str="train.jsonl"
    output_dir:str="checkpoints"
    base_model:str="facebook/musicgen-small"
    sample_rate:int=32000
    max_audio_seconds:float=15.0
    val_split:float=0.1
    seed:int=42
    epochs:int=50
    batch_size:int=1
    grad_accum:int=8
    learning_rate:float=1e-4
    early_stopping_patience:int=6
    use_lora:bool=True
    lora_r:int=16
    lora_alpha:int=32
    freeze_audio_encoder:bool=True
    gradient_checkpointing:bool=True

def resample(a,sr0,sr1):
    if sr0==sr1:return a
    from math import gcd
    g=gcd(sr0,sr1)
    return resample_poly(a,sr1//g,sr0//g).astype(np.float32)

class DS(Dataset):
    def __init__(self,p,proc,cfg):
        self.s,self.proc,self.cfg=[],proc,cfg
        for line in open(p,encoding="utf-8"):
            line=line.strip()
            if not line:continue
            it=json.loads(line)
            t=it.get("text") or it.get("caption") or it.get("prompt")
            ar=it.get("audio") or it.get("audio_path")
            if not t or not ar:continue
            ap=(p.parent/ar).resolve()
            if not ap.exists():continue
            self.s.append({"t":t,"a":str(ap),"g":it.get("genre",""),"c":it.get("category",""),"b":it.get("bpm",0),"m":it.get("mood","")})
        log.info(f"Dataset: {len(self.s)}")
    def __len__(self):return len(self.s)
    def __getitem__(self,i):
        s=self.s[i]
        try:
            a,sr=sf.read(s["a"],dtype="float32",always_2d=False)
            if a.ndim>1:a=a.mean(axis=1)
            if sr!=self.cfg.sample_rate:a=resample(a,sr,self.cfg.sample_rate)
            ml=int(self.cfg.sample_rate*self.cfg.max_audio_seconds)
            if len(a)>ml:a=a[:ml]
            if len(a)<self.cfg.sample_rate:a=np.pad(a,(0,self.cfg.sample_rate-len(a)))
            parts=[s["t"]]
            if s["g"]:parts.append(f"Genre: {s['g']}.")
            if s["c"]:parts.append(f"Category: {s['c']}.")
            if s["b"]:parts.append(f"Tempo: {s['b']} BPM.")
            if s["m"]:parts.append(f"Mood: {s['m']}.")
            inp=self.proc(text=[" ".join(parts)],audio=a,sampling_rate=self.cfg.sample_rate,return_tensors="pt",padding=True)
            lbl=inp["input_values"].squeeze(0) if "input_values" in inp else inp["labels"].squeeze(0)
            return {"input_ids":inp["input_ids"].squeeze(0),"attention_mask":inp["attention_mask"].squeeze(0),"labels":lbl}
        except Exception as e:
            log.warning(f"Erro: {e}")
            return {"input_ids":torch.zeros(8,dtype=torch.long),"attention_mask":torch.zeros(8,dtype=torch.long),"labels":torch.zeros(self.cfg.sample_rate)}

def collate(b):
    mt=max(x["input_ids"].shape[0] for x in b)
    ma=max(x["labels"].shape[0] for x in b)
    ids,am,lb=[],[],[]
    for x in b:
        ids.append(torch.nn.functional.pad(x["input_ids"],(0,mt-x["input_ids"].shape[0])))
        am.append(torch.nn.functional.pad(x["attention_mask"],(0,mt-x["attention_mask"].shape[0])))
        lb.append(torch.nn.functional.pad(x["labels"],(0,ma-x["labels"].shape[0])))
    return {"input_ids":torch.stack(ids),"attention_mask":torch.stack(am),"labels":torch.stack(lb)}

def build(cfg):
    log.info(f"Carregando {cfg.base_model}")
    proc=AutoProcessor.from_pretrained(cfg.base_model)
    m=MusicgenForConditionalGeneration.from_pretrained(cfg.base_model,torch_dtype=torch.float32)
    if cfg.freeze_audio_encoder:
        for p in m.audio_encoder.parameters():p.requires_grad=False
    if cfg.use_lora:
        from peft import LoraConfig,get_peft_model,TaskType
        lc=LoraConfig(r=cfg.lora_r,lora_alpha=cfg.lora_alpha,lora_dropout=0.05,bias="none",target_modules=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"],task_type=TaskType.SEQ_2_SEQ_LM)
        m=get_peft_model(m,lc)
        m.print_trainable_parameters()
    if cfg.gradient_checkpointing:
        try:m.gradient_checkpointing_enable()
        except Exception:pass
    return m,proc

def save(m,proc,cfg,out,tag="final"):
    p=out/tag;p.mkdir(parents=True,exist_ok=True)
    m.save_pretrained(p);proc.save_pretrained(p)
    if cfg.use_lora:(p/"base_model.txt").write_text(cfg.base_model)
    log.info(f"Salvo: {p}")

def train(cfg):
    torch.manual_seed(cfg.seed);np.random.seed(cfg.seed)
    m,proc=build(cfg)
    p=Path(cfg.dataset_dir)/cfg.jsonl_name
    ds=DS(p,proc,cfg)
    vs=max(1,int(len(ds)*cfg.val_split))
    tr,va=torch.utils.data.random_split(ds,[len(ds)-vs,vs],generator=torch.Generator().manual_seed(cfg.seed))
    tl=DataLoader(tr,batch_size=cfg.batch_size,shuffle=True,num_workers=0,collate_fn=collate)
    vl=DataLoader(va,batch_size=cfg.batch_size,shuffle=False,num_workers=0,collate_fn=collate)
    opt=torch.optim.AdamW((x for x in m.parameters() if x.requires_grad),lr=cfg.learning_rate,weight_decay=0.01)
    out=Path(cfg.output_dir);out.mkdir(parents=True,exist_ok=True)
    best=float("inf");pat=cfg.early_stopping_patience
    for ep in range(cfg.epochs):
        m.train();tl_sum,nb=0.0,0;opt.zero_grad()
        for st,b in enumerate(tl):
            try:
                loss=m(input_ids=b["input_ids"],attention_mask=b["attention_mask"],labels=b["labels"]).loss
                if torch.isnan(loss) or torch.isinf(loss):opt.zero_grad();continue
                (loss/cfg.grad_accum).backward()
                if (st+1)%cfg.grad_accum==0:
                    torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
                    opt.step();opt.zero_grad()
                tl_sum+=loss.item();nb+=1
                if st%5==0:log.info(f"Ep {ep+1}/{cfg.epochs} step {st}/{len(tl)} loss={loss.item():.4f}")
            except Exception as e:
                log.warning(f"Erro step {st}: {e}");opt.zero_grad();continue
        m.eval();vsum,nv=0.0,0
        with torch.no_grad():
            for b in vl:
                try:
                    l=m(input_ids=b["input_ids"],attention_mask=b["attention_mask"],labels=b["labels"]).loss
                    if not (torch.isnan(l) or torch.isinf(l)):vsum+=l.item();nv+=1
                except Exception:continue
        tv=tl_sum/max(1,nb);vv=vsum/max(1,nv)
        log.info(f"Ep {ep+1}/{cfg.epochs} train={tv:.4f} val={vv:.4f}")
        if vv<best:
            best=vv;pat=cfg.early_stopping_patience
            save(m,proc,cfg,out,"final")
            log.info(f"Novo melhor: {vv:.4f}")
        else:
            pat-=1
            if pat<=0:log.info("Early stopping");break
    log.info(f"Fim. best={best:.4f}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",default=".")
    ap.add_argument("--jsonl",default="train.jsonl")
    ap.add_argument("--output",default="checkpoints")
    ap.add_argument("--model",default="facebook/musicgen-small")
    ap.add_argument("--epochs",type=int,default=50)
    ap.add_argument("--batch",type=int,default=1)
    ap.add_argument("--grad-accum",type=int,default=8)
    ap.add_argument("--lr",type=float,default=1e-4)
    ap.add_argument("--max-seconds",type=float,default=15.0)
    a=ap.parse_args()
    train(Cfg(dataset_dir=a.dataset,jsonl_name=a.jsonl,output_dir=a.output,base_model=a.model,epochs=a.epochs,batch_size=a.batch,grad_accum=a.grad_accum,learning_rate=a.lr,max_audio_seconds=a.max_seconds))
