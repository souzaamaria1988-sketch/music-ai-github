#!/usr/bin/env python3
import logging, subprocess, argparse
from pathlib import Path
from datetime import datetime
import torch, numpy as np, soundfile as sf
from transformers import MusicgenForConditionalGeneration, AutoProcessor
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

class Cfg:
    checkpoint: str = "checkpoints/final"
    output_dir: str = "outputs"
    sample_rate: int = 32000
    duration_s: float = 15.0
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.95
    cfg_coef: float = 3.0
    seed: int = None

def load(cfg):
    ck = Path(cfg.checkpoint)
    if not ck.exists(): raise FileNotFoundError(ck)
    is_lora = (ck / "adapter_config.json").exists()
    br = ck / "base_model.txt"
    if is_lora:
        bn = br.read_text().strip() if br.exists() else "facebook/musicgen-small"
        log.info(f"Base: {bn} + LoRA")
        proc = AutoProcessor.from_pretrained(ck)
        m = MusicgenForConditionalGeneration.from_pretrained(bn, torch_dtype=torch.float32)
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, str(ck))
    else:
        log.info(f"Modelo: {ck}")
        proc = AutoProcessor.from_pretrained(ck)
        m = MusicgenForConditionalGeneration.from_pretrained(ck, torch_dtype=torch.float32)
    m.eval(); return m, proc

def gen(m, proc, prompt, cfg):
    log.info(f"Gerando: {prompt}")
    if cfg.seed is not None: torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    inp = proc(text=[prompt], padding=True, return_tensors="pt")
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=int(cfg.duration_s*50), do_sample=True, temperature=cfg.temperature, top_k=cfg.top_k, top_p=cfg.top_p, guidance_scale=cfg.cfg_coef)
    return out[0,0].cpu().numpy()

def mp3(w):
    try:
        subprocess.run(["ffmpeg","-y","-i",str(w),"-codec:a","libmp3lame","-qscale:a","2",str(w.with_suffix(".mp3"))], check=True, capture_output=True)
        return w.with_suffix(".mp3")
    except Exception: return None

def run(cfg, prompt):
    m, proc = load(cfg)
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = "".join(c if c.isalnum() else "_" for c in prompt[:40]).strip("_")
    base = f"{ts}_{slug}"
    a = gen(m, proc, prompt, cfg)
    peak = np.max(np.abs(a))
    if peak > 0: a = a / peak * 0.95
    wp = out / f"{base}.wav"
    sf.write(str(wp), a, cfg.sample_rate)
    log.info(f"WAV: {wp}")
    mp = mp3(wp)
    if mp: log.info(f"MP3: {mp}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/final")
    ap.add_argument("--prompt", default="epic orchestral battle music")
    ap.add_argument("--duration", type=float, default=15.0)
    a = ap.parse_args()
    cfg = Cfg(); cfg.checkpoint = a.checkpoint; cfg.duration_s = a.duration
    run(cfg, a.prompt)
