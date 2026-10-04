#!/usr/bin/env python3
import json,logging,subprocess,argparse,os
from pathlib import Path
from datetime import datetime
import torch,numpy as np,soundfile as sf
from transformers import MusicgenForConditionalGeneration,AutoProcessor
logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")
log=logging.getLogger(__name__)

def load_genres():
    p=Path("genres.json")
    if p.exists():return json.loads(p.read_text(encoding="utf-8"))
    return {"game_music_styles":{}}

def build_prompt(base,style_key,genres):
    styles=genres.get("game_music_styles",{})
    st=styles.get(style_key,{})
    prefix=st.get("prompt_prefix","")
    extra=st.get("instruments","")
    parts=[x for x in [prefix,base,extra] if x]
    return ", ".join(parts)

def pick_soundfont(prompt,sf_dir):
    sf_dir=Path(sf_dir)
    if not sf_dir.exists():return None
    fonts=list(sf_dir.glob("*.sf2"))+list(sf_dir.glob("*.sf3"))
    if not fonts:return None
    pl=prompt.lower()
    keywords={
        "8bit":["chip","8-bit","nes","retro","arcade"],
        "16bit":["16-bit","snes","genesis","megadrive"],
        "orchestral":["orchestral","epic","symphony","boss"],
        "metal":["metal","distortion","heavy"],
        "piano":["piano","calm","melancholy"]
    }
    for k,kws in keywords.items():
        if any(kw in pl for kw in kws):
            for f in fonts:
                if k in f.name.lower() or (k=="8bit" and "chip" in f.name.lower()):
                    return f
    return fonts[0]

def load_model(ckpt):
    ck=Path(ckpt)
    if not ck.exists():
        log.warning(f"Checkpoint {ck} nao encontrado, usando base")
        proc=AutoProcessor.from_pretrained("facebook/musicgen-small")
        m=MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small",torch_dtype=torch.float32)
        return m,proc
    is_lora=(ck/"adapter_config.json").exists()
    br=ck/"base_model.txt"
    if is_lora:
        bn=br.read_text().strip() if br.exists() else "facebook/musicgen-small"
        log.info(f"Base: {bn} + LoRA")
        proc=AutoProcessor.from_pretrained(ck)
        m=MusicgenForConditionalGeneration.from_pretrained(bn,torch_dtype=torch.float32)
        from peft import PeftModel
        m=PeftModel.from_pretrained(m,str(ck))
    else:
        proc=AutoProcessor.from_pretrained(ck)
        m=MusicgenForConditionalGeneration.from_pretrained(ck,torch_dtype=torch.float32)
    m.eval();return m,proc

def generate(m,proc,prompt,duration,temp,seed):
    log.info(f"Gerando: {prompt}")
    if seed is not None:torch.manual_seed(seed);np.random.seed(seed)
    inp=proc(text=[prompt],padding=True,return_tensors="pt")
    with torch.no_grad():
        out=m.generate(**inp,max_new_tokens=int(duration*50),do_sample=True,temperature=temp,top_k=250,top_p=0.95,guidance_scale=3.0)
    return out[0,0].cpu().numpy()

def mp3(w):
    try:
        subprocess.run(["ffmpeg","-y","-i",str(w),"-codec:a","libmp3lame","-qscale:a","2",str(w.with_suffix(".mp3"))],check=True,capture_output=True)
        return w.with_suffix(".mp3")
    except Exception as e:
        log.warning(f"ffmpeg: {e}");return None

def try_soundfont_render(wav_in,sf2,out_dir,base_name):
    try:
        from basic_pitch.inference import predict_and_save
        midi_dir=out_dir/"midi";midi_dir.mkdir(exist_ok=True)
        predict_and_save(audio_path_list=[str(wav_in)],output_directory=str(midi_dir),save_midi=True,sonify_midi=False)
        gm=midi_dir/f"{wav_in.stem}_basic_pitch.mid"
        fm=midi_dir/f"{base_name}.mid"
        if gm.exists():gm.replace(fm)
        enh=out_dir/f"{base_name}_sf.wav"
        subprocess.run(["fluidsynth","-F",str(enh),"-T","wav","-r","32000","-ni",str(sf2),str(fm)],check=True,capture_output=True,timeout=120)
        return enh
    except Exception as e:
        log.warning(f"soundfont falhou: {e}")
        return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prompt",default="epic boss fight music")
    ap.add_argument("--style",default="16bit")
    ap.add_argument("--duration",type=float,default=15.0)
    ap.add_argument("--temperature",type=float,default=1.0)
    ap.add_argument("--seed",type=int,default=None)
    ap.add_argument("--checkpoint",default="checkpoints/final")
    ap.add_argument("--output",default="outputs")
    ap.add_argument("--soundfonts",default="soundfonts")
    ap.add_argument("--use-soundfont",action="store_true")
    a=ap.parse_args()

    genres=load_genres()
    prompt=build_prompt(a.prompt,a.style,genres)
    log.info(f"Prompt final: {prompt}")

    m,proc=load_model(a.checkpoint)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    slug="".join(c if c.isalnum() else "_" for c in a.prompt[:40]).strip("_")
    base=f"{ts}_{a.style}_{slug}"

    audio=generate(m,proc,prompt,a.duration,a.temperature,a.seed)
    peak=np.max(np.abs(audio))
    if peak>0:audio=audio/peak*0.95
    wp=out/f"{base}.wav"
    sf.write(str(wp),audio,32000)
    log.info(f"WAV: {wp}")

    if a.use_soundfont:
        sf2=pick_soundfont(prompt,a.soundfonts)
        if sf2:
            log.info(f"SoundFont: {sf2.name}")
            enh=try_soundfont_render(wp,sf2,out,base)
            if enh:
                log.info(f"Enh: {enh}")
                wp=enh
        else:
            log.info("Sem soundfont, mantendo raw")

    mp3(wp)
    log.info("Done")

if __name__=="__main__":
    main()
