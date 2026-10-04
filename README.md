# Music AI - Treinamento no GitHub Actions

Fine-tuning do MusicGen com LoRA no GitHub Actions (2 vCPU, 7 GB RAM, 14 GB SSD).

## Como usar
1. Coloque seus WAVs em `audio/` e edite `train.jsonl`
2. Va em **Actions** > **Train Music AI** > **Run workflow**
3. Baixe o artefato `trained-model`

## Formato do train.jsonl
Cada linha: `{"text": "descricao", "audio": "audio/arquivo.wav"}`

## Treino local
```
pip install -r requirements.txt
python train.py --epochs 50
```

## Gerar musica
```
python generate.py --prompt "sua descricao"
```
