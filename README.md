# Music AI - Game Music Generator

Gerador de musica de jogo com IA. Treinamento e geracao rodando no GitHub Actions (2 vCPU, 7 GB RAM).

## Estilos suportados

- breakcore_chaotic (boss fight caotico)
- boss_fight (epico)
- 8bit (chiptune NES)
- 16bit (SNES/Genesis)
- modern (AAA cinematic)
- tense, calm, dungeon, victory, horror, cyberpunk, fantasy

## Como usar

### 1. Treinar

Actions > Train Music AI > Run workflow. O modelo LoRA treinado vai pra models/latest/.

### 2. Gerar musica

Actions > Generate Game Music > Run workflow.
Escolha: prompt, estilo, duracao, se usa soundfont.

A musica gerada vai pra pasta generated/ e tambem como artifact.

### 3. Soundfonts

Coloque seus .sf2 em soundfonts/. Veja soundfonts/README.md.

## Estrutura

- train.py, generate.py, genres.json
- soundfonts/ (seus soundfonts)
- audio/ (dados de treino)
- train.jsonl (dataset)
- .github/workflows/ (train + generate)
