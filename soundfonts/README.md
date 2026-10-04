# SoundFonts

Coloque seus arquivos .sf2 ou .sf3 aqui.

O generate.py escolhe automaticamente por estilo:
- 8bit -> procura "chip" no nome
- 16bit -> procura "retro"
- orchestral -> procura "orchestral" ou "orchestra"
- metal -> procura "metal"
- piano -> procura "piano"

Se nao encontrar, usa o primeiro disponivel.

**IMPORTANTE**: GitHub tem limite de 100 MB por arquivo. Se seu .sf2 for maior, use Git LFS:

    git lfs install
    git lfs track "*.sf2"
    git add .gitattributes

Para subir seus soundfonts:

    cd /storage/emulated/0/Documents/GGUF/music-ai-github
    cp /caminho/dos/seus/*.sf2 soundfonts/
    git add soundfonts/
    git commit -m "Add soundfonts"
    git push
