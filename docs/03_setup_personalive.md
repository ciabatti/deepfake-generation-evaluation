# Report di Generazione Video con PersonaLive

Questo documento riassume i passaggi tecnici, le configurazioni e le procedure effettuate per gestire l'installazione e la generazione massiva di ritratti animati tramite il modello **PersonaLive**, sfruttando un'architettura multi-GPU per massimizzare la velocità di inferenza.

## 1. Obiettivo e Panoramica del Modello

L'obiettivo è stato quello di creare una pipeline automatizzata per generare animazioni facciali veloci e stabili utilizzando **PersonaLive**. Il modello prende un'immagine di riferimento (identità) e un video guida (movimenti) per produrre un video in cui il ritratto si muove coerentemente. A fini di confronto 1:1 con altri modelli, è stato utilizzato il medesimo dataset di partenza (asset di *LivePortrait*).

## 2. Ambiente Hardware e Requisiti di Sistema

Le operazioni sono state eseguite su un server remoto equipaggiato con due GPU di ultimissima generazione.

| Componente | Specifica / Configurazione | 
| ----- | ----- | 
| **GPU Utilizzate** | $2 \times$ NVIDIA RTX 5090 (architettura `sm_120`, 1 processo per GPU) | 
| **Ambiente Virtuale** | Conda (Python `3.10`) | 
| **PyTorch e CUDA** | PyTorch `2.7.1+cu128` e Torchvision `0.22.1+cu128` (CUDA 12.8) | 
| **Repository Ufficiale** | `https://github.com/GVCLab/PersonaLive` | 
| **Tempi di avvio** | $\sim 3\text{ minuti}$ per processo per il caricamento in VRAM dei modelli | 

## 3. Installazione e Adattamento all'Architettura RTX 5090

L'installazione standard del repository originale richiede accorgimenti specifici per superare le incompatibilità con l'architettura Blackwell (`sm_120`) delle RTX 5090 e l'assenza di alcuni repository Hugging Face.

1. **Clonazione e ambiente:**
   ```bash
   git clone https://github.com/GVCLab/PersonaLive
   cd PersonaLive
   conda create -n personalive python=3.10 -y
   conda activate personalive
   pip install -r requirements_base.txt
   ```

2. **Risoluzione compatibilità PyTorch (`sm_120`):**
   L'installazione base porta a un errore `no kernel image is available`. È stato necessario disinstallare la versione predefinita e forzare la build compatibile:
   ```bash
   pip uninstall -y torch torchvision torchaudio xformers
   pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
   ```
   *(Nota: `xformers` viene rimosso per evitare conflitti; bloccare la versione con `PIP_CONSTRAINT` per evitare downgrade accidentali).*

3. **Recupero pesi da mirror di terzi:**
   Essendo il repository originale `lambdalabs` svuotato, i componenti VAE e il modello base sono stati scaricati tramite script Python personalizzato puntando a mirror funzionanti (es. `fusing/sd-image-variations-diffusers`), applicando poi una correzione manuale alla `projection_dim` (da 512 a 768) nel file `config.json` dell'image encoder per evitare errori di *size mismatch*.

## 4. Acquisizione e Preparazione del Dataset

Per garantire un confronto perfettamente bilanciato con HunyuanPortrait, gli input sono stati prelevati dagli asset pubblici della repository di *LivePortrait* (`assets/examples/source` e `assets/examples/driving`).

* **Pulizia preliminare:** Sono state analizzate 26 immagini sorgente e 12 video guida.
* **Filtraggio:** Sono state escluse 12 immagini (soggetti animali o file corrotti) incompatibili con il rilevamento facciale richiesto dal modello.
* **Dataset Finale:** **14 immagini valide** e **12 video guida**, che genereranno tutte le combinazioni possibili ($14 \times 12 = \textbf{168 video finali}$).

## 5. Automazione della Generazione Batch

Per evitare l'esecuzione manuale di 168 comandi e ottimizzare l'uso delle 2 GPU, è stato sviluppato uno script Python (`make_cases.py`) che crea file di configurazione YAML bilanciati per partizionare il lavoro.

```python
import os, glob, sys
from omegaconf import OmegaConf

# uso: python make_cases.py [video_per_immagine] [max_immagini]
k = int(sys.argv[1]) if len(sys.argv) > 1 else 1
max_imgs = int(sys.argv[2]) if len(sys.argv) > 2 else None
N_GPU = 2          # Partizionamento su 2 GPU

def find(folder, exts):
    return sorted(p for p in glob.glob(os.path.join(os.path.abspath(folder), "*"))
                  if p.lower().endswith(exts))

imgs = find("inputs/images", (".png", ".jpg", ".jpeg", ".webp"))[:max_imgs]
vids = find("inputs/videos", (".mp4", ".mov", ".avi", ".mkv"))

if k > len(vids):
    sys.exit(f"Errore: chiedi {k} video per immagine ma ce ne sono solo {len(vids)}")

# Assegnazione circolare deterministica
cases = {img: [vids[(i + j) % len(vids)] for j in range(k)] for i, img in enumerate(imgs)}
keys = list(cases.keys())

for g in range(N_GPU):
    cfg = OmegaConf.load("configs/prompts/personalive_offline.yaml")
    cfg["test_cases"] = {x: cases[x] for x in keys[g::N_GPU]}
    out = f"configs/prompts/batch_gpu{g}.yaml"
    OmegaConf.save(cfg, out)
    print(f"GPU{g}: {sum(len(v) for v in cfg['test_cases'].values())} video -> {out}")
```

L'orchestrazione è stata lanciata dividendo il carico equamente in sessioni background tramite `tmux`:
```bash
CUDA_VISIBLE_DEVICES=0 nohup python inference_offline.py --config configs/prompts/batch_gpu0.yaml --name gpu0 -L 100 > logs/gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python inference_offline.py --config configs/prompts/batch_gpu1.yaml --name gpu1 -L 100 > logs/gpu1.log 2>&1 &
```

## 6. Controllo Qualità e Tempi di Elaborazione

* **Dataset finale:** 168 video completati con successo (cartella `split_vid`), risoluzione fissa $512 \times 512$, codec H.264.
* **Tempi di esecuzione:** Grazie alla pipeline *in-memory* che non ricarica il modello a ogni ciclo e al parallelismo su 2 GPU, l'intero batch da 168 video richiede un tempo di **circa 15-20 minuti** (circa $10\text{--}13\text{ secondi}$ a video).

## 7. Riepilogo Risoluzione Problemi

| Sintomo / Errore | Causa | Soluzione |
| :--- | :--- | :--- |
| `sm_120 is not compatible` | PyTorch obsoleto per RTX 5090 | Reinstallare con `torch==2.7.1+cu128` |
| Errore `404` su download pesi | Repo HF `lambdalabs` svuotato | Usare mirror (es. `fusing`) tramite script dedicato |
| `size mismatch for visual_projection` | Dimensione errata nel mirror | Cambiare `projection_dim` a 768 in `config.json` |
| `--use_xformers False` ignorato | Argomento tipizzato come booleano | Disinstallare il pacchetto e omettere il flag |
| Un processo si ferma | Volto non rilevato / input anomalo | Identificare dal log, rimuovere il file e rilanciare |
