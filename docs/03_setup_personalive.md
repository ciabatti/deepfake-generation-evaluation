# Report Tecnico di Installazione e Generazione: PersonaLive

Questo documento descrive la configurazione dell'ambiente server, la risoluzione delle incompatibilità hardware e dei pesi del modello, la pipeline di parallelizzazione Multi-GPU e i tempi di esecuzione per la generazione massiva di video tramite **PersonaLive**.

---

## 1. Obiettivo e Panoramica del Modello

**PersonaLive** è un modello di *Face Reenactment* basato su diffusione che anima un'immagine statica di riferimento (*reference image*, che definisce l'identità) seguendo le espressioni e i movimenti della testa di un video guida (*driving video*).

L'obiettivo dell'attività è stato progettare una pipeline ad alta efficienza capace di sfruttare **due GPU RTX 5090 in parallelo**, mantenendo i pesi del modello caricati in VRAM per generare un batch di **100 video** (da 20 immagini del dataset **LFW** combinate con 5 video guida ciascuna) in circa **11 minuti**.

---

## 2. Ambiente Hardware e Requisiti di Sistema

Le operazioni sono state eseguite su un server Linux remoto equipaggiato con doppia GPU NVIDIA RTX 5090.

| Componente | Specifica / Configurazione |
| :--- | :--- |
| **GPU Utilizzate** | $2 \times$ NVIDIA RTX 5090 (architettura `sm_120`, 1 processo per GPU) |
| **Ambiente Virtuale** | Conda (`personalive`, Python `3.10`) |
| **PyTorch e CUDA** | PyTorch `2.7.1+cu128` e Torchvision `0.22.1+cu128` (CUDA 12.8) |
| **Repository Ufficiale** | `https://github.com/GVCLab/PersonaLive` |
| **Spazio su Disco** | $\sim 25\text{--}30\text{ GB}$ (pesi del modello $\sim 15\text{ GB}$, ambiente Python e output video) |

---

## 3. Installazione e Adattamento all'Architettura RTX 5090

L'installazione standard del repository (`requirements_base.txt`) configura PyTorch `2.1` con CUDA `12.1`, la cui lista di architetture supportate si ferma a `sm_90`, risultando incompatibile con le RTX 5090 (`sm_120`).

1. **Clonazione e installazione base:**
   ```bash
   git clone https://github.com/GVCLab/PersonaLive
   cd PersonaLive
   conda create -n personalive python=3.10 -y
   conda activate personalive
   pip install -r requirements_base.txt
   ```

2. **Aggiornamento a PyTorch `2.7.1+cu128` e rimozione di `xformers`:**
   ```bash
   pip uninstall -y torch torchvision torchaudio
   pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
   pip uninstall -y xformers
   ```
   *(Nota: `xformers` viene rimosso per evitare conflitti di compilazione binaria; il modello esegue correttamente il fallback nativo di PyTorch. Non va inoltre passato il flag `--use_xformers False` da riga di comando poiché nel codice è tipizzato come `type=bool`, valutando qualsiasi stringa non vuota come `True`).*

3. **Blocco della versione di PyTorch tramite `PIP_CONSTRAINT`:**
   Per impedire che successive installazioni riportino PyTorch alla versione 2.1:
   ```bash
   printf "torch==2.7.1+cu128\ntorchvision==0.22.1+cu128\n" > ~/torch-constraints.txt
   export PIP_CONSTRAINT=~/torch-constraints.txt
   ```

4. **Recupero dei pesi da Mirror e correzione di `projection_dim`:**
   Lo script ufficiale `tools/download_weights.py` restituiva un errore `404 Not Found` a causa dello svuotamento del repository `lambdalabs/sd-image-variations-diffusers` su Hugging Face. I componenti sono stati quindi scaricati manualmente combinando i repository ufficiali con il mirror `fusing/sd-image-variations-diffusers`:
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download(repo_id="huaichang/PersonaLive", local_dir=".", allow_patterns=["pretrained_weights/personalive/*"])
   snapshot_download(repo_id="stabilityai/sd-vae-ft-mse", local_dir="pretrained_weights/sd-vae-ft-mse", allow_patterns=["config.json", "diffusion_pytorch_model.bin"])
   snapshot_download(repo_id="fusing/sd-image-variations-diffusers", local_dir="pretrained_weights/sd-image-variations-diffusers", allow_patterns=["model_index.json", "image_encoder/*", "unet/*"])
   ```
   Poiché il mirror presentava un valore errato nel file di configurazione dell'encoder visivo (causando l'errore `size mismatch for visual_projection.weight`: atteso $768$, trovato $512$), il parametro `projection_dim` è stato corretto programmaticamente:
   ```bash
   cd pretrained_weights/sd-image-variations-diffusers/image_encoder
   cp config.json config.json.bak
   python -c "import json; c=json.load(open('config.json')); c['projection_dim']=768; json.dump(c, open('config.json','w'), indent=2)"
   cd ../../..
   ```

---

## 4. Acquisizione e Preparazione del Dataset

Il dataset di input è stato strutturato in `inputs/images/` e `inputs/videos/`:
* **Immagini di riferimento (20 ritratti):** estratti dal dataset pubblico **LFW** (*Labeled Faces in the Wild*, Kaggle `atulanandjha/lfwpeople`).
* **Video guida (10 video):** video con volti in primo piano in formato `.mp4`.
* **Schema di abbinamento:** ogni immagine viene animata da **5 video guida consecutivi assegnati a rotazione**, generando esattamente $20 \times 5 = 100 \text{ combinazioni}$.

---

## 5. Automazione della Generazione Batch (Multi-GPU)

A differenza di approcci che invocano l'interprete Python ad ogni singola coppia, lo script `inference_offline.py` di PersonaLive accetta un file di configurazione YAML contenente un dizionario `{immagine: [lista_video]}`, caricando i $\sim 15\text{ GB}$ di modelli in VRAM **una sola volta all'avvio**.

È stato quindi sviluppato lo script `make_cases.py` per partizionare equamente il carico di lavoro su due file di configurazione (`batch_gpu0.yaml` e `batch_gpu1.yaml`, 50 video ciascuno) ed eseguire due processi paralleli sulle due RTX 5090.

### Codice di Partizionamento (`make_cases.py`)

```python
import os, glob, sys
from omegaconf import OmegaConf

# Uso: python make_cases.py [video_per_immagine] [max_immagini]
k = int(sys.argv[1]) if len(sys.argv) > 1 else 1
max_imgs = int(sys.argv[2]) if len(sys.argv) > 2 else None
N_GPU = 2

def find(folder, exts):
    return sorted(p for p in glob.glob(os.path.join(os.path.abspath(folder), "*"))
                  if p.lower().endswith(exts))

imgs = find("inputs/images", (".png", ".jpg", ".jpeg", ".webp"))[:max_imgs]
vids = find("inputs/videos", (".mp4", ".mov", ".avi", ".mkv"))
if k > len(vids):
    sys.exit(f"Errore: chiedi {k} video per immagine ma ce ne sono solo {len(vids)}")

# Ogni immagine riceve k video consecutivi a rotazione
cases = {img: [vids[(i + j) % len(vids)] for j in range(k)] for i, img in enumerate(imgs)}
tot = sum(len(v) for v in cases.values())
print(f"{len(imgs)} immagini, {len(vids)} video guida, {k} per immagine -> {tot} video totali")

keys = list(cases.keys())
for g in range(N_GPU):
    cfg = OmegaConf.load("configs/prompts/personalive_offline.yaml")
    cfg["test_cases"] = {x: cases[x] for x in keys[g::N_GPU]}
    out = f"configs/prompts/batch_gpu{g}.yaml"
    OmegaConf.save(cfg, out)
    print(f"GPU{g}: {sum(len(v) for v in cfg['test_cases'].values())} video -> {out}")
```

### Comandi di Esecuzione Parallela

All'interno di una sessione `tmux`, il batch da 100 video (con limite massimo di 100 frame per video, parametro `-L 100`) viene generato e lanciato con:

```bash
python make_cases.py 5 20

CUDA_VISIBLE_DEVICES=0 nohup python inference_offline.py --config configs/prompts/batch_gpu0.yaml --name gpu0 -L 100 > logs/gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python inference_offline.py --config configs/prompts/batch_gpu1.yaml --name gpu1 -L 100 > logs/gpu1.log 2>&1 &
```

---

## 6. Controllo Qualità, Organizzazione Output e Tempi

Il sistema salva i risultati in due sottocartelle: `concat_vid/` (video diagnostico a 4 pannelli affiancati) e `split_vid/` (contenente esclusivamente il volto animato finale a $512 \times 512$). Tutti i **100 video** generati in `split_vid/` sono risultati integri al primo tentativo.

| Parametro | Valore Rilevato |
| :--- | :--- |
| **Campioni validi finali** | **100 video** (`split_vid/*.mp4`, 50 da `gpu0` + 50 da `gpu1`) |
| **Formato e Risoluzione** | H.264, 25 fps, risoluzione fissa $512 \times 512$ |
| **Durata e Frame** | Da 76 a 100 frame (**mediana: 100 frame**, pari a $4\text{ s}$ a 25 fps) |
| **Tempo di caricamento modelli** | $\sim 3 \text{ minuti}$ per processo (eseguito una sola volta per GPU) |
| **Tempo totale del batch (100 video)** | **$\sim 11 \text{ minuti}$** (in parallelo su $2 \times$ RTX 5090) |
| **Tempo medio per video** | **$\sim 10\text{--}13 \text{ secondi}$** |

---

## 7. Riepilogo Risoluzione Problemi

| Problema Riscontrato | Causa Tecnica | Soluzione Adottata |
| :--- | :--- | :--- |
| Errore `no kernel image is available` (`sm_120`) | PyTorch 2.1 standard supporta fino a `sm_90` | Reinstallazione con PyTorch `2.7.1+cu128` e blocco tramite `PIP_CONSTRAINT` |
| Errore `404` durante `download_weights.py` | Repository Hugging Face `lambdalabs/sd-image-variations-diffusers` svuotato | Download manuale dal mirror `fusing/sd-image-variations-diffusers` |
| Errore `size mismatch for visual_projection.weight` | Disallineamento del parametro `projection_dim` (512 vs 768) nel mirror | Patch JSON su `image_encoder/config.json` impostando `"projection_dim": 768` |
| Flag `--use_xformers False` ignorato | Argomento definito come `type=bool` in `argparse` | Disinstallazione del pacchetto `xformers` e omissione del flag da CLI |