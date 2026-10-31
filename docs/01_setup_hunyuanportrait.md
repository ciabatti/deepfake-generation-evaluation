# Report Tecnico di Installazione e Generazione: HunyuanPortrait

Questo documento descrive la configurazione dell'ambiente server, la risoluzione delle dipendenze hardware, la pipeline di automazione batch e le procedure di *data cleaning* adottate per la generazione massiva di video tramite il modello **HunyuanPortrait** (Tencent).

---

## 1. Obiettivo e Panoramica del Modello

**HunyuanPortrait** è un modello di *Face Reenactment* (animazione di ritratti) che trasferisce il movimento facciale e le espressioni da un video guida (*driving video*) a un'immagine statica di riferimento (*source image*).

L'obiettivo dell'attività è stato la realizzazione di un'architettura automatizzata basata sulla tecnica del **Cross-Driving**: incrociare sistematicamente un set di immagini sorgente con un set di video guida per generare tutte le combinazioni possibili, riducendo al minimo l'intervento manuale e isolando un dataset finale pulito di **120 video validi**.

---

## 2. Ambiente Hardware e Requisiti di Sistema

Le operazioni sono state eseguite su un server Linux remoto (`Lamarr`) equipaggiato con GPU di architettura NVIDIA Blackwell.

| Componente | Specifica / Configurazione |
| :--- | :--- |
| **GPU Utilizzata** | $1 \times$ NVIDIA RTX 5090 (architettura `sm_120`) |
| **Ambiente Virtuale** | Conda (`hunyuan_env`) |
| **PyTorch e CUDA** | PyTorch `2.14.1+cu130` (compilato con CUDA 13.0) |
| **Repository Ufficiale** | `https://github.com/Tencent-Hunyuan/HunyuanPortrait` |
| **Accesso e Sessioni** | VPN WireGuard, SSH/SCP, **Tmux** per l'esecuzione persistente in background |

---

## 3. Installazione e Adattamento all'Architettura RTX 5090

Per garantire il funzionamento sull'architettura della RTX 5090 ed evitare conflitti di libreria con gli altri modelli presenti sul server, è stato predisposto un ambiente isolato dedicato:

1. **Accesso e persistenza della sessione:**
   ```bash
   ssh <UTENTE>@<IP_SERVER>
   tmux new -s hunyuan
   ```
2. **Clonazione del repository e creazione dell'ambiente:**
   ```bash
   git clone https://github.com/Tencent-Hunyuan/HunyuanPortrait.git
   cd HunyuanPortrait
   conda create -n hunyuan_env python=3.10 -y
   conda activate hunyuan_env
   ```
3. **Configurazione PyTorch / CUDA:**
   Le versioni standard di PyTorch non supportano nativamente i kernel per le GPU RTX serie 50. È stata quindi installata la build `2.14.1+cu130` (CUDA 13.0), verificando il corretto rilevamento della GPU tramite `torch.cuda.is_available()` prima di procedere con le restanti dipendenze del repository.

---

## 4. Acquisizione e Preparazione del Dataset

Per evitare artefatti dovuti a un errato allineamento facciale negli input, il dataset di partenza è stato estratto dagli asset pubblici di test del repository **LivePortrait** (*utilizzati esclusivamente a scopo di ricerca accademica*):

1. **Download selettivo del repository sorgente:**
   ```bash
   git clone --depth 1 https://github.com/KlingAIResearch/LivePortrait.git
   ```
2. **Estrazione e standardizzazione degli input:**
   * **Immagini sorgente (26 file):** prelevate da `assets/examples/source` e copiate in `batch_inputs/images/`.
   * **Video guida (12 file):** prelevati da `assets/examples/driving` e copiati in `batch_inputs/videos/`.
   * **Rinominazione sequenziale:** eseguita tramite script Bash per uniformare i nomi dei file (`img_1.jpg` ... `img_26.jpg` e `drive_1.mp4` ... `drive_12.mp4`).

---

## 5. Automazione della Generazione Batch

Per evitare l'esecuzione manuale dei **312 comandi previsti** ($26 \text{ immagini} \times 12 \text{ video guida}$), è stato sviluppato lo script di orchestrazione `batch_generate.py`.

### Caratteristiche dello Script di Orchestrazione
* **Isolamento *process-per-job*:** lo script invoca `inference.py` tramite `subprocess.run` per ciascuna combinazione. Sebbene questo approccio comporti il ricaricamento del modello in memoria ad ogni iterazione, garantisce che un eventuale errore su un singolo file non comprometta l'intero batch.
* **Soppressione dell'output verboso:** gli stream standard della rete neurale vengono reindirizzati a ` subprocess.DEVNULL` per mantenere a terminale un log leggibile con orario, percentuale di avanzamento e coppia in lavorazione.
* **Struttura modulare dell'output:** al termine di ogni inferenza, lo script individua l'ultima cartella generata in `results/`, preleva `full_resolution.mp4` e crea un pacchetto autoconsistente in `batch_outputs/img_X_guidato_da_drive_Y/` contenente l'immagine di input, il video guida e il video renderizzato (`risultato_finale.mp4`).

### Codice Sorgente (`batch_generate.py`)

```python
import os
import glob
import shutil
import subprocess
from datetime import datetime

IMG_DIR = "batch_inputs/images"
VID_DIR = "batch_inputs/videos"
OUT_DIR = "batch_outputs"

images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))])
videos = sorted([f for f in os.listdir(VID_DIR) if f.endswith('.mp4')])

total_jobs = len(images) * len(videos)

print("=" * 60)
print(f" AVVIO BATCH GENERATION alle {datetime.now().strftime('%H:%M:%S')}")
print(f" Immagini trovate: {len(images)}")
print(f" Video guida trovati: {len(videos)}")
print(f" TOTALE VIDEO DA GENERARE: {total_jobs}")
print("=" * 60 + "\n")

counter = 1
for img_name in images:
    for vid_name in videos:
        percent = int((counter / total_jobs) * 100)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] JOB {counter}/{total_jobs} ({percent}%)")
        print(f"    -> Animazione: {img_name} + {vid_name}")
        
        img_path = os.path.join(IMG_DIR, img_name)
        vid_path = os.path.join(VID_DIR, vid_name)
        
        subprocess.run(
            ["python", "inference.py", "--image_path", img_path, "--video_path", vid_path], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.STDOUT
        )
        
        all_result_dirs = glob.glob("results/*/")
        if not all_result_dirs:
            print("    -> ERRORE CRITICO: Nessun output generato dal server.\n")
            counter += 1
            continue
            
        latest_dir = max(all_result_dirs, key=os.path.getmtime)
        generated_mp4 = os.path.join(latest_dir, "full_resolution.mp4")
        
        img_base = os.path.splitext(img_name)[0]
        vid_base = os.path.splitext(vid_name)[0]
        out_folder_path = os.path.join(OUT_DIR, f"{img_base}_guidato_da_{vid_base}")
        os.makedirs(out_folder_path, exist_ok=True)
        
        if os.path.exists(generated_mp4):
            shutil.copy(img_path, os.path.join(out_folder_path, f"input_{img_name}"))
            shutil.copy(vid_path, os.path.join(out_folder_path, f"input_{vid_name}"))
            shutil.copy(generated_mp4, os.path.join(out_folder_path, "risultato_finale.mp4"))
            print(f"    -> COMPLETATO: Salvato in {out_folder_path}\n")
        else:
            print(f"    -> ERRORE: Generazione fallita per {img_name} + {vid_name}\n")
        
        counter += 1

print("=" * 60)
print(f" BATCH CONCLUSO CON SUCCESSO alle {datetime.now().strftime('%H:%M:%S')}")
print("=" * 60)
```

---

## 6. Controllo Qualità, Pulizia del Dataset e Tempi

Una volta completata l'esecuzione delle 312 combinazioni, il dataset è stato sottoposto a un processo di filtraggio in due fasi per garantire la validità dei campioni destinati alla valutazione quantitativa:

1. **Filtraggio sul server ($312 \rightarrow 156 \text{ video}$):** sono state rimosse 13 immagini sorgente non idonee al task biometrico umano (12 raffiguranti animali e 1 file malformato che causava output corrotti).
2. **Cernita post-download in locale ($156 \rightarrow 120 \text{ video}$):** dopo il trasferimento tramite SCP, un'ispezione visiva manuale ha permesso di scartare ulteriori 36 video affetti da artefatti visivi o anomalie di rendering.

### Specifiche del Dataset Finale e Metriche di Esecuzione

| Parametro | Valore Rilevato |
| :--- | :--- |
| **Campioni validi finali** | **120 video** (`.mp4`, codec H.264) |
| **Framerate e Durata** | 25 o 30 fps (ereditato dal video guida); da 78 a 587 frame (**mediana: 301 frame**, $\sim 10\text{ s}$) |
| **Risoluzione** | Variabile in base all'input (es. $720 \times 720$, $600 \times 704$, $708 \times 648$) |
| **Finestra temporale batch** | 22 ore e 53 minuti (ricostruita dai timestamp dei file `.mp4` su filesystem) |
| **Tempo medio per job** | **$\sim 4 \text{ min } 25 \text{ s}$** (incluso il caricamento completo dei pesi ad ogni iterazione) |

---

## 7. Riepilogo Risoluzione Problemi

| Problema Riscontrato | Causa Tecnica | Soluzione Adottata |
| :--- | :--- | :--- |
| Incompatibilità architettura GPU | Kernel CUDA standard non compilati per RTX 5090 | Ambiente isolato con PyTorch `2.14.1+cu130` |
| Presenza di campioni non umani o corrotti | Asset di test *LivePortrait* contenenti anche volti animali | Rimozione a monte di 13 sorgenti e scarto manuale di 36 output anomali |
| Rischio interruzione processi lunghi via SSH | Disconnessione del client locale durante le ~23 ore di calcolo | Esecuzione incapsulata in sessione `tmux` persistente |