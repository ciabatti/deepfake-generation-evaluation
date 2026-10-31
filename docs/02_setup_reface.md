# Report Tecnico di Installazione e Generazione: REFace

Questo documento descrive la configurazione dell'ambiente server, la risoluzione delle incompatibilità software/hardware, la pipeline di automazione batch e le procedure di controllo qualità adottate per la generazione di immagini di *Face Swap* tramite il modello **REFace**.

---

## 1. Obiettivo e Panoramica del Modello

**REFace** è un modello basato su architettura di diffusione (*Latent Diffusion*) progettato per il **Face Swap** su immagini statiche: trasferisce l'identità biometrica di un volto sorgente (*source*) all'interno di un'immagine destinataria (*target*), preservando posa, espressione, illuminazione e sfondo del target.

L'obiettivo dell'attività è stato configurare il modello su hardware di ultima generazione e automatizzare la generazione di **100 coppie disgiunte** estratte dal dataset pubblico **FFHQ**, isolando **100 immagini valide** in formato $512 \times 512$.

---

## 2. Ambiente Hardware e Requisiti di Sistema

Le operazioni sono state eseguite su un server Linux remoto dotato di due GPU NVIDIA RTX 5090.

| Componente | Specifica / Configurazione |
| :--- | :--- |
| **GPU Utilizzata** | $1 \times$ NVIDIA RTX 5090 (architettura `sm_120`, selezionata tramite indice GPU libero) |
| **Ambiente Virtuale** | Conda ( Python `3.10.13`) |
| **PyTorch e CUDA** | PyTorch con indice `cu128` (CUDA 12.8 per supporto `sm_120`) |
| **Repository Ufficiale** | `https://github.com/Sanoojan/REFace` (pesi da Hugging Face `Sanoojan/REFace`) |
| **Spazio su Disco** | $\sim 30\text{ GB}$ (checkpoint $\sim 7\text{ GB}$, dataset FFHQ $\sim 2\text{ GB}$, ambiente Conda) |

---

## 3. Installazione e Adattamento all'Architettura RTX 5090

Lo script di installazione originale del repository (`setup.sh`) installa PyTorch `1.13.1+cu117`, incompatibile con le GPU RTX serie 50 (`sm_120`) e con i kernel Linux recenti. È stata quindi necessaria una procedura di installazione manuale e controllata:

1. **Clonazione e pulizia delle cartelle segnaposto:**
   ```bash
   git clone https://github.com/Sanoojan/REFace.git
   cd REFace
   rmdir src/taming-transformers src/clip
   ```
   *(La rimozione delle due cartelle vuote evita il blocco dell'installazione editable di `taming-transformers` e `clip`).*

2. **Creazione ambiente e installazione di PyTorch `cu128`:**
   ```bash
   conda create -n REFace python=3.10.13 -y
   conda activate REFace
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

3. **Filtraggio dei requisiti e protezione della versione di PyTorch:**
   I pacchetti non necessari all'inferenza CLI o conflittuali sono stati esclusi da `requirements.txt`, installando i moduli critici con il flag `--no-deps` per impedire il downgrade automatico di PyTorch alla versione 1.13:
   ```bash
   grep -v -i -E "^(streamlit|pudb|glfw|pyopengl|wandb|dlib|setuptools|pytorch_lightning|torchmetrics)" requirements.txt > req_filtrato.txt
   pip install -r req_filtrato.txt
   pip install --no-deps pytorch_lightning==1.4.2 torchmetrics==0.6.0
   pip install pyDeprecate==0.3.1 "fsspec[http]" future tensorboard wandb==0.18.1 flask
   ```

4. **Collegamento dei moduli locali (`taming`, `clip`, `ldm`):**
   ```bash
   pip install --no-deps -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
   pip install --no-deps -e git+https://github.com/openai/CLIP.git@main#egg=clip
   pip install --no-deps -e .
   SP=$(python -c 'import site; print(site.getsitepackages()[0])')
   echo "$PWD/src/taming-transformers" > "$SP/taming_path.pth"
   echo "$PWD" > "$SP/reface_path.pth"
   ```

5. **Compilazione di `dlib` e patch binaria di `bezier`:**
   * `dlib` è stato compilato da sorgente forzando la policy CMake:
     ```bash
     CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install dlib==19.24.2
     ```
   * Su kernel Linux recenti, la libreria precompilata `bezier` generava l'errore `cannot enable executable stack as shared object requires`. Il flag è stato rimosso dai file `.so` tramite `patchelf`:
     ```bash
     pip install patchelf
     find "$SP" -path "*bezier*" -name "*.so*" -exec patchelf --clear-execstack {} \;
     ```

6. **Download dei pesi e configurazione offline del *Safety Checker*:**
   I checkpoint (`last.ckpt` rinominato in `saved.ckpt`, ArcFace, Face Parsing e landmark DLIB) sono stati scaricati dal repository Hugging Face `Sanoojan/REFace`. Inoltre, poiché la vecchia versione di `transformers` falliva il download runtime di `CompVis/stable-diffusion-safety-checker`, il modello di sicurezza è stato scaricato localmente in `~/safety_checker` e referenziato negli script tramite `sed`.

---

## 4. Acquisizione e Preparazione del Dataset

Per il benchmark è stato utilizzato il dataset **FFHQ** nella versione `thumbnails128x128` (~70.000 volti), scaricato da Hugging Face (`nuwandaa/ffhq128`) ed estratto tramite il modulo Python `zipfile`.

Con lo script `make_pairs.py` (con `random.seed(42)` per garantire la riproducibilità), sono state selezionate casualmente 200 immagini distinte, suddividendole in **100 immagini sorgente** (`Batch100_source/pair_000.png` ... `pair_099.png`) e **100 immagini target** (`Batch100_target/pair_000.png` ... `pair_099.png`).

---

## 5. Automazione della Generazione Batch

L'utilizzo diretto dello script nativo `inference_swap_selected.py` su due cartelle da 100 immagini presenta tre criticità architetturali:
1. **Prodotto cartesiano:** lo script originale incrocia ogni sorgente con ogni target ($100 \times 100 = 10.000 \text{ immagini}$) anziché elaborare le 100 coppie 1-a-1.
2. **Bug di etichettatura nei file di output:** il file salvato come `*_inpaint.png` contiene in realtà l'immagine target mascherata con un ovale grigio, mentre il vero risultato dello scambio viene salvato senza suffisso in `<outdir>/results/<indice>/<nome>.png`.
3. **Alterazione dell'immagine sorgente:** il file `*_ref.png` salvato dal modello è la versione deformata internamente dal pre-processor, non la foto sorgente originale.

### Codice di Orchestrazione (`loop_final.sh`)

Per risolvere questi problemi è stato realizzato uno script Bash *fault-tolerant* e riprendibile che isola una coppia alla volta in directory temporanee (`/tmp`), invoca il modello con 50 step DDIM (`--scale 3.5`) e organizza l'output pulito (`source.png`, `target.png`, `swapped.png`):

```bash
#!/bin/bash
GPU=${GPU:-0}

SRC_DIR=examples/FaceSwap/Batch100_source
TGT_DIR=examples/FaceSwap/Batch100_target
WORK=examples/FaceSwap/Batch100_work
FINAL=examples/FaceSwap/Batch100_clean

mkdir -p "$FINAL"

for f in "$SRC_DIR"/*.png; do
    name=$(basename "$f" .png)

    if [ -f "$FINAL/$name/swapped.png" ]; then
        echo "Già fatto, salto $name"
        continue
    fi

    echo "=== Elaboro $name ==="

    tmp_src=/tmp/pair_src_$$
    tmp_tgt=/tmp/pair_tgt_$$
    rm -rf "$tmp_src" "$tmp_tgt" "$WORK"
    mkdir -p "$tmp_src" "$tmp_tgt"
    cp "$SRC_DIR/$name.png" "$tmp_src/"
    cp "$TGT_DIR/$name.png" "$tmp_tgt/"

    CUDA_VISIBLE_DEVICES=$GPU python scripts/inference_swap_selected.py \
        --outdir "$WORK/results" \
        --target_folder "$tmp_tgt" \
        --config models/REFace/configs/project_ffhq.yaml \
        --ckpt models/REFace/checkpoints/saved.ckpt \
        --src_folder "$tmp_src" \
        --Base_dir "$WORK/Outs" \
        --n_samples 1 \
        --scale 3.5 \
        --ddim_steps 50

    swap_file=$(find "$WORK/results/results" -name "*.png" 2>/dev/null | head -1)

    if [ -z "$swap_file" ]; then
        echo "ERRORE: nessun risultato per $name, salto"
        rm -rf "$tmp_src" "$tmp_tgt" "$WORK"
        continue
    fi

    out_dir="$FINAL/$name"
    mkdir -p "$out_dir"
    cp "$SRC_DIR/$name.png" "$out_dir/source.png"
    cp "$TGT_DIR/$name.png" "$out_dir/target.png"
    cp "$swap_file" "$out_dir/swapped.png"

    rm -rf "$tmp_src" "$tmp_tgt" "$WORK"
done

echo "FATTO: ciclo terminato"
```

---

## 6. Controllo Qualità, Pulizia del Dataset e Tempi

Lo script ha elaborato l'intero batch in background all'interno di `tmux`.

| Parametro | Valore Rilevato |
| :--- | :--- |
| **Campioni validi finali** | **100 immagini** (`swapped.png`, formato PNG) su 100 coppie previste |
| **Risoluzione Input / Output** | Input $128 \times 128$ (thumbnail FFHQ) $\rightarrow$ Output $512 \times 512$ |
| **Tempo totale di generazione** | **$\sim 4 \text{ ore e } 45 \text{ minuti}$** (su singola GPU RTX 5090) |
| **Tempo medio per coppia** | **$\sim 2 \text{ min } 50 \text{ s}$** (incluso il ricaricamento del modello ad ogni iterazione) |

*Nota qualitativa:* poiché il modello opera nativamente a $512 \times 512$ ma riceve in ingresso immagini a $128 \times 128$, le immagini generate presentano una morbidezza fisiologica dovuta all'upscaling e un'inquadratura molto ravvicinata sul volto.

---

## 7. Riepilogo Risoluzione Problemi

| Problema Riscontrato | Causa Tecnica | Soluzione Adottata |
| :--- | :--- | :--- |
| Errore `sm_120 is not compatible` | `setup.sh` installa PyTorch 1.13 + CUDA 11.7 | Installazione PyTorch con `cu128` e flag `--no-deps` sui pacchetti dipendenti |
| `ModuleNotFoundError: No module named 'taming'` / `'ldm'` | Fallimento del link simbolico nelle installazioni editable | Creazione manuale dei file `taming_path.pth` e `reface_path.pth` in `site-packages` |
| Errore `cannot enable executable stack` su `bezier` | Incompatibilità del binario precompilato con kernel Linux recenti | Rimozione del flag `execstack` tramite `patchelf --clear-execstack` |
| Errore di connessione a Hugging Face per il *Safety Checker* | Libreria `transformers` datata | Download manuale in `~/safety_checker` e sostituzione del path locale negli script |

