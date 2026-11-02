# Valutazione Quantitativa di HunyuanPortrait, PersonaLive e REFace

## 1. Obiettivo e Setup dei Dati

L'obiettivo è confrontare oggettivamente la qualità visiva e la coerenza temporale dei contenuti generati da tre modelli. **Cruciale in questa analisi è il confronto 1:1 tra HunyuanPortrait e PersonaLive**, eseguiti esattamente sullo stesso dataset di partenza (asset di *LivePortrait*), eliminando ogni bias demografico e di formato.

| Parametro | HunyuanPortrait | PersonaLive | REFace |
| :--- | :--- | :--- | :--- |
| **Campioni analizzati** | 120 video | 100 video | 99 immagini |
| **Dati di partenza** | Asset pubblici *LivePortrait* | Asset pubblici *LivePortrait* | Coppie casuali *FFHQ* 128×128 |
| **Formato Output** | H.264, 78–587 frame, risoluzioni variabili | H.264, max 100 frame (4s a 25fps), 512×512 | PNG 512×512 |
| **Età stimata (mediana)** | 23,5 anni | 23,5 anni | 28 anni |
| **Demografia (donne)** | ~80% | ~80% | 57% |

*Nota metodologica:* Età e sesso sono stime generate automaticamente da InsightFace. Avendo allineato i dataset video, HunyuanPortrait e PersonaLive operano sulla medesima distribuzione.

## 2. Metodologia di Valutazione

* **Qualità percettiva (NR-IQA):** Sei metriche senza riferimento calcolate tramite `pyiqa` sul ritaglio ravvicinato (crop) dei volti (32 frame equidistanti per video). Metriche: `topiq_nr-face`, `musiq`, `clipiqa`, `maniqa` ($\uparrow$); `brisque`, `niqe` ($\downarrow$).
* **Coerenza temporale:** Stabilità dell'identità (`CSIM_drift` tramite ArcFace), sfarfallio percettivo (distanza LPIPS tra frame consecutivi) e stabilità geometrica (Jitter sui landmark 3D). I modelli sono stati confrontati tramite test statistico di Mann-Whitney.

## 3. Risultati: Qualità Percettiva

A parità di dataset di input, l'architettura basata su diffusione di PersonaLive dimostra una qualità visiva nettamente superiore sul singolo frame.

| Metrica | HunyuanPortrait | PersonaLive | REFace (su subset 40%) | p (Hunyuan vs PersonaLive) |
| :--- | :--- | :--- | :--- | :--- |
| **topiq_nr-face** $\uparrow$ | 0,487 ± 0,161 | 0,515 ± 0,102 | 0,531 ± 0,113 | 0,0001058 |
| **clipiqa** $\uparrow$ | 0,434 ± 0,122 | 0,520 ± 0,117 | 0,356 ± 0,064 | 1.883e-07 |
| **brisque** $\downarrow$ | 38,7 ± 16,1 | 18,3 ± 8,6 | 52,1 ± 4,9 | 4.501e-24 |
| **niqe** $\downarrow$ | 6,18 ± 1,40 | 5,00 ± 1,25 | 7,30 ± 0,63 | 1.374e-10 |

PersonaLive domina su tutte le principali metriche di qualità dell'immagine (`topiq_nr-face`, `clipiqa`, `brisque`, `niqe`) con una significatività statistica estremamente marcata ($p < 0.01$). HunyuanPortrait si conferma molto variabile tra un video e l'altro. REFace sconta l'upscaling dai thumbnail 128x128. I valori IQA di REFace sono calcolati sul 40,4% delle immagini a causa dell'inquadratura estremamente ravvicinata di FFHQ.

## 4. Risultati: Coerenza Temporale

Il test 1:1 sullo stesso dataset rivela che la pretesa superiorità di PersonaLive sulla coerenza dell'identità era un bias legato all'inattività dei dataset precedenti.

| Metrica | HunyuanPortrait | PersonaLive | p |
| :--- | :--- | :--- | :--- |
| **CSIM_drift** $\uparrow$ | 0,7215 ± 0,1669 | 0,7211 ± 0,2183 | *Non Sig.* |
| **Flicker LPIPS** $\downarrow$ | 0,0058 ± 0,0093 | 0,0101 ± 0,0152 | *Sig.* |
| **CSIM_adj** $\uparrow$ | 0,9748 ± 0,0204 | 0,9651 ± 0,0334 | *Sig.* |

Sui movimenti ampi e prolungati tipici di *LivePortrait*, **entrambi i modelli mostrano un'identica perdita di identità nel tempo** (`CSIM_drift` 0.72). HunyuanPortrait mantiene però un vantaggio tecnico strutturale sulla fluidità di transizione locale, garantendo un minore sfarfallio visivo (`Flicker LPIPS` 0.0058 contro 0.0101) e una migliore aderenza tra frame contigui (`CSIM_adj`).

## 5. Costo Computazionale e Orchestrazione

| Parametro | HunyuanPortrait | PersonaLive | REFace |
| :--- | :--- | :--- | :--- |
| **Hardware** | 1 × RTX 5090 | 2 × RTX 5090 | 1 × RTX 5090 |
| **Orchestrazione** | *Process-per-job* | *In-memory Multi-GPU* | *Process-per-job* |
| **Tempo totale** | ~22 h 53 min (312 job) | ~15-20 min (168 video) | ~4 h 45 min (100 job) |
| **Tempo a elemento**| ~4 min 25 s | ~10-13 s | ~2 min 50 s |

Le tempistiche di HunyuanPortrait e REFace includono l'overhead di caricamento dei pesi in VRAM per singola iterazione, garantendo isolamento dei job ma dilatando i tempi. PersonaLive abbatte i tempi agganciando un dizionario YAML che mantiene il modello allocato durante l'intero ciclo. 

## 6. Conclusioni Scientifiche

Il test incrociato sul medesimo dataset elimina i bias demografici e produce una sentenza netta sulle architetture:
1. **Ribaltamento del Trade-Off:** PersonaLive non vince più sulla stabilità dell'identità nel tempo (che degrada in modo analogo per entrambi i modelli sui movimenti complessi), ma si afferma come il **modello di gran lunga superiore per qualità visiva del fotogramma e assenza di distorsioni spaziali**.
2. **Fluidità Temporale:** HunyuanPortrait conserva il primato sulla fluidità pura del movimento, producendo meno micro-flickering e un'interpolazione migliore tra frame adiacenti.
3. **Efficienza in Produzione:** L'architettura software di PersonaLive dimostra una chiara superiorità in contesti di generazione batch massiva.
