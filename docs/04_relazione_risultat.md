# Relazione Tecnica: Valutazione Quantitativa di HunyuanPortrait, PersonaLive e REFace

Questo documento presenta il benchmark quantitativo eseguito sui contenuti generati dai tre modelli oggetto di studio: **HunyuanPortrait** e **PersonaLive** (*Face Reenactment* / animazione di ritratti tramite video guida) e **REFace** (*Face Swap* su immagini statiche). L'analisi valuta sia la qualità percettiva spaziale sui singoli fotogrammi sia la stabilità biometrica e temporale lungo le sequenze video.

---

## 1. Obiettivo e Caratteristiche del Dataset Valutato

L'obiettivo dell'analisi è misurare in modo oggettivo e riproducibile la qualità visiva e la coerenza temporale degli output prodotti dalle tre pipeline di generazione, mettendo in luce i compromessi (*trade-off*) tra realismo del singolo fotogramma, mantenimento dell'identità e costo computazionale.

| Parametro | HunyuanPortrait | PersonaLive | REFace |
| :--- | :--- | :--- | :--- |
| **Campioni analizzati** | **120 video** | **100 video** | **99 immagini** |
| **Dataset di partenza** | Asset pubblici *LivePortrait* | Ritratti *LFW* (*Labeled Faces in the Wild*) | Coppie casuali *FFHQ* ($128 \times 128$) |
| **Formato e Codifica** | `.mp4` (H.264), 25/30 fps | `.mp4` (H.264), 25 fps | `.png` senza perdita |
| **Durata (Frame)** | Da 78 a 587 frame (**mediana: 301**, $\sim 10\text{ s}$) | Da 76 a 100 frame (**mediana: 100**, $4\text{ s}$) | Immagine statica (1 frame) |
| **Risoluzione Output** | Variabile (es. $720 \times 720$, $600 \times 704$, $708 \times 648$) | Fissa $512 \times 512$ | Fissa $512 \times 512$ |
| **Età stimata (mediana, Q1–Q3)** | 23,5 anni (22–26) | 51,0 anni (42–64) | 28,0 anni (23–39) |
| **Distribuzione di genere (Donne)** | 80% | 35% | 57% |

*Nota demografica: i valori relativi a età e genere rappresentano stime automatiche estratte dal modulo biometrico InsightFace sui campioni generati e non costituiscono dati anagrafici reali dei soggetti.*

---

## 2. Metodologia di Valutazione

L'intera pipeline di valutazione è stata eseguita su ambiente Google Colab equipaggiato con GPU **NVIDIA A100-SXM4-40GB**, strutturando l'analisi in due moduli complementari:

### 2.1 Qualità Percettiva Spaziale (NR-IQA)
Per valutare la qualità visiva in assenza di un'immagine di riferimento (*No-Reference Image Quality Assessment*), sono state impiegate **6 metriche stato dell'arte** tramite la libreria `pyiqa`:
* **Metriche orientate alla qualità estetica e percettiva ($\uparrow$ più alto è meglio):** `topiq_nr-face` (specifica per i volti), `musiq`, `clipiqa`, `maniqa`.
* **Metriche orientate alla naturalezza statistica e assenza di distorsioni ($\downarrow$ più basso è meglio):** `brisque`, `niqe`.

Per ciascun video sono stati estratti **32 frame equidistanti** (per un totale di **7.139 tra frame e immagini** elaborati). Su ogni frame, il modello **InsightFace (`buffalo_l`)** ha localizzato il volto principale, applicando un ritaglio quadrato centrato e scalato di un fattore $1,6\times$ rispetto al *bounding box* (`face_crop`). In questo modo la valutazione IQA è stata circoscritta esclusivamente alla regione facciale sintetizzata, evitando contaminazioni dovute allo sfondo. I punteggi dei singoli frame sono stati poi aggregati calcolando la media per ciascun file; il confronto statistico tra i due modelli video è stato condotto tramite il **test non parametrico di Mann-Whitney U** (a due code).

### 2.2 Coerenza Temporale e Identità Biometrica (Solo Video)
Sui 220 video complessivi (*120 HunyuanPortrait + 100 PersonaLive*) sono state calcolate metriche specifiche su due schemi di campionamento (20 frame uniformemente distribuiti sull'intero video e 5 finestre composte da 8 frame consecutivi ciascuna):
* **Tenuta dell'Identità (`CSIM_drift`, `CSIM_drift_min`, `CSIM_adj` — $\uparrow$):** *Cosine Similarity* tra i vettori di embedding biometrico **ArcFace** (512 dimensioni). `CSIM_drift` misura la somiglianza media (e minima, con `CSIM_drift_min`) tra il primo frame valido del video e i 20 frame sparsi lungo tutta la durata, quantificando la deriva d'identità (*identity drift*). `CSIM_adj` valuta la continuità identitaria tra coppie di frame immediatamente adiacenti.
* **Sfarfallio Percettivo (`Flicker LPIPS` — $\downarrow$):** distanza percettiva calcolata con rete **LPIPS (backbone AlexNet)** tra frame consecutivi all'interno delle finestre temporali, per rilevare variazioni anomale di texture o illuminazione.
* **Stabilità Geometrica dei Tratti (`Jitter` e `Rapporto Jitter/Movimento` — $\downarrow$):** norma della derivata seconda discreta ($\|l_c - 2l_b + l_a\|$) calcolata sui **68 landmark facciali 3D** normalizzati su triplette di frame consecutivi $(a, b, c)$, sia in valore assoluto sia normalizzata rispetto allo spostamento medio tra frame (`motion`).

---

## 3. Risultati: Qualità Percettiva Spaziale (NR-IQA)

La tabella seguente riporta media e deviazione standard calcolate tra i file di ciascun modello, insieme al *p-value* del test di Mann-Whitney tra i due modelli di animazione video:

| Metrica | HunyuanPortrait ($n=120$) | PersonaLive ($n=100$) | REFace ($n=99$, subset 40,4%) | *p-value* (Hunyuan vs PersonaLive) |
| :--- | :---: | :---: | :---: | :---: |
| **volto\_trovato\_%** | 100,0% | 100,0% | 40,4% | — |
| **crop\_mediano\_px** | $424 \times 424$ | $450 \times 452$ | $512 \times 512$ | — |
| **topiq\_nr-face** $\uparrow$ | $0,487 \pm 0,161$ | $0,381 \pm 0,047$ | $0,531 \pm 0,113$ | $6,06 \cdot 10^{-7}$ |
| **musiq** $\uparrow$ | $46,889 \pm 13,501$ | $43,961 \pm 7,246$ | $44,745 \pm 9,226$ | $0,7862$ |
| **clipiqa** $\uparrow$ | $0,434 \pm 0,122$ | $0,308 \pm 0,034$ | $0,356 \pm 0,064$ | $2,03 \cdot 10^{-19}$ |
| **maniqa** $\uparrow$ | $0,296 \pm 0,072$ | $0,294 \pm 0,042$ | $0,263 \pm 0,054$ | $0,2185$ |
| **brisque** $\downarrow$ | $38,671 \pm 16,104$ | $44,614 \pm 5,669$ | $52,114 \pm 4,859$ | $5,23 \cdot 10^{-6}$ |
| **niqe** $\downarrow$ | $6,181 \pm 1,399$ | $6,693 \pm 0,503$ | $7,300 \pm 0,631$ | $1,31 \cdot 10^{-6}$ |

### Analisi dei Risultati IQA
1. **Confronto tra i modelli video:** **HunyuanPortrait** ottiene risultati significativamente superiori a **PersonaLive** su quattro metriche su sei (`topiq_nr-face`, `clipiqa`, `brisque` e `niqe`, tutte con $p < 10^{-5}$), evidenziando una maggiore nitidezza e naturalezza sul singolo fotogramma. Sulle metriche `musiq` ($p = 0,79$) e `maniqa` ($p = 0,22$) non si osservano invece differenze statisticamente significative.
2. **Dispersone e uniformità:** A fronte di una media superiore, **HunyuanPortrait** mostra una variabilità inter-video molto marcata (deviazioni standard elevate su tutte le metriche), mentre **PersonaLive** garantisce un comportamento estremamente uniforme e prevedibile su tutto il batch.
3. **Analisi del comportamento su REFace (Subset rilevato 40,4%):** Sulle immagini di REFace, il detector `buffalo_l` ha localizzato il bounding box facciale nel **40,4%** dei casi (40 immagini su 99). Ciò non dipende da difetti di generazione del volto, bensì dal taglio fotografico nativo del dataset *FFHQ*, in cui il viso occupa quasi l'intera superficie $512 \times 512$ senza margine di sfondo attorno alla testa (elemento che penalizza i detector basati su anchor box contestuali). Sul subset rilevato, REFace registra il miglior punteggio assoluto su `topiq_nr-face` ($0,531$), ma risulta penalizzato sulle metriche di micro-texture (`brisque`: $52,11$; `niqe`: $7,30$) a causa dell'upscaling dai thumbnail di partenza a $128 \times 128$ pixel.

---

## 4. Risultati: Coerenza Temporale e Stabilità Video

La tabella seguente riassume le metriche temporali calcolate sui 220 video generati da HunyuanPortrait e PersonaLive:

| Metrica | HunyuanPortrait ($n=120$) | PersonaLive ($n=100$) | *p-value* (Mann-Whitney) |
| :--- | :---: | :---: | :---: |
| **CSIM\_drift** $\uparrow$ | $0,7215 \pm 0,1669$ | **$0,9063 \pm 0,0374$** | $2 \cdot 10^{-30}$ |
| **CSIM\_drift\_min** $\uparrow$ | $0,5462 \pm 0,2047$ | **$0,8376 \pm 0,1048$** | $1 \cdot 10^{-31}$ |
| **CSIM\_adj** $\uparrow$ | $0,9748 \pm 0,0204$ | **$0,9869 \pm 0,0106$** | $4 \cdot 10^{-9}$ |
| **Flicker LPIPS** $\downarrow$ | **$0,0058 \pm 0,0093$** | $0,0126 \pm 0,0174$ | $4 \cdot 10^{-9}$ |
| **Jitter (landmark 3D)** $\downarrow$ | $0,0150 \pm 0,0104$ | $0,0141 \pm 0,0134$ | $0,06$ (n.s.) |
| **Rapporto Jitter / Movimento** $\downarrow$ | **$1,0925 \pm 0,2801$** | $1,1939 \pm 0,2365$ | $3 \cdot 10^{-4}$ |
| **Movimento medio (`motion`)** $\cdot$ | $0,0141 \pm 0,0102$ | $0,0119 \pm 0,0101$ | — |

### Analisi dei Risultati Temporali
1. **Conservazione dell'identità (`CSIM`):** **PersonaLive** supera nettamente HunyuanPortrait nella stabilità dell'identità lungo l'intero arco del video (`CSIM_drift` medio pari a **$0,91$ contro $0,72$**, $p = 2 \cdot 10^{-30}$; valore minimo **$0,84$ contro $0,55$**). In HunyuanPortrait, al crescere dell'ampiezza dei movimenti della testa o della durata della sequenza, alcuni campioni subiscono una marcata deriva morfologica rispetto al volto iniziale.
2. **Continuità percettiva (`Flicker LPIPS`):** **HunyuanPortrait** presenta invece una migliore fluidità locale tra fotogrammi adiacenti, dimezzando lo sfarfallio percettivo rispetto a PersonaLive ($0,0058$ contro $0,0126$, $p = 4 \cdot 10^{-9}$) e mostrando un miglior rapporto tra jitter e movimento effettivo ($1,09$ contro $1,19$). Il valore di `Jitter` assoluto ($p = 0,06$) non discrimina invece in modo significativo i due modelli, riflettendo in larga parte l'incertezza intrinseca dello stimatore di landmark 3D frame-by-frame.
3. **Verifica di robustezza rispetto alla compressione video (Frame I/P/B):** Poiché circa il $96\text{--}97\%$ dei frame campionati appartiene alle tipologie predittive *P-frame* o *B-frame* (mentre gli *I-frame* coincidono quasi esclusivamente con il primo fotogramma), il confronto statistico è stato stratificato per tipologia di frame: la graduatoria tra i modelli rimane invariata su 5 metriche su 6 (con l'unica eccezione di `maniqa`, già statisticamente non significativa).

---

## 5. Analisi del Costo Computazionale

Le tempistiche di generazione (su server con GPU NVIDIA RTX 5090) e di valutazione (su Google Colab con GPU NVIDIA A100) evidenziano l'impatto determinante dell'architettura di orchestrazione del batch:

| Fase / Parametro | HunyuanPortrait | PersonaLive | REFace |
| :--- | :--- | :--- | :--- |
| **Hardware di Generazione** | $1 \times$ RTX 5090 | $2 \times$ RTX 5090 (parallelo) | $1 \times$ RTX 5090 |
| **Strategia di Caricamento Pesi** | Per singolo job (isolamento processo) | Una tantum in VRAM per GPU (via YAML) | Per singola coppia (isolamento processo) |
| **Tempo Totale Generazione Batch** | $\sim 22\text{ h } 53\text{ min}$ (312 job iniziali) | **$\sim 11\text{ min}$** (100 video) | $\sim 4\text{ h } 45\text{ min}$ (100 coppie) |
| **Tempo Medio per Campione** | $\sim 4\text{ min } 25\text{ s}$ | **$\sim 13\text{ s}$** ($\sim 10\text{ s}$ pura inferenza) | $\sim 2\text{ min } 50\text{ s}$ |
| **Tempo Valutazione IQA (su A100)** | $45\text{ min } 12\text{ s}$ ($22,6\text{ s/video}$) | $37\text{ min } 52\text{ s}$ ($22,7\text{ s/video}$) | $33\text{ s}$ ($0,33\text{ s/img}$) |
| **Tempo Valutazione Video (su A100)** | $24\text{ min } 40\text{ s}$ ($12,3\text{ s/video}$) | $20\text{ min } 25\text{ s}$ ($12,3\text{ s/video}$) | *N/A (immagini statiche)* |

---

## 6. Limiti Metodologici e Conclusioni

Dall'analisi sperimentale emergono due conclusioni principali e alcune importanti considerazioni metodologiche:

1. **Il compromesso (*Trade-off*) nei modelli di Face Reenactment:**
   * **HunyuanPortrait** privilegia la qualità percettiva del singolo frame e la morbidezza delle transizioni locali (basso *flickering*), ma paga un prezzo elevato in termini di tenuta dell'identità biometrica sulle lunghe sequenze e presenta una maggiore dispersione qualitativa tra un video e l'altro.
   * **PersonaLive** eccelle nella fedeltà e costanza dell'identità (`CSIM_drift` $> 0,90$) e nell'efficienza computazionale (grazie alla pipeline nativa multi-GPU in-memory), pur presentando una nitidezza media inferiore sui singoli fotogrammi e un leggero sfarfallio ad alta frequenza.
2. **Eterogeneità degli input (Confronto non controllato):**
   È opportuno sottolineare che i tre modelli sono stati testati sui rispettivi benchmark d'elezione e non su un dataset unificato. Le differenze demografiche (soggetti con età mediana di $23,5\text{ anni}$ per HunyuanPortrait contro $51\text{ anni}$ per PersonaLive), così come le differenze di risoluzione nativa, durata ($10\text{ s}$ contro $4\text{ s}$) e compressione degli input, influenzano fisiologicamente i punteggi NR-IQA (che tendono a premiare volti giovani, levigati e ad alta risoluzione).
3. **Specificità di REFace:**
   Nel caso del *Face Swap* statico con REFace, la qualità estetica globale del volto generato è elevata (`topiq_nr-face` $= 0,531$), ma la partenza da thumbnail a bassa risoluzione ($128 \times 128$) limita la nitidezza delle micro-texture (`brisque`/`niqe`), mentre l'inquadratura a pieno formato di FFHQ richiede l'aggiunta di *padding* contestuale o il calcolo delle metriche sull'intero frame $512 \times 512$ per estendere la copertura del face detector al $100\%$ della popolazione.