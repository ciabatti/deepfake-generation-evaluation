# Valutazione e Confronto di Modelli Deepfake: HunyuanPortrait, PersonaLive e REFace

Questo repository raccoglie la documentazione tecnica, gli script di automazione e il codice di valutazione per un'analisi comparativa su larga scala di tre modelli di generazione deepfake/face swap.

## Struttura del Repository

* **`docs/`**: Relazioni tecniche dettagliate sui setup sperimentali e sui risultati:
  * [Setup HunyuanPortrait](docs/01_setup_hunyuanportrait.md)
  * [Setup REFace](docs/02_setup_reface.md)
  * [Setup PersonaLive](docs/03_setup_personalive.md)
  * [Relazione e Risultati Quantitativi](docs/04_relazione_risultat.md)
* **`scripts/`**: Script Python e Bash sviluppati per la gestione batch, la pulizia dei dataset e l'orchestrazione multi-GPU.
* **`notebooks/`**: Notebook Google Colab (`valutazione_deepfake.ipynb`) per l'esecuzione delle metriche NR-IQA (`pyiqa`) e di coerenza temporale (ArcFace, LPIPS, Jitter).
* **`results/`**: Tabelle riepilogative (CSV) e grafici boxplot generati in fase di valutazione.

## Sintesi dei Risultati
* **PersonaLive**: Qualità visiva del fotogramma nettamente superiore (migliori punteggi IQA) ed elevatissima efficienza computazionale in batch (architettura multi-GPU *in-memory*). Tuttavia, a parità di dataset, subisce lo stesso decadimento dell'identità (*identity drift*) di HunyuanPortrait sui movimenti ampi.
* **HunyuanPortrait**: Conserva il primato sulla fluidità pura del movimento (minor flickering tra frame adiacenti e interpolazione più morbida), ma presenta una resa estetica del singolo frame più variabile e mediamente inferiore a PersonaLive.
* **REFace**: Solido sul face swap statico, ma fortemente dipendente dalla risoluzione e dal framing del dataset di partenza (FFHQ), che ha limitato il rilevamento facciale per il calcolo delle metriche al 40,4% del campione.
