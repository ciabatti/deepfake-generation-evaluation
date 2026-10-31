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
