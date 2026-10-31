import random, shutil, os

random.seed(42)
src_dir = os.path.expanduser("~/ffhq_dataset/thumbnails128x128")
out_source = os.path.expanduser("~/REFace/examples/FaceSwap/Batch100_source")
out_target = os.path.expanduser("~/REFace/examples/FaceSwap/Batch100_target")

os.makedirs(out_source, exist_ok=True)
os.makedirs(out_target, exist_ok=True)

files = sorted(f for f in os.listdir(src_dir) if f.endswith(".png"))
random.shuffle(files)

n = 100
sources = files[:n]
targets = files[n:2*n]

for i, (s, t) in enumerate(zip(sources, targets)):
    pair_name = f"pair_{i:03d}.png"
    shutil.copy(os.path.join(src_dir, s), os.path.join(out_source, pair_name))
    shutil.copy(os.path.join(src_dir, t), os.path.join(out_target, pair_name))

print(f"Create {n} coppie in {out_source} e {out_target}")
