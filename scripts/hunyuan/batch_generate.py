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