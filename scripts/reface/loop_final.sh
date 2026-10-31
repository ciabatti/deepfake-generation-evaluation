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
