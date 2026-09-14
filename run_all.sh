#!/bin/bash
set -e

echo "Starting full data extraction and training..."

# We already extracted SIFT and SIFT-BoVW in the previous run, they are in outputs/features/sift_full.npz and sift_bovw_full.npz.
# If they don't exist, they will be extracted.
if [ ! -f outputs/features/sift_full.npz ]; then
    echo "Extracting SIFT..."
    python main.py extract --feature sift --output outputs/features/sift_full.npz
fi

if [ ! -f outputs/features/sift_bovw_full.npz ]; then
    echo "Extracting SIFT-BoVW..."
    python main.py extract --feature sift_bovw --output outputs/features/sift_bovw_full.npz
fi

# 1. Pixel
echo "Training Pixel SVM (200 epochs)..."
python main.py train --features outputs/features/pixel.npz --name pixel_svm_200 --model svm --epochs 200 --batch-size 256
echo "Training Pixel RF (200 trees)..."
python main.py train --features outputs/features/pixel.npz --name pixel_rf_200 --model rf --num-trees 200

# 2. HOG
echo "Training HOG SVM (200 epochs)..."
python main.py train --features outputs/features/hog.npz --name hog_svm_200 --model svm --epochs 200 --batch-size 256
echo "Training HOG RF (200 trees)..."
python main.py train --features outputs/features/hog.npz --name hog_rf_200 --model rf --num-trees 200

# 3. SIFT 
echo "Training SIFT SVM (200 epochs)..."
python main.py train --features outputs/features/sift_full.npz --name sift_svm_200 --model svm --epochs 200 --batch-size 256
echo "Training SIFT RF (200 trees)..."
python main.py train --features outputs/features/sift_full.npz --name sift_rf_200 --model rf --num-trees 200

# 5. Report Table Update
echo "Generating Report Table..."
python main.py report-table \
    outputs/metrics/pixel_svm_200.json \
    outputs/metrics/pixel_rf_200.json \
    outputs/metrics/hog_svm_200.json \
    outputs/metrics/hog_rf_200.json \
    outputs/metrics/sift_svm_200.json \
    outputs/metrics/sift_rf_200.json \
    --output report/assets/results/generated_results.tex

echo "Building Report..."
bash report/build.sh

echo "Done!"
