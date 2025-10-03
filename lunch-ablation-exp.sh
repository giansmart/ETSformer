#!/bin/bash

#SBATCH --job-name=gipo_torch
#SBATCH --partition=gpu
#SBATCH --output=logs/ablation_K_%j.out
#SBATCH --error=logs/ablation_K_%j.err
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1        # Maximo de procesos que se usaran
#SBATCH --cpus-per-task=10
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=giancarlo.poemape@utec.edu.pe


module load python3/3.11.11

source .venv/bin/activate

# Crear carpeta para logs si no existe
mkdir -p logs

# Directorio de datasets
ROOT_PATH=./dataset/ETT-small/

# ========================================
# EXPERIMENTO DE ABLACION: Top-K Fourier Bases
# ========================================
# Objetivo: Evaluar el impacto del parametro K (Frequency Attention)
# K controla cuantas frecuencias estacionales captura el modelo
# Baseline: K=3 (valor usado en experimentos anteriores)
# Valores a probar: K=1, K=5, K=7
# ========================================

# Configuracion del experimento
dataset="ETTh1"
horizons=(24 96)
K_values=(1 3 5 7)  # Incluimos K=3 como baseline para comparacion

echo "============================================"
echo "INICIO: Experimento de Ablacion - Top-K Fourier Bases"
echo "Dataset: $dataset"
echo "Horizontes: ${horizons[@]}"
echo "Valores de K: ${K_values[@]}"
echo "============================================"
echo ""

# Grid Search: recorrer horizontes y valores de K
for pred_len in "${horizons[@]}"; do
  for K in "${K_values[@]}"; do
    echo "--------------------------------------------"
    echo "Ejecutando: dataset=$dataset, pred_len=$pred_len, K=$K"
    echo "--------------------------------------------"

    python3 run.py \
      --root_path $ROOT_PATH \
      --data_path ${dataset}.csv \
      --model_id ${dataset}_K${K}_pl${pred_len} \
      --model ETSformer \
      --data $dataset \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 2 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des "Ablation_K${K}" \
      --K $K \
      --learning_rate 1e-5 \
      --itr 1 \
      --train_epochs 20

    echo "Completado: K=$K, pred_len=$pred_len"
    echo ""
  done
done

echo "============================================"
echo "COMPLETADO: Experimento de Ablacion - Top-K"
echo "Total de experimentos: $((${#horizons[@]} * ${#K_values[@]}))"
echo "============================================"