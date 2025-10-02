#!/bin/bash

#SBATCH --job-name=torch
#SBATCH --partition=gpu
#SBATCH --output=logs/dl_run_%j.out
#SBATCH --error=logs/dl_run_%j.err
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1        # Maximo de procesos que se usar  n
#SBATCH --cpus-per-task=10
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=giancarlo.poemape@utec.edu.pe


module load python3/3.11.11

source .venv/bin/activate

# Nombre del script a ejecutar
#script="avion_resnet18.py"

# install modules
# pip install -r requirements.txt

# Crear carpeta para logs si no existe
mkdir -p logs

# Directorio de datasets
ROOT_PATH=./dataset/ETT-small/

# Lista de datasets y horizontes
datasets=("ETTh1" "ETTh2" "ETTm1" "ETTm2")
horizons=(24 48 96 192 336 720)

# Grid Search: recorrer datasets y horizontes
for data in "${datasets[@]}"; do
  for pred_len in "${horizons[@]}"; do
    echo "Ejecutando experimento con dataset=$data y pred_len=$pred_len"
    
    python3 run.py \
      --root_path $ROOT_PATH \
      --data_path ${data}.csv \
      --model_id ${data}96${pred_len} \
      --model ETSformer \
      --data $data \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 2 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --K 3 \
      --learning_rate 1e-5 \
      --itr 1 \
      --train_epochs 20
  done
done

echo "run completed"