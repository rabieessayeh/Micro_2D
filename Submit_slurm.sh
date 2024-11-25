#!/bin/bash
#SBATCH --job-name=train_gpu          # Nom du job
#SBATCH --output=train_gpu.out        # Fichier de sortie standard
#SBATCH --error=train_gpu.err         # Fichier de sortie des erreurs
#SBATCH --ntasks=1                    # Nombre total de tâches
#SBATCH --cpus-per-task=4             # Nombre de CPU par tâche
#SBATCH --gres=gpu:1                  # Nombre de GPUs nécessaires
#SBATCH --mem=16G                     # Mémoire allouée
#SBATCH --time=04:00:00               # Temps limite (HH:MM:SS)
#SBATCH --partition=gpu               # Partition réservée aux GPU

# Charger les modules nécessaires
module load python/3.9
module load cuda/11.8


# Lancer le script Python
python train_model.py
