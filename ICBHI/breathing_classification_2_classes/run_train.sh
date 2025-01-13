# hoffman run code
source ~/.bashrc
module load cuda/11.8
conda activate /u/home/s/spaceman/.conda/envs/speechbrain
cd /u/home/s/spaceman/project-voicesafe/spapl/vijay/speechbrain/speechbrain/recipes/ICBHI/breathing_classification_2_classes/
python train.py hparams/train_patients.yaml 