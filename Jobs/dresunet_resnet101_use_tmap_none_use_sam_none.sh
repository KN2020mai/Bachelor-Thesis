#!/usr/local_rwth/bin/zsh
### ask for 15 GB memory
#SBATCH --mem-per-cpu=15360M   #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)
### name the job
#SBATCH --job-name=dresunet_resnet101_use_tmap_none_use_sam_none
### job run time
#SBATCH --time=10:00:00
### declare the merged STDOUT/STDERR file
#SBATCH --output=output.%J.dresunet_resnet101_use_tmap_none_use_sam_none.txt
###
#SBATCH --mail-type=ALL
###
#SBATCH --mail-user=yongli.mou@rwth-aachen.de
### request a GPU
#SBATCH --gres=gpu:1

### begin of executable commands
cd $HOME/Keni_BA/Code
module switch intel gcc
module load python/3.8.7
module load cuda/11.6
### pip install --user opencv-python fastremap tifffile numba natsort
### pip install --user torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
python3 train_sartorius.py --model dresunet --backbone resnet101