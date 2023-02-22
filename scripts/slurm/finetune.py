import os
import subprocess
import uuid

os.makedirs(os.path.join("cluster", "logs"), exist_ok=True)
SLURM_ACCOUNT = os.environ.get("SLURM_ACCOUNT", "")

model_names = [
    "vit_base_patch16_224_in21k_ft_cifar10",
    "vit_base_patch16_224_in21k_ft_cifar100",
    "vit_base_patch16_224_in21k_ft_svhn",
]

for model_name in model_names:
    job_name = f"ce_{model_name}"
    file = f"""#!/bin/sh

#SBATCH --job-name={job_name}
#SBATCH --gres=gpu:1
#SBATCH --account={SLURM_ACCOUNT}
#SBATCH --no-requeue
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=cluster/logs/%A_%a_%x.out
#SBATCH --error=cluster/logs/%A_%a_%x.err

echo $CHECKPOINTS_DIR
module purge
source .venv/bin/activate

python -m scripts.train \
    --model {model_name} \
    --dataset {model_name.split("_")[-1]} \
    --seed $SLURM_ARRAY_TASK_ID \
    --config scripts/train_configs/ft_{model_name.split("_")[-1]}.json

wait
    """

    filename = f"./tmp_{str(uuid.uuid1())}.sh"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(file)
    f.close()

    try:
        process = subprocess.Popen((f"sbatch --array=1-1 {filename}").split(" "))
        processed, _ = process.communicate()
    except Exception as e:
        print(e)
    finally:
        os.remove(filename)
    print(f"Submitted job array: {job_name}")
