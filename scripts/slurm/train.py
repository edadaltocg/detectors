import os
import subprocess
import uuid

os.makedirs(os.path.join("cluster", "logs"), exist_ok=True)
SLURM_ACCOUNT = os.environ.get("SLURM_ACCOUNT", "")

model_names = [
    "densenet121_cifar10",
    "densenet121_cifar100",
    "densenet121_svhn",
    "resnet18_cifar10",
    "resnet18_cifar100",
    "resnet18_svhn",
    "resnet34_cifar10",
    "resnet34_cifar100",
    "resnet34_svhn",
    "resnet50_cifar10",
    "resnet50_cifar100",
    "resnet50_svhn",
    "vgg16_cifar10",
    "vgg16_cifar100",
    "vgg16_svhn",
    "vgg16_bn_cifar10",
    "vgg16_bn_cifar100",
    "vgg16_bn_svhn",
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
    --config scripts/train_configs/{model_name.split("_")[-1]}.json

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
