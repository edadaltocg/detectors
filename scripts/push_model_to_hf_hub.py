"""Requirements:
- HF_TOKEN environment variable set to your HuggingFace token
- Jinja2 installed
- Git LFS installed
"""
import argparse
import json
import logging
import os
from typing import Optional

import timm
import timm.models
import torch
from huggingface_hub.hf_api import create_repo, upload_file, whoami
from huggingface_hub.repocard import ModelCard
from huggingface_hub.repocard_data import EvalResult, ModelCardData

import detectors

_logger = logging.getLogger(__name__)

DATASET_TYPE_TO_DATASET_NAME = dict(
    cifar10="CIFAR-10",
    cifar100="CIFAR-100",
    imagenet="ImageNet",
    svhn="SVHN",
)


def main(
    model_name: str,
    dataset_name: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    private: bool = False,
    seed: int = 1,
):
    token = os.environ["HF_TOKEN"]
    repo_id = model_name
    user = whoami(token=token)["name"]
    user_info = whoami(token=token)
    folder_name = model_name + f"_{seed}"
    _logger.debug(json.dumps(user_info, indent=2))

    model = timm.create_model(model_name, pretrained=False)
    if checkpoint_path is None:
        checkpoint_path = os.path.join(detectors.config.CHECKPOINTS_DIR, folder_name, "best.pth")
    if dataset_name is None:
        dataset_name = model_name.split("_")[-1]
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint)
    model_config = model.default_cfg
    eval_results = json.load(open(os.path.join(detectors.config.CHECKPOINTS_DIR, folder_name, "eval_results.json")))
    acc = eval_results["best_accuracy"]

    card_data = ModelCardData(
        language="en",
        license="mit",
        library_name="timm",
        model_name=model_name,
        tags=["image-classification", model_config["architecture"], dataset_name],
        datasets=dataset_name,
        metrics=["accuracy"],
        eval_results=[
            EvalResult(
                task_type="image-classification",
                dataset_type=dataset_name,
                dataset_name=DATASET_TYPE_TO_DATASET_NAME[dataset_name],
                metric_type="accuracy",
                metric_value=acc,
            )
        ],
    )

    hyperparameters = json.load(
        open(os.path.join(detectors.config.CHECKPOINTS_DIR, folder_name, "hyperparameters.json"))
    )
    model_card = ModelCard.from_template(
        card_data,
        template_path="templates/MODEL_CARD_TEMPLATE.md",
        model_summary=f"This model is a small {model_config['architecture']} trained on {dataset_name}.",
        license="MIT",
        training_data=f"Training data is {dataset_name}.",
        testing_data=f"Testing data is {dataset_name}.",
        results=f"{acc}",
        author="Eduardo Dadalto",
        hyperparameters=hyperparameters,
        get_started_code=f'```python\nimport detectors\nimport timm\n\nmodel = timm.create_model("{model_name}", pretrained=True)\n```',
    )
    _logger.info("Model card: %s", model_card.data)
    model_card.save(os.path.join(detectors.config.CHECKPOINTS_DIR, folder_name, "README.md"))

    create_repo(repo_id=f"{user}/{repo_id}", exist_ok=True, token=token, private=private, repo_type="model")
    model_card.push_to_hub(repo_id=f"{user}/{repo_id}", token=token)

    # upload files
    file_list = ["train_results.json", "eval_results.json", "hyperparameters.json"]
    for file_name in file_list:
        try:
            with open(os.path.join(detectors.config.CHECKPOINTS_DIR, folder_name, file_name), "rb") as f:
                upload_file(
                    path_or_fileobj=f,
                    path_in_repo=file_name,
                    repo_id=f"{user}/{repo_id}",
                    token=token,
                    commit_message=f"Upload {file_name}",
                )
        except FileNotFoundError:
            _logger.warning(f"File {file_name} not found.")

    # push model
    timm.models.push_to_hf_hub(
        model,
        repo_id=f"{repo_id}",
        commit_message="Update model",
        model_config=model_config,
        model_card=model_card,
        private=private,
        token=token,
    )

    model_reloaded = timm.create_model(f"{model_name}", pretrained=True)
    assert str(model.state_dict()) == str(model_reloaded.state_dict())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18_cifar10")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    _logger.info(json.dumps(args.__dict__, indent=2))

    main(args.model, args.dataset, args.checkpoint_path, private=args.private, seed=args.seed)
