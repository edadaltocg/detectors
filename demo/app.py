"""
Gradio demo of image classification with OOD detection.
If the image example is probably OOD, the model will abstain from the prediction.

Requires:
    - gradio
"""
import json
import logging
import os
import pickle
from glob import glob

import gradio as gr
import numpy as np
import timm
import torch
from gradio.components import JSON, Image, Label
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

_logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
TOPK = 3

# load model
print("Loading model...")
model = timm.create_model("resnet50", pretrained=True)
model.to(device)
model.eval()

# dataset labels
idx2label = json.loads(open("ilsvrc2012.json").read())
idx2label = {int(k): v for k, v in idx2label.items()}
print(idx2label)

# transformation
config = resolve_data_config({}, model=model)
config["is_training"] = False
transform = create_transform(**config)

# print features names
print(get_graph_node_names(model)[0])

# load train scores
penultimate_features_key = "global_pool.flatten"
logits_key = "fc"
features_names = [penultimate_features_key, logits_key]

# create feature extractor
feature_extractor = create_feature_extractor(model, features_names)

# OOD dtector thresholds
msp_threshold = 0.3796
energy_threshold = 0.3781

## unpickle detectors


def mahalanobis_penult(features):
    scores = torch.norm(features, dim=1, keepdims=True)
    s = torch.min(scores, dim=1)[0]
    return -s.item()


def msp(logits):
    return torch.softmax(logits, dim=1).max(-1)[0].item()


def energy(logits):
    return torch.logsumexp(logits, dim=1).item()


def predict(image):
    # forward pass
    inputs = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(inputs)

    # top 5 predictions
    probabilities = torch.softmax(features[logits_key], dim=-1)
    softmax, class_idxs = torch.topk(probabilities, TOPK)
    _logger.info(softmax)
    _logger.info(class_idxs)

    result = {idx2label[i.item()]: v.item() for i, v in zip(class_idxs.squeeze(), softmax.squeeze())}
    # OOD
    msp_score = msp(features[logits_key])
    energy_score = energy(features[logits_key])
    ood_scores = {
        "msp": msp_score,
        "msp_is_ood": msp_score < msp_threshold,
        "energy": energy_score,
        "energy_is_ood": energy_score < energy_threshold,
    }
    _logger.info(ood_scores)
    return result, ood_scores


def main():
    # image examples for demo shuffled
    examples = glob("images/imagenet/*.jpg") + glob("images/ood/*.jpg")
    np.random.seed(42)
    np.random.shuffle(examples)

    # gradio interface
    interface = gr.Interface(
        fn=predict,
        inputs=Image(type="pil"),
        outputs=[
            Label(num_top_classes=TOPK, label="Model prediction"),
            JSON(label="OOD scores"),
        ],
        examples=examples,
        examples_per_page=len(examples),
        allow_flagging="never",
        theme="default",
        title="OOD Detection 🧐",
        description="Out-of-distribution (OOD) detection is an essential safety measure for machine learning models. This app demonstrates how these methods can be useful. They try to determine wether we can trust the predictions of a ResNet-50 model trained on ImageNet-1K. Enjoy the demo!",
    )
    interface.launch(
        server_port=7860,
    )
    interface.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)

    gr.close_all()
    main()
