---
title: Detectors
emoji: 🧐
colorFrom: purple
colorTo: purple
sdk: gradio
sdk_version: 3.12.0
app_file: app.py
pinned: true
license: mit
---
# Detectors Gradio Demo 🧐

Out-of-distribution (OOD) detection is an essential safety measure for machine learning models. This app demonstrates how these methods can be useful. They try to determine whether we can trust the predictions of a ResNet-50 model trained on ImageNet-1K.

This demo is [online](https://huggingface.co/spaces/edadaltocg/ood-detection) at `https://huggingface.co/spaces/edadaltocg/ood-detection`

## Running Gradio app locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
python app.py
```

3. Open the app in your browser at `http://localhost:7860`.

## Methods implemented

- [x] [Mahalanobis Distance](https://arxiv.org/abs/1807.03888)
- [x] [Maximum Softmax Probability](https://arxiv.org/abs/1610.02136)
- [x] [Energy Based Out-of-Distribution Detection](https://arxiv.org/abs/2010.03759)
