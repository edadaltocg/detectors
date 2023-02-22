from typing import Optional, Tuple, Union


def hf_hub_url_template(model_name: str):
    return f"https://huggingface.co/edadaltocg/{model_name}/resolve/main/pytorch_model.bin"


class ModelDefaultConfig(dict):
    """
    Default configuration for models from `timm` library.

    Example:
    --------
    ```
    {
        'url': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': (7, 7),
        'crop_pct': 0.875,
        'interpolation': 'bilinear',
        'fixed_input_size': True,
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'first_conv': 'conv1',
        'classifier': 'fc',
        'architecture': 'resnet18'
    }
    ```
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(
        self,
        url: str,
        num_classes: int,
        input_size: Union[Tuple[int, int, int], Tuple[int, int], int],
        pool_size: Optional[Tuple[int, int]],
        crop_pct: float,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        first_conv: str,
        classifier: str,
        architecture: str,
        interpolation: str = "bilinear",
        fixed_input_size: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            url=url,
            num_classes=num_classes,
            input_size=input_size,
            pool_size=pool_size,
            crop_pct=crop_pct,
            interpolation=interpolation,
            fixed_input_size=fixed_input_size,
            mean=mean,
            std=std,
            first_conv=first_conv,
            classifier=classifier,
            architecture=architecture,
            **kwargs,
        )
