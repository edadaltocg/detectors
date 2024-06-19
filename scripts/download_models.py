import timm

models = [
    "resnet34.tv_in1k",
    "resnet50.tv_in1k",
    "resnet101.tv_in1k",
    "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "vit_large_patch16_224.augreg_in21k_ft_in1k",
    "densenet121.tv_in1k",
    "vgg16.tv_in1k",
    "mobilenetv3_small_100.lamb_in1k",
    "mobilenetv3_large_100.ra_in1k",
    "mobilenetv3_large_100.miil_in21k_ft_in1k",
]

for model_name in models:
    model = timm.create_model(model_name, pretrained=True)
    print(f"Downloaded {model_name}")
