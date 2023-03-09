import torch


# Our proposed RankFeat Score
def iterate_data_rankfeat(data_loader, model, temperature):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        if b % 100 == 0:
            print("{} batches processed".format(b))
        inputs = x.cuda()

        # Logit of Block 4 feature
        feat1 = model.module.intermediate_forward(inputs, layer_index=4)
        B, C, H, W = feat1.size()
        feat1 = feat1.view(B, C, H * W)
        u, s, v = torch.linalg.svd(feat1, full_matrices=False)
        feat1 = feat1 - s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(v[:, 0:1, :])
        # if you want to use PI for acceleration, comment the above 2 lines and uncomment the line below
        # feat1 = feat1 - power_iteration(feat1, iter=20)
        feat1 = feat1.view(B, C, H, W)
        logits1 = model.module.head(model.module.before_head(feat1))

        # Logit of Block 4 feature
        feat2 = model.module.intermediate_forward(inputs, layer_index=3)
        B, C, H, W = feat2.size()
        feat2 = feat2.view(B, C, H * W)
        u, s, v = torch.linalg.svd(feat2, full_matrices=False)
        feat2 = feat2 - s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(v[:, 0:1, :])
        # if you want to use PI for acceleration, comment the above 2 lines and uncomment the line below
        # feat2 = feat2 - power_iteration(feat2, iter=20)
        feat2 = feat2.view(B, C, H, W)
        feat2 = model.module.body.block4(feat2)
        logits2 = model.module.head(model.module.before_head(feat2))

        # Fusion at the logit space
        logits = (logits1 + logits2) / 2
        conf = temperature * torch.logsumexp(logits / temperature, dim=1)
        confs.extend(conf.data.cpu().numpy())

    return confs
