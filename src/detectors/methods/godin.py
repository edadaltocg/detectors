def get_godin_score(inputs, model, forward_func, method_args):
    noiseMagnitude1 = method_args["magnitude"]

    criterion = nn.CrossEntropyLoss()
    inputs = torch.autograd.Variable(inputs, requires_grad=True)
    # outputs = model(inputs)
    outputs, _, _ = forward_func(inputs, model)

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    labels = torch.autograd.Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
    # outputs = model(Variable(tempInputs))
    with torch.no_grad():
        _, hx, _ = forward_func(tempInputs, model)
    # Calculating the confidence after adding perturbations
    nnOutputs = hx.data.cpu()
    nnOutputs = nnOutputs.numpy()
    # nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    # nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    scores = np.max(nnOutputs, axis=1)

    return scores
