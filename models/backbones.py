from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock

def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


# def resnet50w2(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 6, 3], widen=2, **kwargs)


# def resnet50w4(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 6, 3], widen=4, **kwargs)


# def resnet50w5(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 6, 3], widen=5, **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3])

def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3])