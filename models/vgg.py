# import os.path as osp
#
# import fcn
#
# import torch
# import torchvision
#
#
# def VGG16(pretrained=False):
#     model = torchvision.models.vgg16(pretrained=False)
#     if not pretrained:
#         return model
#     model_file = _get_vgg16_pretrained_model()
#     state_dict = torch.load(model_file)
#     model.load_state_dict(state_dict)
#     return model
#
#
# def _get_vgg16_pretrained_model():
#     return fcn.data.cached_download(
#         url='http://drive.google.com/uc?id=0B9P1L--7Wd2vLTJZMXpIRkVVRFk',
#         path=osp.expanduser('~/data/models/pytorch/vgg16_from_caffe.pth'),
#         md5='aa75b158f4181e7f6230029eb96c1b13',
#     )


import paddle.fluid as fluid

from ..models.nn import ReLU, Dropout2d


class VGG(fluid.dygraph.Layer):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = fluid.dygraph.nn.Pool2D(pool_size=(7, 7), global_pooling=True, pool_type="avg")
        self.classifier = fluid.dygraph.container.Sequential(
            fluid.dygraph.Linear(512*7*7, 4096, act="relu"),
            ReLU(),
            Dropout2d(),
            fluid.dygraph.Linear(input_dim=4096, output_dim=4096, act="relu"),
            ReLU(),
            Dropout2d(),
            fluid.dygraph.Linear(input_dim=4096, output_dim=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = fluid.layers.flatten(x, axis=1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [fluid.dygraph.Pool2D(pool_size=2, pool_stride=2, pool_type="max")]
        else:
            conv2d = fluid.dygraph.Conv2D(in_channels, v, filter_size=3, padding=1)
            if batch_norm:
                layers += [conv2d,
                           fluid.dygraph.BatchNorm(num_channels=v, ),
                           ReLU()]
            else:
                # conv2d = fluid.dygraph.Conv2D(in_channels, v, filter_size=3, padding=1, act="relu")
                layers += [conv2d,
                           ReLU()]
            in_channels = v
    return fluid.dygraph.container.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = fluid.dygraph.load_dygraph()
        model.set_dict(state_dict)
    return model


def vgg16(pretrained=False, progress=True, **kwargs):
    return _vgg("vgg16", "D", False, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    return _vgg("vgg19", "E", False, pretrained, progress, **kwargs)