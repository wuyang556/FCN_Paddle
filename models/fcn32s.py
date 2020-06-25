
import paddle.fluid as fluid
from .nn import ReLU, Dropout2d


class FCN32s(fluid.dygraph.Layer):

    def __init__(self, n_class=21):
        super(FCN32s, self).__init__()
        # conv1
        self.conv1_1 = fluid.dygraph.Conv2D(3, 64, 3, padding=100)
        self.relu1_1 = ReLU()
        self.conv1_2 = fluid.dygraph.Conv2D(64, 64, 3, padding=1)
        self.relu1_2 = ReLU()
        self.pool1 = fluid.dygraph.Pool2D(2, pool_stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = fluid.dygraph.Conv2D(64, 128, 3, padding=1)
        self.relu2_1 = ReLU()
        self.conv2_2 = fluid.dygraph.Conv2D(128, 128, 3, padding=1)
        self.relu2_2 = ReLU()
        self.pool2 = fluid.dygraph.Pool2D(2, pool_stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = fluid.dygraph.Conv2D(128, 256, 3, padding=1)
        self.relu3_1 = ReLU()
        self.conv3_2 = fluid.dygraph.Conv2D(256, 256, 3, padding=1)
        self.relu3_2 = ReLU()
        self.conv3_3 = fluid.dygraph.Conv2D(256, 256, 3, padding=1)
        self.relu3_3 = ReLU()
        self.pool3 = fluid.dygraph.Pool2D(2, pool_stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = fluid.dygraph.Conv2D(256, 512, 3, padding=1)
        self.relu4_1 = ReLU()
        self.conv4_2 = fluid.dygraph.Conv2D(512, 512, 3, padding=1)
        self.relu4_2 = ReLU()
        self.conv4_3 = fluid.dygraph.Conv2D(512, 512, 3, padding=1)
        self.relu4_3 = ReLU()
        self.pool4 = fluid.dygraph.Pool2D(2, pool_stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = fluid.dygraph.Conv2D(512, 512, 3, padding=1)
        self.relu5_1 = ReLU()
        self.conv5_2 = fluid.dygraph.Conv2D(512, 512, 3, padding=1)
        self.relu5_2 = ReLU()
        self.conv5_3 = fluid.dygraph.Conv2D(512, 512, 3, padding=1)
        self.relu5_3 = ReLU()
        self.pool5 = fluid.dygraph.Pool2D(2, pool_stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = fluid.dygraph.Conv2D(512, 4096, 7)
        self.relu6 = ReLU()
        self.drop6 = Dropout2d()

        # fc7
        self.fc7 = fluid.dygraph.Conv2D(4096, 4096, 1)
        self.relu7 = ReLU()
        self.drop7 = Dropout2d()

        self.score_fr = fluid.dygraph.Conv2D(4096, n_class, 1)
        self.upscore = fluid.dygraph.Conv2DTranspose(n_class, n_class, 64, stride=32,
                                                     bias_attr=False)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)

        h = self.upscore(h)
        h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()

        return h

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, fluid.dygraph.Conv2D) and isinstance(l2, fluid.dygraph.Conv2D):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
