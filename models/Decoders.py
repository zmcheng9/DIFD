import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class ASPP(nn.Module):
    def __init__(self, out_channels=256):
        super(ASPP, self).__init__()
        self.layer6_0 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.3)
        )
        self.layer6_1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.3)
        )
        self.layer6_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.3)
        )
        self.layer6_3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.3)
        )
        self.layer6_4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.3)
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        feature_size = x.shape[-2:]
        global_feature = F.avg_pool2d(x, kernel_size=feature_size)  # [8, 256, 1, 1]

        global_feature = self.layer6_0(global_feature)

        global_feature = global_feature.expand(-1, -1, feature_size[0], feature_size[1])  # [8, 256, 41, 41]
        out = torch.cat(
            [global_feature, self.layer6_1(x), self.layer6_2(x), self.layer6_3(x), self.layer6_4(x)], dim=1)
        return out  # [8, 1280, 41, 41]   1280=256*5


class Decoder(nn.Module):
    def __init__(self, reduce_dim=512):
        super(Decoder, self).__init__()

        # self.ASPP = ASPP()
        self.merge = nn.Sequential(
            nn.Conv2d(1024+3, 512, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
            # nn.Conv2d(512, reduce_dim, kernel_size=1, padding=0, bias=False),
            # nn.ReLU(inplace=True),
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, 1, kernel_size=1, padding=0, bias=False)
        )



    def forward(self, x):

        x = self.merge(x)
        # x = self.ASPP(x)
        x = self.res1(x)
        x = self.res2(x) + x
        out = self.cls(x)

        return out

# decoder = Decoder()
# fts = torch.rand(8, 1026, 64, 64)
# y = decoder(fts)
# res = torch.cat([y, y], dim=1)
# print(y.size(), res.size())