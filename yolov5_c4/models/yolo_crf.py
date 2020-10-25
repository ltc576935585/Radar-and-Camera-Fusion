import argparse

from models.experimental import *

def min_max_pool2d(x):
    max_x = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(x)
    min_x = min_pool2d(x)
    return torch.cat([max_x, min_x], dim=1)  # concatenate on channel


def min_max_pool2d_output_shape(input_shape):
    shape = list(input_shape)
    shape[1] = int(shape[1] / 2)
    shape[2] = int(shape[2] / 2)
    shape[3] *= 2
    return tuple(shape)


def min_pool2d(x, padding=1):
    max_val = torch.max(x) + 1  # we gonna replace all zeros with that value
    # replace all 0s with very high numbers
    is_zero = torch.where(torch.eq(x, 0.), max_val, x)
    x = is_zero + x

    # execute pooling with 0s being replaced by a high number
    min_x = -nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=padding)(-x)

    # depending on the value we either substract the zero replacement or not
    is_result_zero = torch.where(torch.eq(min_x, max_val), max_val, min_x)
    min_x = min_x - is_result_zero

    return min_x  # concatenate on channel


def min_pool2d_output_shape(input_shape):
    shape = list(input_shape)
    shape[1] = int(shape[1] / 2)
    shape[2] = int(shape[2] / 2)
    return tuple(shape)
class VggMax(nn.Module):
    def __init__(self, backbone='vgg16'):
        super(VggMax, self).__init__()

        # Read config variables
        self.fusion_blocks = [0,1,2,3,4,5]
        self.channels = 5

        self.block1_Image = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.block2_Image = nn.Sequential(
            nn.Conv2d(65, 128,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.block3_Image = nn.Sequential(
            nn.Conv2d(130, int(256 * cfg.network_width),
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.block4_Image = nn.Sequential(
            nn.Conv2d(258, 512,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.block5_Image = nn.Sequential(
            nn.Conv2d(514, 1024,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # separate input
        if len(self.cfg.channels) > 3:
            image_input = x[:, :3, :, :]
            radar_input = x[:, 3:, :, :]
        else:
            image_input = x
            radar_input = None

        # Bock 0 Fusion
        if len(self.cfg.channels) > 3:
            if 0 in self.cfg.fusion_blocks:
                x = torch.cat([image_input, radar_input], dim=1)
            else:
                x = image_input
        else:
            x = image_input
        concat_0 = x

        # Block 1 - Image
        x = self.block1_Image(x)
        if self.cfg.pooling == 'maxmin':
            x = min_max_pool2d(x)
        else:
            x = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(x)

        block1_pool = x
        # Block 1 - Radar
        if len(self.cfg.channels) > 3:
            if self.cfg.pooling == 'min':
                y = min_pool2d(radar_input)
            elif self.cfg.pooling == 'maxmin':
                y = min_max_pool2d(radar_input)
            elif self.cfg.pooling == 'conv':
                y = nn.Conv2d(len(self.cfg.channels.shape[3:]), 64 * self.cfg.network_width,
                              kernel_size=3, stride=1, padding=1)(radar_input)
                y = nn.ReLU()(y)
            else:
                y = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(radar_input)

            rad_block1_pool = y
            # Concatenate Block 1 Radar to image
            if 1 in self.cfg.fusion_blocks:
                x = torch.cat([x, y], dim=1)

        concat_1 = x
        # Block2
        x = self.block2_Image(x)

        if self.cfg.pooling == 'maxmin':
            x = min_max_pool2d(x)
        else:
            x = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(x)

        block2_pool = x
        # Block 2 - Radar
        if len(self.cfg.channels) > 3:
            if self.cfg.pooling == 'min':
                y = min_pool2d(y)
            elif self.cfg.pooling == 'maxmin':
                y = min_max_pool2d(y)
            elif self.cfg.pooling == 'conv':
                y = nn.Conv2d(len(self.cfg.channels.shape[3:]), 64 * self.cfg.network_width,
                              kernel_size=3, stride=1, padding=1)(y)
                y = nn.ReLU()(y)
            else:
                y = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(y)

            rad_block2_pool = y
            # Concatenate Block 1 Radar to image
            if 2 in self.cfg.fusion_blocks:
                x = torch.cat([x, y], dim=1)

        concat_2 = x
        # Block 3
        x = self.block3_Image(x)

        if self.cfg.pooling == 'maxmin':
            x = min_max_pool2d(x)
        else:
            x = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(x)

        block3_pool = x
        # Block 3 - Radar
        if len(self.cfg.channels) > 3:
            if self.cfg.pooling == 'min':
                y = min_pool2d(y)
            elif self.cfg.pooling == 'maxmin':
                y = min_max_pool2d(y)
            elif self.cfg.pooling == 'conv':
                y = nn.Conv2d(len(self.cfg.channels.shape[3:]), 64 * self.cfg.network_width,
                              kernel_size=3, stride=1, padding=1)(y)
                y = nn.ReLU()(y)
            else:
                y = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(y)

            rad_block3_pool = y
            # Concatenate Block 1 Radar to image
            if 3 in self.cfg.fusion_blocks:
                x = torch.cat([x, y], dim=1)

        concat_3 = x
        # Block 4
        x = self.block4_Image(x)

        if self.cfg.pooling == 'maxmin':
            x = min_max_pool2d(x)
        else:
            x = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(x)

        block4_pool = x
        # Block 4 - Radar
        if len(self.cfg.channels) > 3:
            if self.cfg.pooling == 'min':
                y = min_pool2d(y)
            elif self.cfg.pooling == 'maxmin':
                y = min_max_pool2d(y)
            elif self.cfg.pooling == 'conv':
                y = nn.Conv2d(len(self.cfg.channels.shape[3:]), 64 * self.cfg.network_width,
                              kernel_size=3, stride=1, padding=1)(y)
                y = nn.ReLU()(y)
            else:
                y = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(y)

            rad_block4_pool = y
            # Concatenate Block 1 Radar to image
            if 4 in self.cfg.fusion_blocks:
                x = torch.cat([x, y], dim=1)

        concat_4 = x
        # Block 5
        x = self.block5_Image(x)

        if self.cfg.pooling == 'maxmin':
            x = min_max_pool2d(x)
        else:
            x = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(x)

        block5_pool = x
        # Block 5 - Radar
        if len(self.cfg.channels) > 3:
            if self.cfg.pooling == 'min':
                y = min_pool2d(y)
            elif self.cfg.pooling == 'maxmin':
                y = min_max_pool2d(y)
            elif self.cfg.pooling == 'conv':
                y = nn.Conv2d(len(self.cfg.channels.shape[3:]), 64 * self.cfg.network_width,
                              kernel_size=3, stride=1, padding=1)(y)
                y = nn.ReLU()(y)
            else:
                y = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(y)

            rad_block5_pool = y
            # Concatenate Block 1 Radar to image
            if 5 in self.cfg.fusion_blocks:
                x = torch.cat([x, y], dim=1)

        concat_5 = x
        layer_outputs = [concat_3, concat_4, concat_5]
        radar_layers = [rad_block1_pool, rad_block2_pool, rad_block3_pool, rad_block4_pool, rad_block5_pool]
        # print('a')
        return {'layer_outputs': layer_outputs, 'radar_layers': radar_layers}

class Detect(nn.Module):
    def __init__(self, nc=80, anchors=()):  # detection layer
        super(Detect, self).__init__()
        self.stride = None  # strides computed during build
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.export = False  # onnx export

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, model_cfg='yolov5s.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if type(model_cfg) is dict:
            self.md = model_cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            with open(model_cfg) as f:
                self.md = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc and nc != self.md['nc']:
            print('Overriding %s nc=%g with nc=%g' % (model_cfg, self.md['nc'], nc))
            self.md['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(self.md, ch=[ch])  # model, savelist, ch_out
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        torch_utils.initialize_weights(self)
        self._initialize_biases()  # only run once
        torch_utils.model_info(self)
        # print('')

    def forward(self, x, augment=False, profile=False):#(bs,3,H,W)
        # print('a')
        return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                try:
                    import thop
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
                except:
                    o = 0
                t = torch_utils.time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((torch_utils.time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run#(bs,32 0.5W 0.5H)
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for f, s in zip(m.f, m.stride):  #  from
            mi = self.model[f % m.i]
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for f in sorted([x % m.i for x in m.f]):  #  from
            b = self.model[f].bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%g Conv2d.bias:' + '%10.3g' * 6) % (f, *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ', end='')
        for m in self.model.modules():
            if type(m) is Conv:
                m.conv = torch_utils.fuse_conv_and_bn(m.conv, m.bn)  # update conv
                m.bn = None  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        torch_utils.model_info(self)
        return self

def parse_model(md, ch):  # model_dict, input_channels(3)
    print('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = md['anchors'], md['nc'], md['depth_multiple'], md['width_multiple']
    na = (len(anchors[0]) // 2)  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(md['backbone'] + md['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            f = f or list(reversed([(-1 if j == i else j - 1) for j, x in enumerate(ch) if x == no]))
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    device = torch_utils.select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # ONNX export
    # model.model[-1].export = True
    # torch.onnx.export(model, img, opt.cfg.replace('.yaml', '.onnx'), verbose=True, opset_version=11)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
