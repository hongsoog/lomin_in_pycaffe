DetectionDemo.__init__ { // BEGIN
    self.model = build_detection_model(self.cfg)
    GeneralizedRCNN.__init__(self, cfg) { //BEGIN
        // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/detector/generalized_rcnn.py

        Params:
        cfg:
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg) // CALL

        build_backbone(cfg) { // BEGIN
            defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/backbone/backbone.py

            Params:
            cfg:

            build_resnet_fpn_p3p7_backbone(cfg) { // BEGIN
                // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/backbone/backbone.py

                Params:
                cfg:
                body = resnet.ResNet(cfg) // CALL

                Resnet.__init__ { //BEGIN
                    // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/backbone/resnet.py

                    Params:
                    cfg:
                    _STEM_MODULES: {'StemWithFixedBatchNorm': <class 'maskrcnn_benchmark.modeling.backbone.resnet.StemWithFixedBatchNorm'>}
                    _cfg.MODEL.RESNETS.STEM_FUNC: StemWithFixedBatchNorm
                    stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
                    stem_module: <class 'maskrcnn_benchmark.modeling.backbone.resnet.StemWithFixedBatchNorm'>
                    cfg.MODEL.BACKBONE.CONV_BODY: R-50-FPN-RETINANET
                    stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY=R-50-FPN-RETINANET]
                    stage_specs: (StageSpec(index=1, block_count=3, return_features=True), StageSpec(index=2, block_count=4, return_features=True), StageSpec(index=3, block_count=6, return_features=True), StageSpec(index=4, block_count=3, return_features=True))
                    _TRANSFORMATION_MODULES: {'BottleneckWithFixedBatchNorm': <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>}
                    cfg.MODEL.RESNETS.TRANS_FUNC: BottleneckWithFixedBatchNorm
                    transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC=BottleneckWithFixedBatchNorm]
                    transformation_module: <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>
                    self.stem = stem_module(cfg)
                    self.stem: StemWithFixedBatchNorm(
                        (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                        (bn1): FrozenBatchNorm2d()
                        )
                    num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
                    num_groups.stem: 1
                    width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
                    width_per_group: 64
                    in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
                    in_channels: 64
                    stage2_bottleneck_channels = num_groups * width_per_group
                    stage2_bottleneck_channels: 64
                    stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
                    stage2_out_channels: 256
                    self.stages = []
                    self.return_features = {}
                    for stage_spec in stage_specs {


                        --------------------------------------------------------


                        stage_spec: StageSpec(index=1, block_count=3, return_features=True)
                        stage_spec.index: 1


                        --------------------------------------------------------
                        name = "layer" + str(stage_spec.index)
                        name: layer1
                        stage2_relative_factor = 2 ** (stage_spec.index - 1)
                        stage2_relative_factor: 1
                        bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
                        bottlenec_channels: 64
                        out_channels = stage2_out_channels * stage2_relative_factor
                        out_channels: 256
                        stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]
                        stage_with_dcn: False
                        module = _make_stage(
                            transformation_module = <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>,
                            in_channels = 64,
                            bottleneck_channels = 64,
                            out_channels = 256,
                            stage_spec.block_count = 3,
                            num_groups = 1,
                            cfg.MODEL.RESNETS.STRIDE_IN_1X1 : True,
                            first_stride=int(stage_spec.index > 1) + 1: 1,
                            dcn_config={
                                'stage_with_dcn': False,
                                'with_modulated_dcn': False,
                                'deformable_groups': 1,
                                }
                            )
                        in_channels = out_channels
                        in_channels: 256
                        self.add_module(name=layer1, module)
                        self.stages.append(name=layer1)
                        stage_spec.return_features: True
                        self.return_features[name] = stage_spec.return_features


                        --------------------------------------------------------


                        stage_spec: StageSpec(index=2, block_count=4, return_features=True)
                        stage_spec.index: 2


                    --------------------------------------------------------
            name = "layer" + str(stage_spec.index)
            name: layer2
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            stage2_relative_factor: 2
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            bottlenec_channels: 128
            out_channels = stage2_out_channels * stage2_relative_factor
            out_channels: 512
            stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]
            stage_with_dcn: False
            module = _make_stage(
                    transformation_module = <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>,
                    in_channels = 256,
                    bottleneck_channels = 128,
                    out_channels = 512,
                    stage_spec.block_count = 4,
                    num_groups = 1,
                    cfg.MODEL.RESNETS.STRIDE_IN_1X1 : True,
                    first_stride=int(stage_spec.index > 1) + 1: 2,
                    dcn_config={
                        'stage_with_dcn': False,
                        'with_modulated_dcn': False,
                        'deformable_groups': 1,
                        }
                    )
            in_channels = out_channels
            in_channels: 512
            self.add_module(name=layer2, module)
            self.stages.append(name=layer2)
            stage_spec.return_features: True
            self.return_features[name] = stage_spec.return_features


            --------------------------------------------------------


            stage_spec: StageSpec(index=3, block_count=6, return_features=True)
            stage_spec.index: 3


            --------------------------------------------------------
            name = "layer" + str(stage_spec.index)
            name: layer3
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            stage2_relative_factor: 4
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            bottlenec_channels: 256
            out_channels = stage2_out_channels * stage2_relative_factor
            out_channels: 1024
            stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]
            stage_with_dcn: False
            module = _make_stage(
                    transformation_module = <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>,
                    in_channels = 512,
                    bottleneck_channels = 256,
                    out_channels = 1024,
                    stage_spec.block_count = 6,
                    num_groups = 1,
                    cfg.MODEL.RESNETS.STRIDE_IN_1X1 : True,
                    first_stride=int(stage_spec.index > 1) + 1: 2,
                    dcn_config={
                        'stage_with_dcn': False,
                        'with_modulated_dcn': False,
                        'deformable_groups': 1,
                        }
                    )
            in_channels = out_channels
            in_channels: 1024
            self.add_module(name=layer3, module)
            self.stages.append(name=layer3)
            stage_spec.return_features: True
            self.return_features[name] = stage_spec.return_features


            --------------------------------------------------------


            stage_spec: StageSpec(index=4, block_count=3, return_features=True)
            stage_spec.index: 4


            --------------------------------------------------------
            name = "layer" + str(stage_spec.index)
            name: layer4
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            stage2_relative_factor: 8
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            bottlenec_channels: 512
            out_channels = stage2_out_channels * stage2_relative_factor
            out_channels: 2048
            stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]
            stage_with_dcn: False
            module = _make_stage(
                    transformation_module = <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>,
                    in_channels = 1024,
                    bottleneck_channels = 512,
                    out_channels = 2048,
                    stage_spec.block_count = 3,
                    num_groups = 1,
                    cfg.MODEL.RESNETS.STRIDE_IN_1X1 : True,
                    first_stride=int(stage_spec.index > 1) + 1: 2,
                    dcn_config={
                        'stage_with_dcn': False,
                        'with_modulated_dcn': False,
                        'deformable_groups': 1,
                        }
                    )
            in_channels = out_channels
            in_channels: 2048
            self.add_module(name=layer4, module)
            self.stages.append(name=layer4)
            stage_spec.return_features: True
            self.return_features[name] = stage_spec.return_features
} // END for stage_spec in stage_specs:
    cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT: 2)
            self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    Resnet.__freeze_backbone(self, freeze_at) { // BEGIN

            Params:

            freeze_at: 2
            } // END Resnet.__freeze_backbone(self, freeze_at)

    } // END Resnet.__init__ END

        body = resnet.ResNet(cfg) // RETURNED
        cfg.MODEL.RESNETS.RES2_OUT_CHANNELS: 256
        in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        in_channels_stage2 = 256
        cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS:1024
        out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        out_channels = 1024
        in_channels_stage2: 256
        out_channels: 1024
        cfg.MODEL.RETINANET.USE_C5: True
        in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 else out_channels
        in_channels_p6p7 = 2048

        fpn = fpn_module.FPN(

                in_channels_list = [0, 512, 1024, 2048],

                out_channels = 1024, 

                conv_block=conv_with_kaiming_uniform( cfg.MODEL.FPN.USE_GN =False, cfg.MODEL.FPN.USE_RELU =False ),

                top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7=2048, out_channels=1024,) // CALL

                conv_with_kaiming_uniform(use_gn=False, use_relut=False) { //BEGIN
                    // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
                    } // END conv_with_kaiming_uniform(use_gn=False, use_relu=False)



                LastLevelP6P7.__init__(self, in_channels, out_channels) { //BEGIN
                    Param:
                    in_channels: 2048
                    out_channels: 1024
                    super(LastLevelP6P7, self).__init__()
                    self.p6 = nn.Conv2d(in_channels=2048, out_channels=1024, 3, 2, 1)
                    self.p7 = nn.Conv2d(out_channels=1024, out_channels=1024, 3, 2, 1)
                    for module in [self.p6, self.p7] {
                        module=Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                        nn.init.kaiming_uniform_(module.weight=module.weight, a=1)
                        nn.init.constant_(module.bias=module.bias, 0)
                        module=Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                        nn.init.kaiming_uniform_(module.weight=module.weight, a=1)
                        nn.init.constant_(module.bias=module.bias, 0)
                        } // END for module in [self.p6, self.p7]
                    self.use_p5 : False

                    } // END LastLevelP6P7.__init__(self, in_channels, out_channels)




                FPN.__init__ { // BEGIN
                    defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/backbone/fpn.py
                    Params
                    in_channels_list: [0, 512, 1024, 2048]
                    out_channels: 1024
                    conv_block: <function conv_with_kaiming_uniform.<locals>.make_conv at 0x7f492f576268>
                    top_blocks: LastLevelP6P7(
                        (p6): Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                        (p7): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                        )
                    super(FPN, self).__init__()

                    for idx, in_channels in enumerate(in_channels_list, 1) {

                        -----------------------------------------------------

                        iteration with idx:1, in_channels:0

                        -----------------------------------------------------
                        if in_channels ==0, skip


                        -----------------------------------------------------

                        iteration with idx:2, in_channels:512

                        -----------------------------------------------------
                        inner_block: fpn_inner2
                        layer_block: fpn_layer2
                        inner_block_module = conv_block(in_channels=512, out_channels=1024, 1)

                        make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
                            // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
                            Param
                            in_channels: 512
                            out_channels: 1024
                            kernel_size: 1
                            stride: 1
                            dilation: 1
                            conv = Conv2d(in_channles=512, out_channels=1024, kernel_size=1, stride=1
                                padding=0, dilation=1, bias=True, )
                            conv: Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
                            nn.init.kaiming_uniform_(conv.weight, a=1)
                            if not use_gn:
                            nn.init.constant_(conv.bias, 0)
                            module = [conv,]
                            module: [Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))]
                            conv: {conv}
                            return conv

                            } // END conv_with_kaiming_uniform().make_conv()

                        layer_block_module = conv_block(out_channels=1024, out_channels=1024, 3,1)

                        make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
                            // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
                            Param
                            in_channels: 1024
                            out_channels: 1024
                            kernel_size: 3
                            stride: 1
                            dilation: 1
                            conv = Conv2d(in_channles=1024, out_channels=1024, kernel_size=3, stride=1
                                padding=1, dilation=1, bias=True, )
                            conv: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            nn.init.kaiming_uniform_(conv.weight, a=1)
                            if not use_gn:
                            nn.init.constant_(conv.bias, 0)
                            module = [conv,]
                            module: [Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
                            conv: {conv}
                            return conv

                            } // END conv_with_kaiming_uniform().make_conv()

                        self.add_module(fpn_inner2, inner_block_module)
        self.add_module(fpn_layer2, layer_block_module)
        self.inner_blocks.append(fpn_inner2)
        self.layer_blocks.append(fpn_layer2)

        -----------------------------------------------------

        iteration with idx:3, in_channels:1024

        -----------------------------------------------------
        inner_block: fpn_inner3
        layer_block: fpn_layer3
        inner_block_module = conv_block(in_channels=1024, out_channels=1024, 1)

                make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
                        // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
                        Param
                        in_channels: 1024
                        out_channels: 1024
                        kernel_size: 1
                        stride: 1
                        dilation: 1
                        conv = Conv2d(in_channles=1024, out_channels=1024, kernel_size=1, stride=1
                            padding=0, dilation=1, bias=True, )
                        conv: Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
                        nn.init.kaiming_uniform_(conv.weight, a=1)
                        if not use_gn:
                        nn.init.constant_(conv.bias, 0)
                        module = [conv,]
                        module: [Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))]
                        conv: {conv}
                        return conv

                        } // END conv_with_kaiming_uniform().make_conv()

                layer_block_module = conv_block(out_channels=1024, out_channels=1024, 3,1)

                make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
                        // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
                        Param
                        in_channels: 1024
                        out_channels: 1024
                        kernel_size: 3
                        stride: 1
                        dilation: 1
                        conv = Conv2d(in_channles=1024, out_channels=1024, kernel_size=3, stride=1
                            padding=1, dilation=1, bias=True, )
                        conv: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                        nn.init.kaiming_uniform_(conv.weight, a=1)
                        if not use_gn:
                        nn.init.constant_(conv.bias, 0)
                        module = [conv,]
                        module: [Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
                        conv: {conv}
                        return conv

                        } // END conv_with_kaiming_uniform().make_conv()

                self.add_module(fpn_inner3, inner_block_module)
        self.add_module(fpn_layer3, layer_block_module)
        self.inner_blocks.append(fpn_inner3)
        self.layer_blocks.append(fpn_layer3)

        -----------------------------------------------------

        iteration with idx:4, in_channels:2048

        -----------------------------------------------------
        inner_block: fpn_inner4
        layer_block: fpn_layer4
        inner_block_module = conv_block(in_channels=2048, out_channels=1024, 1)

                make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
                        // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
                        Param
                        in_channels: 2048
                        out_channels: 1024
                        kernel_size: 1
                        stride: 1
                        dilation: 1
                        conv = Conv2d(in_channles=2048, out_channels=1024, kernel_size=1, stride=1
                            padding=0, dilation=1, bias=True, )
                        conv: Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))
                        nn.init.kaiming_uniform_(conv.weight, a=1)
                        if not use_gn:
                        nn.init.constant_(conv.bias, 0)
                        module = [conv,]
                        module: [Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))]
                        conv: {conv}
                        return conv

                        } // END conv_with_kaiming_uniform().make_conv()

                layer_block_module = conv_block(out_channels=1024, out_channels=1024, 3,1)

                make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
                        // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
                        Param
                        in_channels: 1024
                        out_channels: 1024
                        kernel_size: 3
                        stride: 1
                        dilation: 1
                        conv = Conv2d(in_channles=1024, out_channels=1024, kernel_size=3, stride=1
                            padding=1, dilation=1, bias=True, )
                        conv: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                        nn.init.kaiming_uniform_(conv.weight, a=1)
                        if not use_gn:
                        nn.init.constant_(conv.bias, 0)
                        module = [conv,]
                        module: [Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
                        conv: {conv}
                        return conv

                        } // END conv_with_kaiming_uniform().make_conv()

                self.add_module(fpn_inner4, inner_block_module)
        self.add_module(fpn_layer4, layer_block_module)
        self.inner_blocks.append(fpn_inner4)
        self.layer_blocks.append(fpn_layer4)
    } // END for idx, in_channels in enumerate(in_channels_list, 1)
    self.top_blocks = top_blocks

    self.inner_blocks: ['fpn_inner2', 'fpn_inner3', 'fpn_inner4']
        self.fpn_inner2: Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.fpn_inner3: Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.fpn_inner4: Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))

    self.layer_blocks: ['fpn_layer2', 'fpn_layer3', 'fpn_layer4']
        self.fpn_layer2: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fpn_layer3: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fpn_layer4: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    self.top_blocks: LastLevelP6P7(
            (p6): Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            (p7): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            )
    } // END FPN.__init__



        fpn = fpn_module.FPN(

                in_channels_list = [0, 512, 1024, 2048],

                out_channels = 1024, 

                conv_block=conv_with_kaiming_uniform( cfg.MODEL.FPN.USE_GN =False, cfg.MODEL.FPN.USE_RELU =False ),

                top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7=2048, out_channels=1024,) // RETURNED
                fpn: {fpn}
                model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)])) // CALL
                model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)])) // RETURNED
                model: Sequential(
                    (body): ResNet(
                        (stem): StemWithFixedBatchNorm(
                            (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                            (bn1): FrozenBatchNorm2d()
                            )
                        (layer1): Sequential(
                            (0): BottleneckWithFixedBatchNorm(
                                (downsample): Sequential(
                                    (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (1): FrozenBatchNorm2d()
                                    )
                                (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn1): FrozenBatchNorm2d()
                                (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                (bn2): FrozenBatchNorm2d()
                                (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn3): FrozenBatchNorm2d()
                                )
                            (1): BottleneckWithFixedBatchNorm(
                                (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn1): FrozenBatchNorm2d()
                                (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                (bn2): FrozenBatchNorm2d()
                                (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn3): FrozenBatchNorm2d()
                                )
                            (2): BottleneckWithFixedBatchNorm(
                                (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn1): FrozenBatchNorm2d()
                                (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                (bn2): FrozenBatchNorm2d()
                                (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn3): FrozenBatchNorm2d()
                                )
                            )
                        (layer2): Sequential(
                            (0): BottleneckWithFixedBatchNorm(
                                (downsample): Sequential(
                                    (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                                    (1): FrozenBatchNorm2d()
                                    )
                                (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
                                (bn1): FrozenBatchNorm2d()
                                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                (bn2): FrozenBatchNorm2d()
                                (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn3): FrozenBatchNorm2d()
                                )
                            (1): BottleneckWithFixedBatchNorm(
                                (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn1): FrozenBatchNorm2d()
                                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                (bn2): FrozenBatchNorm2d()
                                (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn3): FrozenBatchNorm2d()
                                )
                            (2): BottleneckWithFixedBatchNorm(
                                (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn1): FrozenBatchNorm2d()
                                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                (bn2): FrozenBatchNorm2d()
                                (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn3): FrozenBatchNorm2d()
                                )
                            (3): BottleneckWithFixedBatchNorm(
                                (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn1): FrozenBatchNorm2d()
                                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                (bn2): FrozenBatchNorm2d()
                                (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn3): FrozenBatchNorm2d()
                                )
                            )
                        (layer3): Sequential(
                                (0): BottleneckWithFixedBatchNorm(
                                    (downsample): Sequential(
                                        (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
                                        (1): FrozenBatchNorm2d()
                                        )
                                    (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
                                    (bn1): FrozenBatchNorm2d()
                                    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                    (bn2): FrozenBatchNorm2d()
                                    (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn3): FrozenBatchNorm2d()
                                    )
                                (1): BottleneckWithFixedBatchNorm(
                                    (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn1): FrozenBatchNorm2d()
                                    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                    (bn2): FrozenBatchNorm2d()
                                    (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn3): FrozenBatchNorm2d()
                                    )
                                (2): BottleneckWithFixedBatchNorm(
                                    (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn1): FrozenBatchNorm2d()
                                    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                    (bn2): FrozenBatchNorm2d()
                                    (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn3): FrozenBatchNorm2d()
                                    )
                                (3): BottleneckWithFixedBatchNorm(
                                    (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn1): FrozenBatchNorm2d()
                                    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                    (bn2): FrozenBatchNorm2d()
                                    (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn3): FrozenBatchNorm2d()
                                    )
                                (4): BottleneckWithFixedBatchNorm(
                                    (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn1): FrozenBatchNorm2d()
                                    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                    (bn2): FrozenBatchNorm2d()
                                    (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn3): FrozenBatchNorm2d()
                                    )
                                (5): BottleneckWithFixedBatchNorm(
                                    (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn1): FrozenBatchNorm2d()
                                    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                    (bn2): FrozenBatchNorm2d()
                                    (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn3): FrozenBatchNorm2d()
                                    )
                                )
    (layer4): Sequential(
            (0): BottleneckWithFixedBatchNorm(
                (downsample): Sequential(
                    (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
                    (1): FrozenBatchNorm2d()
                    )
                (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
            (1): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
            (2): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
            )
    )
  (fpn): FPN(
          (fpn_inner2): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
          (fpn_layer2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (fpn_inner3): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (fpn_layer3): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (fpn_inner4): Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))
          (fpn_layer4): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (top_blocks): LastLevelP6P7(
              (p6): Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
              (p7): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
              )
          )
  )
        model.out_channels = out_channels
            model.out_channels: 1024
        return model
    } // END build_resnet_fpn_p3p7_backbone(cfg) 


    registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg): Sequential(
            (body): ResNet(
                (stem): StemWithFixedBatchNorm(
                    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                    (bn1): FrozenBatchNorm2d()
                    )
                (layer1): Sequential(
                    (0): BottleneckWithFixedBatchNorm(
                        (downsample): Sequential(
                            (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (1): FrozenBatchNorm2d()
                            )
                        (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (bn1): FrozenBatchNorm2d()
                        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                        (bn2): FrozenBatchNorm2d()
                        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (bn3): FrozenBatchNorm2d()
                        )
                    (1): BottleneckWithFixedBatchNorm(
                        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (bn1): FrozenBatchNorm2d()
                        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                        (bn2): FrozenBatchNorm2d()
                        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (bn3): FrozenBatchNorm2d()
                        )
                    (2): BottleneckWithFixedBatchNorm(
                        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (bn1): FrozenBatchNorm2d()
                        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                        (bn2): FrozenBatchNorm2d()
                        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (bn3): FrozenBatchNorm2d()
                        )
                    )
                (layer2): Sequential(
                    (0): BottleneckWithFixedBatchNorm(
                        (downsample): Sequential(
                            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                            (1): FrozenBatchNorm2d()
                            )
                        (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
                        (bn1): FrozenBatchNorm2d()
                        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                        (bn2): FrozenBatchNorm2d()
                        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (bn3): FrozenBatchNorm2d()
                        )
                    (1): BottleneckWithFixedBatchNorm(
                        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (bn1): FrozenBatchNorm2d()
                        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                        (bn2): FrozenBatchNorm2d()
                        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (bn3): FrozenBatchNorm2d()
                        )
                    (2): BottleneckWithFixedBatchNorm(
                        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (bn1): FrozenBatchNorm2d()
                        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                        (bn2): FrozenBatchNorm2d()
                        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (bn3): FrozenBatchNorm2d()
                        )
                    (3): BottleneckWithFixedBatchNorm(
                        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (bn1): FrozenBatchNorm2d()
                        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                        (bn2): FrozenBatchNorm2d()
                        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (bn3): FrozenBatchNorm2d()
                        )
                    )
                (layer3): Sequential(
                        (0): BottleneckWithFixedBatchNorm(
                            (downsample): Sequential(
                                (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
                                (1): FrozenBatchNorm2d()
                                )
                            (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
                            (bn1): FrozenBatchNorm2d()
                            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                            (bn2): FrozenBatchNorm2d()
                            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn3): FrozenBatchNorm2d()
                            )
                        (1): BottleneckWithFixedBatchNorm(
                            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn1): FrozenBatchNorm2d()
                            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                            (bn2): FrozenBatchNorm2d()
                            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn3): FrozenBatchNorm2d()
                            )
                        (2): BottleneckWithFixedBatchNorm(
                            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn1): FrozenBatchNorm2d()
                            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                            (bn2): FrozenBatchNorm2d()
                            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn3): FrozenBatchNorm2d()
                            )
                        (3): BottleneckWithFixedBatchNorm(
                            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn1): FrozenBatchNorm2d()
                            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                            (bn2): FrozenBatchNorm2d()
                            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn3): FrozenBatchNorm2d()
                            )
                        (4): BottleneckWithFixedBatchNorm(
                            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn1): FrozenBatchNorm2d()
                            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                            (bn2): FrozenBatchNorm2d()
                            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn3): FrozenBatchNorm2d()
                            )
                        (5): BottleneckWithFixedBatchNorm(
                            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn1): FrozenBatchNorm2d()
                            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                            (bn2): FrozenBatchNorm2d()
                            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn3): FrozenBatchNorm2d()
                            )
                        )
    (layer4): Sequential(
            (0): BottleneckWithFixedBatchNorm(
                (downsample): Sequential(
                    (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
                    (1): FrozenBatchNorm2d()
                    )
                (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
            (1): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
            (2): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
            )
    )
  (fpn): FPN(
          (fpn_inner2): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
          (fpn_layer2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (fpn_inner3): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (fpn_layer3): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (fpn_inner4): Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))
          (fpn_layer4): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (top_blocks): LastLevelP6P7(
              (p6): Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
              (p7): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
              )
          )
  )
        return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
    } // END build_backbone(cfg)


    build_resnet_fpn_p3p7_backbone(cfg) { // BEGIN
            // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/backbone/backbone.py

            Params:
            cfg:
            body = resnet.ResNet(cfg) // CALL

            Resnet.__init__ { //BEGIN
                // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/backbone/resnet.py

                Params:
                cfg:
                _STEM_MODULES: {'StemWithFixedBatchNorm': <class 'maskrcnn_benchmark.modeling.backbone.resnet.StemWithFixedBatchNorm'>}
                _cfg.MODEL.RESNETS.STEM_FUNC: StemWithFixedBatchNorm
                stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
                stem_module: <class 'maskrcnn_benchmark.modeling.backbone.resnet.StemWithFixedBatchNorm'>
                cfg.MODEL.BACKBONE.CONV_BODY: R-50-FPN-RETINANET
                stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY=R-50-FPN-RETINANET]
                stage_specs: (StageSpec(index=1, block_count=3, return_features=True), StageSpec(index=2, block_count=4, return_features=True), StageSpec(index=3, block_count=6, return_features=True), StageSpec(index=4, block_count=3, return_features=True))
                _TRANSFORMATION_MODULES: {'BottleneckWithFixedBatchNorm': <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>}
                cfg.MODEL.RESNETS.TRANS_FUNC: BottleneckWithFixedBatchNorm
                transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC=BottleneckWithFixedBatchNorm]
                transformation_module: <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>
                self.stem = stem_module(cfg)
                self.stem: StemWithFixedBatchNorm(
                    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                    (bn1): FrozenBatchNorm2d()
                    )
                num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
                num_groups.stem: 1
                width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
                width_per_group: 64
                in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
                in_channels: 64
                stage2_bottleneck_channels = num_groups * width_per_group
                stage2_bottleneck_channels: 64
                stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
                stage2_out_channels: 256
                self.stages = []
                self.return_features = {}
                for stage_spec in stage_specs {


                    --------------------------------------------------------


                    stage_spec: StageSpec(index=1, block_count=3, return_features=True)
                    stage_spec.index: 1


                    --------------------------------------------------------
                    name = "layer" + str(stage_spec.index)
                    name: layer1
                    stage2_relative_factor = 2 ** (stage_spec.index - 1)
                    stage2_relative_factor: 1
                    bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
                    bottlenec_channels: 64
                    out_channels = stage2_out_channels * stage2_relative_factor
                    out_channels: 256
                    stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]
                    stage_with_dcn: False
                    module = _make_stage(
                        transformation_module = <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>,
                        in_channels = 64,
                        bottleneck_channels = 64,
                        out_channels = 256,
                        stage_spec.block_count = 3,
                        num_groups = 1,
                        cfg.MODEL.RESNETS.STRIDE_IN_1X1 : True,
                        first_stride=int(stage_spec.index > 1) + 1: 1,
                        dcn_config={
                            'stage_with_dcn': False,
                            'with_modulated_dcn': False,
                            'deformable_groups': 1,
                            }
                        )
                    in_channels = out_channels
                    in_channels: 256
                    self.add_module(name=layer1, module)
                    self.stages.append(name=layer1)
                    stage_spec.return_features: True
                    self.return_features[name] = stage_spec.return_features


                    --------------------------------------------------------


                    stage_spec: StageSpec(index=2, block_count=4, return_features=True)
                    stage_spec.index: 2


                --------------------------------------------------------
            name = "layer" + str(stage_spec.index)
            name: layer2
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            stage2_relative_factor: 2
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            bottlenec_channels: 128
            out_channels = stage2_out_channels * stage2_relative_factor
            out_channels: 512
            stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]
            stage_with_dcn: False
            module = _make_stage(
                    transformation_module = <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>,
                    in_channels = 256,
                    bottleneck_channels = 128,
                    out_channels = 512,
                    stage_spec.block_count = 4,
                    num_groups = 1,
                    cfg.MODEL.RESNETS.STRIDE_IN_1X1 : True,
                    first_stride=int(stage_spec.index > 1) + 1: 2,
                    dcn_config={
                        'stage_with_dcn': False,
                        'with_modulated_dcn': False,
                        'deformable_groups': 1,
                        }
                    )
            in_channels = out_channels
            in_channels: 512
            self.add_module(name=layer2, module)
            self.stages.append(name=layer2)
            stage_spec.return_features: True
            self.return_features[name] = stage_spec.return_features


            --------------------------------------------------------


            stage_spec: StageSpec(index=3, block_count=6, return_features=True)
            stage_spec.index: 3


            --------------------------------------------------------
            name = "layer" + str(stage_spec.index)
            name: layer3
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            stage2_relative_factor: 4
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            bottlenec_channels: 256
            out_channels = stage2_out_channels * stage2_relative_factor
            out_channels: 1024
            stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]
            stage_with_dcn: False
            module = _make_stage(
                    transformation_module = <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>,
                    in_channels = 512,
                    bottleneck_channels = 256,
                    out_channels = 1024,
                    stage_spec.block_count = 6,
                    num_groups = 1,
                    cfg.MODEL.RESNETS.STRIDE_IN_1X1 : True,
                    first_stride=int(stage_spec.index > 1) + 1: 2,
                    dcn_config={
                        'stage_with_dcn': False,
                        'with_modulated_dcn': False,
                        'deformable_groups': 1,
                        }
                    )
            in_channels = out_channels
            in_channels: 1024
            self.add_module(name=layer3, module)
            self.stages.append(name=layer3)
            stage_spec.return_features: True
            self.return_features[name] = stage_spec.return_features


            --------------------------------------------------------


            stage_spec: StageSpec(index=4, block_count=3, return_features=True)
            stage_spec.index: 4


            --------------------------------------------------------
            name = "layer" + str(stage_spec.index)
            name: layer4
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            stage2_relative_factor: 8
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            bottlenec_channels: 512
            out_channels = stage2_out_channels * stage2_relative_factor
            out_channels: 2048
            stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]
            stage_with_dcn: False
            module = _make_stage(
                    transformation_module = <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>,
                    in_channels = 1024,
                    bottleneck_channels = 512,
                    out_channels = 2048,
                    stage_spec.block_count = 3,
                    num_groups = 1,
                    cfg.MODEL.RESNETS.STRIDE_IN_1X1 : True,
                    first_stride=int(stage_spec.index > 1) + 1: 2,
                    dcn_config={
                        'stage_with_dcn': False,
                        'with_modulated_dcn': False,
                        'deformable_groups': 1,
                        }
                    )
            in_channels = out_channels
            in_channels: 2048
            self.add_module(name=layer4, module)
            self.stages.append(name=layer4)
            stage_spec.return_features: True
            self.return_features[name] = stage_spec.return_features
} // END for stage_spec in stage_specs:
    cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT: 2)
            self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    Resnet.__freeze_backbone(self, freeze_at) { // BEGIN

            Params:

            freeze_at: 2
            } // END Resnet.__freeze_backbone(self, freeze_at)

    } // END Resnet.__init__ END

        body = resnet.ResNet(cfg) // RETURNED
        cfg.MODEL.RESNETS.RES2_OUT_CHANNELS: 256
        in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        in_channels_stage2 = 256
        cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS:1024
        out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        out_channels = 1024
        in_channels_stage2: 256
        out_channels: 1024
        cfg.MODEL.RETINANET.USE_C5: True
        in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 else out_channels
        in_channels_p6p7 = 2048

        fpn = fpn_module.FPN(

                in_channels_list = [0, 512, 1024, 2048],

                out_channels = 1024, 

                conv_block=conv_with_kaiming_uniform( cfg.MODEL.FPN.USE_GN =False, cfg.MODEL.FPN.USE_RELU =False ),

                top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7=2048, out_channels=1024,) // CALL

                conv_with_kaiming_uniform(use_gn=False, use_relut=False) { //BEGIN
                    // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
                    } // END conv_with_kaiming_uniform(use_gn=False, use_relu=False)



                LastLevelP6P7.__init__(self, in_channels, out_channels) { //BEGIN
                    Param:
                    in_channels: 2048
                    out_channels: 1024
                    super(LastLevelP6P7, self).__init__()
                    self.p6 = nn.Conv2d(in_channels=2048, out_channels=1024, 3, 2, 1)
                    self.p7 = nn.Conv2d(out_channels=1024, out_channels=1024, 3, 2, 1)
                    for module in [self.p6, self.p7] {
                        module=Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                        nn.init.kaiming_uniform_(module.weight=module.weight, a=1)
                        nn.init.constant_(module.bias=module.bias, 0)
                        module=Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                        nn.init.kaiming_uniform_(module.weight=module.weight, a=1)
                        nn.init.constant_(module.bias=module.bias, 0)
                        } // END for module in [self.p6, self.p7]
                    self.use_p5 : False

                    } // END LastLevelP6P7.__init__(self, in_channels, out_channels)




                FPN.__init__ { // BEGIN
                    defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/backbone/fpn.py
                    Params
                    in_channels_list: [0, 512, 1024, 2048]
                    out_channels: 1024
                    conv_block: <function conv_with_kaiming_uniform.<locals>.make_conv at 0x7f492f579d90>
                    top_blocks: LastLevelP6P7(
                        (p6): Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                        (p7): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                        )
                    super(FPN, self).__init__()

                    for idx, in_channels in enumerate(in_channels_list, 1) {

                        -----------------------------------------------------

                        iteration with idx:1, in_channels:0

                        -----------------------------------------------------
                        if in_channels ==0, skip


                        -----------------------------------------------------

                        iteration with idx:2, in_channels:512

                        -----------------------------------------------------
                        inner_block: fpn_inner2
                        layer_block: fpn_layer2
                        inner_block_module = conv_block(in_channels=512, out_channels=1024, 1)

                        make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
                            // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
                            Param
                            in_channels: 512
                            out_channels: 1024
                            kernel_size: 1
                            stride: 1
                            dilation: 1
                            conv = Conv2d(in_channles=512, out_channels=1024, kernel_size=1, stride=1
                                padding=0, dilation=1, bias=True, )
                            conv: Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
                            nn.init.kaiming_uniform_(conv.weight, a=1)
                            if not use_gn:
                            nn.init.constant_(conv.bias, 0)
                            module = [conv,]
                            module: [Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))]
                            conv: {conv}
                            return conv

                            } // END conv_with_kaiming_uniform().make_conv()

                        layer_block_module = conv_block(out_channels=1024, out_channels=1024, 3,1)

                        make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
                            // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
                            Param
                            in_channels: 1024
                            out_channels: 1024
                            kernel_size: 3
                            stride: 1
                            dilation: 1
                            conv = Conv2d(in_channles=1024, out_channels=1024, kernel_size=3, stride=1
                                padding=1, dilation=1, bias=True, )
                            conv: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            nn.init.kaiming_uniform_(conv.weight, a=1)
                            if not use_gn:
                            nn.init.constant_(conv.bias, 0)
                            module = [conv,]
                            module: [Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
                            conv: {conv}
                            return conv

                            } // END conv_with_kaiming_uniform().make_conv()

                        self.add_module(fpn_inner2, inner_block_module)
        self.add_module(fpn_layer2, layer_block_module)
        self.inner_blocks.append(fpn_inner2)
        self.layer_blocks.append(fpn_layer2)

        -----------------------------------------------------

        iteration with idx:3, in_channels:1024

        -----------------------------------------------------
        inner_block: fpn_inner3
        layer_block: fpn_layer3
        inner_block_module = conv_block(in_channels=1024, out_channels=1024, 1)

                make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
                        // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
                        Param
                        in_channels: 1024
                        out_channels: 1024
                        kernel_size: 1
                        stride: 1
                        dilation: 1
                        conv = Conv2d(in_channles=1024, out_channels=1024, kernel_size=1, stride=1
                            padding=0, dilation=1, bias=True, )
                        conv: Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
                        nn.init.kaiming_uniform_(conv.weight, a=1)
                        if not use_gn:
                        nn.init.constant_(conv.bias, 0)
                        module = [conv,]
                        module: [Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))]
                        conv: {conv}
                        return conv

                        } // END conv_with_kaiming_uniform().make_conv()

                layer_block_module = conv_block(out_channels=1024, out_channels=1024, 3,1)

                make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
                        // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
                        Param
                        in_channels: 1024
                        out_channels: 1024
                        kernel_size: 3
                        stride: 1
                        dilation: 1
                        conv = Conv2d(in_channles=1024, out_channels=1024, kernel_size=3, stride=1
                            padding=1, dilation=1, bias=True, )
                        conv: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                        nn.init.kaiming_uniform_(conv.weight, a=1)
                        if not use_gn:
                        nn.init.constant_(conv.bias, 0)
                        module = [conv,]
                        module: [Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
                        conv: {conv}
                        return conv

                        } // END conv_with_kaiming_uniform().make_conv()

                self.add_module(fpn_inner3, inner_block_module)
        self.add_module(fpn_layer3, layer_block_module)
        self.inner_blocks.append(fpn_inner3)
        self.layer_blocks.append(fpn_layer3)

        -----------------------------------------------------

        iteration with idx:4, in_channels:2048

        -----------------------------------------------------
        inner_block: fpn_inner4
        layer_block: fpn_layer4
        inner_block_module = conv_block(in_channels=2048, out_channels=1024, 1)

                make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
                        // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
                        Param
                        in_channels: 2048
                        out_channels: 1024
                        kernel_size: 1
                        stride: 1
                        dilation: 1
                        conv = Conv2d(in_channles=2048, out_channels=1024, kernel_size=1, stride=1
                            padding=0, dilation=1, bias=True, )
                        conv: Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))
                        nn.init.kaiming_uniform_(conv.weight, a=1)
                        if not use_gn:
                        nn.init.constant_(conv.bias, 0)
                        module = [conv,]
                        module: [Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))]
                        conv: {conv}
                        return conv

                        } // END conv_with_kaiming_uniform().make_conv()

                layer_block_module = conv_block(out_channels=1024, out_channels=1024, 3,1)

                make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
                        // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
                        Param
                        in_channels: 1024
                        out_channels: 1024
                        kernel_size: 3
                        stride: 1
                        dilation: 1
                        conv = Conv2d(in_channles=1024, out_channels=1024, kernel_size=3, stride=1
                            padding=1, dilation=1, bias=True, )
                        conv: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                        nn.init.kaiming_uniform_(conv.weight, a=1)
                        if not use_gn:
                        nn.init.constant_(conv.bias, 0)
                        module = [conv,]
                        module: [Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
                        conv: {conv}
                        return conv

                        } // END conv_with_kaiming_uniform().make_conv()

                self.add_module(fpn_inner4, inner_block_module)
        self.add_module(fpn_layer4, layer_block_module)
        self.inner_blocks.append(fpn_inner4)
        self.layer_blocks.append(fpn_layer4)
    } // END for idx, in_channels in enumerate(in_channels_list, 1)
    self.top_blocks = top_blocks

    self.inner_blocks: ['fpn_inner2', 'fpn_inner3', 'fpn_inner4']
        self.fpn_inner2: Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.fpn_inner3: Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.fpn_inner4: Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))

    self.layer_blocks: ['fpn_layer2', 'fpn_layer3', 'fpn_layer4']
        self.fpn_layer2: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fpn_layer3: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fpn_layer4: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    self.top_blocks: LastLevelP6P7(
            (p6): Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            (p7): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            )
    } // END FPN.__init__



        fpn = fpn_module.FPN(

                in_channels_list = [0, 512, 1024, 2048],

                out_channels = 1024, 

                conv_block=conv_with_kaiming_uniform( cfg.MODEL.FPN.USE_GN =False, cfg.MODEL.FPN.USE_RELU =False ),

                top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7=2048, out_channels=1024,) // RETURNED
                fpn: {fpn}
                model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)])) // CALL
                model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)])) // RETURNED
                model: Sequential(
                    (body): ResNet(
                        (stem): StemWithFixedBatchNorm(
                            (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                            (bn1): FrozenBatchNorm2d()
                            )
                        (layer1): Sequential(
                            (0): BottleneckWithFixedBatchNorm(
                                (downsample): Sequential(
                                    (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (1): FrozenBatchNorm2d()
                                    )
                                (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn1): FrozenBatchNorm2d()
                                (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                (bn2): FrozenBatchNorm2d()
                                (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn3): FrozenBatchNorm2d()
                                )
                            (1): BottleneckWithFixedBatchNorm(
                                (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn1): FrozenBatchNorm2d()
                                (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                (bn2): FrozenBatchNorm2d()
                                (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn3): FrozenBatchNorm2d()
                                )
                            (2): BottleneckWithFixedBatchNorm(
                                (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn1): FrozenBatchNorm2d()
                                (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                (bn2): FrozenBatchNorm2d()
                                (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn3): FrozenBatchNorm2d()
                                )
                            )
                        (layer2): Sequential(
                            (0): BottleneckWithFixedBatchNorm(
                                (downsample): Sequential(
                                    (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                                    (1): FrozenBatchNorm2d()
                                    )
                                (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
                                (bn1): FrozenBatchNorm2d()
                                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                (bn2): FrozenBatchNorm2d()
                                (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn3): FrozenBatchNorm2d()
                                )
                            (1): BottleneckWithFixedBatchNorm(
                                (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn1): FrozenBatchNorm2d()
                                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                (bn2): FrozenBatchNorm2d()
                                (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn3): FrozenBatchNorm2d()
                                )
                            (2): BottleneckWithFixedBatchNorm(
                                (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn1): FrozenBatchNorm2d()
                                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                (bn2): FrozenBatchNorm2d()
                                (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn3): FrozenBatchNorm2d()
                                )
                            (3): BottleneckWithFixedBatchNorm(
                                (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn1): FrozenBatchNorm2d()
                                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                (bn2): FrozenBatchNorm2d()
                                (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn3): FrozenBatchNorm2d()
                                )
                            )
                        (layer3): Sequential(
                                (0): BottleneckWithFixedBatchNorm(
                                    (downsample): Sequential(
                                        (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
                                        (1): FrozenBatchNorm2d()
                                        )
                                    (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
                                    (bn1): FrozenBatchNorm2d()
                                    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                    (bn2): FrozenBatchNorm2d()
                                    (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn3): FrozenBatchNorm2d()
                                    )
                                (1): BottleneckWithFixedBatchNorm(
                                    (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn1): FrozenBatchNorm2d()
                                    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                    (bn2): FrozenBatchNorm2d()
                                    (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn3): FrozenBatchNorm2d()
                                    )
                                (2): BottleneckWithFixedBatchNorm(
                                    (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn1): FrozenBatchNorm2d()
                                    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                    (bn2): FrozenBatchNorm2d()
                                    (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn3): FrozenBatchNorm2d()
                                    )
                                (3): BottleneckWithFixedBatchNorm(
                                    (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn1): FrozenBatchNorm2d()
                                    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                    (bn2): FrozenBatchNorm2d()
                                    (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn3): FrozenBatchNorm2d()
                                    )
                                (4): BottleneckWithFixedBatchNorm(
                                    (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn1): FrozenBatchNorm2d()
                                    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                    (bn2): FrozenBatchNorm2d()
                                    (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn3): FrozenBatchNorm2d()
                                    )
                                (5): BottleneckWithFixedBatchNorm(
                                    (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn1): FrozenBatchNorm2d()
                                    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                    (bn2): FrozenBatchNorm2d()
                                    (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                    (bn3): FrozenBatchNorm2d()
                                    )
                                )
    (layer4): Sequential(
            (0): BottleneckWithFixedBatchNorm(
                (downsample): Sequential(
                    (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
                    (1): FrozenBatchNorm2d()
                    )
                (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
            (1): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
            (2): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
            )
    )
  (fpn): FPN(
          (fpn_inner2): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
          (fpn_layer2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (fpn_inner3): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (fpn_layer3): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (fpn_inner4): Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))
          (fpn_layer4): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (top_blocks): LastLevelP6P7(
              (p6): Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
              (p7): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
              )
          )
  )
        model.out_channels = out_channels
            model.out_channels: 1024
        return model
    } // END build_resnet_fpn_p3p7_backbone(cfg) 


    self.backbone = build_backbone(cfg) // RETURNED
        self.backbone: Sequential(
                (body): ResNet(
                    (stem): StemWithFixedBatchNorm(
                        (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                        (bn1): FrozenBatchNorm2d()
                        )
                    (layer1): Sequential(
                        (0): BottleneckWithFixedBatchNorm(
                            (downsample): Sequential(
                                (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (1): FrozenBatchNorm2d()
                                )
                            (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn1): FrozenBatchNorm2d()
                            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                            (bn2): FrozenBatchNorm2d()
                            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn3): FrozenBatchNorm2d()
                            )
                        (1): BottleneckWithFixedBatchNorm(
                            (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn1): FrozenBatchNorm2d()
                            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                            (bn2): FrozenBatchNorm2d()
                            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn3): FrozenBatchNorm2d()
                            )
                        (2): BottleneckWithFixedBatchNorm(
                            (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn1): FrozenBatchNorm2d()
                            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                            (bn2): FrozenBatchNorm2d()
                            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn3): FrozenBatchNorm2d()
                            )
                        )
                    (layer2): Sequential(
                        (0): BottleneckWithFixedBatchNorm(
                            (downsample): Sequential(
                                (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                                (1): FrozenBatchNorm2d()
                                )
                            (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
                            (bn1): FrozenBatchNorm2d()
                            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                            (bn2): FrozenBatchNorm2d()
                            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn3): FrozenBatchNorm2d()
                            )
                        (1): BottleneckWithFixedBatchNorm(
                            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn1): FrozenBatchNorm2d()
                            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                            (bn2): FrozenBatchNorm2d()
                            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn3): FrozenBatchNorm2d()
                            )
                        (2): BottleneckWithFixedBatchNorm(
                            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn1): FrozenBatchNorm2d()
                            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                            (bn2): FrozenBatchNorm2d()
                            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn3): FrozenBatchNorm2d()
                            )
                        (3): BottleneckWithFixedBatchNorm(
                            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn1): FrozenBatchNorm2d()
                            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                            (bn2): FrozenBatchNorm2d()
                            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn3): FrozenBatchNorm2d()
                            )
                        )
                    (layer3): Sequential(
                            (0): BottleneckWithFixedBatchNorm(
                                (downsample): Sequential(
                                    (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
                                    (1): FrozenBatchNorm2d()
                                    )
                                (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
                                (bn1): FrozenBatchNorm2d()
                                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                (bn2): FrozenBatchNorm2d()
                                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn3): FrozenBatchNorm2d()
                                )
                            (1): BottleneckWithFixedBatchNorm(
                                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn1): FrozenBatchNorm2d()
                                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                (bn2): FrozenBatchNorm2d()
                                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn3): FrozenBatchNorm2d()
                                )
                            (2): BottleneckWithFixedBatchNorm(
                                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn1): FrozenBatchNorm2d()
                                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                (bn2): FrozenBatchNorm2d()
                                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn3): FrozenBatchNorm2d()
                                )
                            (3): BottleneckWithFixedBatchNorm(
                                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn1): FrozenBatchNorm2d()
                                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                (bn2): FrozenBatchNorm2d()
                                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn3): FrozenBatchNorm2d()
                                )
                            (4): BottleneckWithFixedBatchNorm(
                                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn1): FrozenBatchNorm2d()
                                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                (bn2): FrozenBatchNorm2d()
                                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn3): FrozenBatchNorm2d()
                                )
                            (5): BottleneckWithFixedBatchNorm(
                                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn1): FrozenBatchNorm2d()
                                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                (bn2): FrozenBatchNorm2d()
                                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                (bn3): FrozenBatchNorm2d()
                                )
                            )
    (layer4): Sequential(
            (0): BottleneckWithFixedBatchNorm(
                (downsample): Sequential(
                    (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
                    (1): FrozenBatchNorm2d()
                    )
                (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
            (1): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
            (2): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
            )
    )
  (fpn): FPN(
          (fpn_inner2): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
          (fpn_layer2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (fpn_inner3): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (fpn_layer3): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (fpn_inner4): Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))
          (fpn_layer4): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (top_blocks): LastLevelP6P7(
              (p6): Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
              (p7): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
              )
          )
  )
    self.backbone.out_channels: 1024
    self.rpn = build_rpn(cfg, self.backbone.out_channels) // CALL
    build_retinanet(cfg, in_channels) { // BEGIN
            // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/retinanet/retinanet.py
            Param:
            cfg:
            in_channels: 1024
            return RetinaNetModule(cfg, in_channels)


            RetinaNetModule.__init__(self, cfg, in_channels) { // BEGIN
                // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/retinanet/retinanet.py
                Params:
                cfg:
                in_channels: 1024

                super(RetinaNetModule, self).__init__()
                self.cfg = cfg.clone()
                anchor_generator = make_anchor_generator_retinanet(cfg)


                make_anchor_generator_retinanet(config) { // BEGIN
                    /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                    config params
                    anchor_sizes: (32, 64, 128, 256, 512)
                    aspect_ratios: (0.5, 1.0, 2.0)
                    anchor_strides: (8, 16, 32, 64, 128)
                    straddle_thresh: -1
                    octave: 2.0
                    scales_per_octave: 3
                    new_anchor_sizes = []

                    for size in anchor_sizes {
                        ----------- 
                        size: 32
                        ----------- 
                        per_layer_anchor_sizes = []

                        for scale_per_octave in range(scales_per_octave) { // BEGIN
                            ------------------------
                            scale_per_octave : 0
                            ------------------------
                            octave : 2.0

                            octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

                            octave_scale: 1.0
                            size: 32

                            per_layer_anchor_sizes.append(octave_scale * size)
                            per_layer_anchor_sizes: [32.0]

                            ------------------------
                            scale_per_octave : 1
                            ------------------------
                            octave : 2.0

                            octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

                            octave_scale: 1.2599210498948732
                            size: 32

                            per_layer_anchor_sizes.append(octave_scale * size)
                            per_layer_anchor_sizes: [32.0, 40.31747359663594]

                            ------------------------
                            scale_per_octave : 2
                            ------------------------
                            octave : 2.0

                            octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

                            octave_scale: 1.5874010519681994
                            size: 32

                            per_layer_anchor_sizes.append(octave_scale * size)
                            per_layer_anchor_sizes: [32.0, 40.31747359663594, 50.79683366298238]

                            } // EDN for scale_per_octave in range(scales_per_octave)


                        new_anchor_sizes.append(tuple(per_layer_anchor_sizes))
                        new_anchor_sizes: [(32.0, 40.31747359663594, 50.79683366298238)]
                    ----------- 
            size: 64
            ----------- 
            per_layer_anchor_sizes = []

            for scale_per_octave in range(scales_per_octave) { // BEGIN
                    ------------------------
                    scale_per_octave : 0
                    ------------------------
                    octave : 2.0

                    octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

                    octave_scale: 1.0
                    size: 64

                    per_layer_anchor_sizes.append(octave_scale * size)
                    per_layer_anchor_sizes: [64.0]

                    ------------------------
                    scale_per_octave : 1
                    ------------------------
                    octave : 2.0

                    octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

                    octave_scale: 1.2599210498948732
                    size: 64

                    per_layer_anchor_sizes.append(octave_scale * size)
                    per_layer_anchor_sizes: [64.0, 80.63494719327188]

                    ------------------------
                    scale_per_octave : 2
                    ------------------------
                    octave : 2.0

                    octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

                    octave_scale: 1.5874010519681994
                    size: 64

                    per_layer_anchor_sizes.append(octave_scale * size)
                    per_layer_anchor_sizes: [64.0, 80.63494719327188, 101.59366732596476]

                    } // EDN for scale_per_octave in range(scales_per_octave)


            new_anchor_sizes.append(tuple(per_layer_anchor_sizes))
            new_anchor_sizes: [(32.0, 40.31747359663594, 50.79683366298238), (64.0, 80.63494719327188, 101.59366732596476)]
            ----------- 
            size: 128
            ----------- 
            per_layer_anchor_sizes = []

            for scale_per_octave in range(scales_per_octave) { // BEGIN
                    ------------------------
                    scale_per_octave : 0
                    ------------------------
                    octave : 2.0

                    octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

                    octave_scale: 1.0
                    size: 128

                    per_layer_anchor_sizes.append(octave_scale * size)
                    per_layer_anchor_sizes: [128.0]

                    ------------------------
                    scale_per_octave : 1
                    ------------------------
                    octave : 2.0

                    octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

                    octave_scale: 1.2599210498948732
                    size: 128

                    per_layer_anchor_sizes.append(octave_scale * size)
                    per_layer_anchor_sizes: [128.0, 161.26989438654377]

                    ------------------------
                    scale_per_octave : 2
                    ------------------------
                    octave : 2.0

                    octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

                    octave_scale: 1.5874010519681994
                    size: 128

                    per_layer_anchor_sizes.append(octave_scale * size)
                    per_layer_anchor_sizes: [128.0, 161.26989438654377, 203.18733465192952]

                    } // EDN for scale_per_octave in range(scales_per_octave)


            new_anchor_sizes.append(tuple(per_layer_anchor_sizes))
            new_anchor_sizes: [(32.0, 40.31747359663594, 50.79683366298238), (64.0, 80.63494719327188, 101.59366732596476), (128.0, 161.26989438654377, 203.18733465192952)]
            ----------- 
            size: 256
            ----------- 
            per_layer_anchor_sizes = []

            for scale_per_octave in range(scales_per_octave) { // BEGIN
                    ------------------------
                    scale_per_octave : 0
                    ------------------------
                    octave : 2.0

                    octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

                    octave_scale: 1.0
                    size: 256

                    per_layer_anchor_sizes.append(octave_scale * size)
                    per_layer_anchor_sizes: [256.0]

                    ------------------------
                    scale_per_octave : 1
                    ------------------------
                    octave : 2.0

                    octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

                    octave_scale: 1.2599210498948732
                    size: 256

                    per_layer_anchor_sizes.append(octave_scale * size)
                    per_layer_anchor_sizes: [256.0, 322.53978877308754]

                    ------------------------
                    scale_per_octave : 2
                    ------------------------
                    octave : 2.0

                    octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

                    octave_scale: 1.5874010519681994
                    size: 256

                    per_layer_anchor_sizes.append(octave_scale * size)
                    per_layer_anchor_sizes: [256.0, 322.53978877308754, 406.37466930385904]

                    } // EDN for scale_per_octave in range(scales_per_octave)


            new_anchor_sizes.append(tuple(per_layer_anchor_sizes))
            new_anchor_sizes: [(32.0, 40.31747359663594, 50.79683366298238), (64.0, 80.63494719327188, 101.59366732596476), (128.0, 161.26989438654377, 203.18733465192952), (256.0, 322.53978877308754, 406.37466930385904)]
            ----------- 
            size: 512
            ----------- 
            per_layer_anchor_sizes = []

            for scale_per_octave in range(scales_per_octave) { // BEGIN
                    ------------------------
                    scale_per_octave : 0
                    ------------------------
                    octave : 2.0

                    octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

                    octave_scale: 1.0
                    size: 512

                    per_layer_anchor_sizes.append(octave_scale * size)
                    per_layer_anchor_sizes: [512.0]

                    ------------------------
                    scale_per_octave : 1
                    ------------------------
                    octave : 2.0

                    octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

                    octave_scale: 1.2599210498948732
                    size: 512

                    per_layer_anchor_sizes.append(octave_scale * size)
                    per_layer_anchor_sizes: [512.0, 645.0795775461751]

                    ------------------------
                    scale_per_octave : 2
                    ------------------------
                    octave : 2.0

                    octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

                    octave_scale: 1.5874010519681994
                    size: 512

                    per_layer_anchor_sizes.append(octave_scale * size)
                    per_layer_anchor_sizes: [512.0, 645.0795775461751, 812.7493386077181]

                    } // EDN for scale_per_octave in range(scales_per_octave)


            new_anchor_sizes.append(tuple(per_layer_anchor_sizes))
            new_anchor_sizes: [(32.0, 40.31747359663594, 50.79683366298238), (64.0, 80.63494719327188, 101.59366732596476), (128.0, 161.26989438654377, 203.18733465192952), (256.0, 322.53978877308754, 406.37466930385904), (512.0, 645.0795775461751, 812.7493386077181)]
        } // END for size in anchor_sizes

        new_anchor_sizes:
            [(32.0, 40.31747359663594, 50.79683366298238), (64.0, 80.63494719327188, 101.59366732596476), (128.0, 161.26989438654377, 203.18733465192952), (256.0, 322.53978877308754, 406.37466930385904), (512.0, 645.0795775461751, 812.7493386077181)]
        aspect_ratios:
            (0.5, 1.0, 2.0)
        anchor_strides:
            (8, 16, 32, 64, 128)
        straddle_thresh:
            -1
        anchor_generator = AnchorGenerator( tuple(new_anchor_sizes), aspect_ratios, anchor_strides, straddle_thresh )
        AnchorGenerator.__init__(sizes, aspect_ratios, anchor_strides, straddle_thresh) { //BEGIN
                // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                Params
                sizes: ((32.0, 40.31747359663594, 50.79683366298238), (64.0, 80.63494719327188, 101.59366732596476), (128.0, 161.26989438654377, 203.18733465192952), (256.0, 322.53978877308754, 406.37466930385904), (512.0, 645.0795775461751, 812.7493386077181))
                aspect_ratios: (0.5, 1.0, 2.0)
                anchor_strides: (8, 16, 32, 64, 128)
                straddle_thresh: -1
                else: i.e, len(anchor_strides) !=1
                anchor_stride = anchor_strides[0]
                len(anchor_strides):5, len(size): 5
                else: i.e, len(anchor_strides) == len(sizes)
                cell_anchors = [ generate_anchors( anchor_stride,
                    size if isinstance(size, (tuple, list)) else (size,), 
                    aspect_ratios).float()
                    for anchor_stride, size in zip(anchor_strides, sizes)
                    generate_anchors(stride, sizes, aspect_ratios) { //BEGIN
                        /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                        Params:
                        stride: 8
                        sizes: (32.0, 40.31747359663594, 50.79683366298238)
                        aspect_ratios: (0.5, 1.0, 2.0)
                        return _generate_anchors(stride,
                            np.array(sizes, dtype=np.float) / stride,
                            np.array(aspect_ratios, dtype=np.float),)
                        _generate_anchors(base_size, scales, aspect_ratios) { //BEGIN
                            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                            Params:
                            base_size: 8
                            scales: [4.         5.0396842  6.34960421]
                            aspect_ratios: [0.5 1.  2. ]
                            anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
                            anchor: [0. 0. 7. 7.]
                            anchors = _ratio_enum(anchor, aspect_ratios)
                            _ratio_enum(anchor, ratios) { //BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                anchor: [0. 0. 7. 7.]
                                ratios: [0.5 1.  2. ]
                                _whctrs(anchors) { //BEGIN
                                    /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                    Param:
                                    anchor: [0. 0. 7. 7.]
                                    w = anchor[2] - anchor[0] + 1
                                    w: 8.0
                                    h = anchor[3] - anchor[1] + 1
                                    h: 8.0
                                    x_ctr = anchor[0] + 0.5 * (w - 1)
                                    x_ctr: 3.5
                                    y_ctr = anchor[1] + 0.5 * (h - 1)
                                    y_ctr: 3.5
                                    return w, h, x_ctr, y_ctr
                                    } // END _whctrs(anchors)
                                w, h, x_ctr, y_ctr = _whctrs(anchor)
                                w: 8.0, h: 8.0, x_ctr: 3.5, y_ctr: 3.5
                                size = w * h
                                size: 64.0
                                size_ratios = size / ratios
                                size_ratios: [128.  64.  32.]
                                ws = np.round(np.sqrt(size_ratios))
                                ws: [11.  8.  6.]
                                hs = np.round(ws * ratios)
                                hs: [ 6.  8. 12.]
                                _mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
                                    /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                    Param:
                                    ws: [11.  8.  6.]
                                    hs: [ 6.  8. 12.]
                                    x_ctr: 3.5
                                    y_ctr: 3.5
                                    ws = ws[:, np.newaxis]
                                    ws: [[11.]
                                        [ 8.]
                                        [ 6.]]
                                    hs = hs[:, np.newaxis]
                                    hs: [[ 6.]
                                        [ 8.]
                                        [12.]]
                                    anchors = np.hstack(
                                        (
                                            x_ctr - 0.5 * (ws - 1),
                                            y_ctr - 0.5 * (hs - 1),
                                            x_ctr + 0.5 * (ws - 1),
                                            y_ctr + 0.5 * (hs - 1),
                                            )
                                        )
                                    anchors: [[-1.5  1.   8.5  6. ]
                                        [ 0.   0.   7.   7. ]
                                        [ 1.  -2.   6.   9. ]]
                                    return anchors
                                    } // END _mkanchors(ws, hs, x_ctr, y_ctr)
                                anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
                    anchors: [[-1.5  1.   8.5  6. ]
                            [ 0.   0.   7.   7. ]
                            [ 1.  -2.   6.   9. ]]
                    return anchors
                } // END _ratio_enum(anchor, ratios)
anchors: [[-1.5  1.   8.5  6. ]
        [ 0.   0.   7.   7. ]
        [ 1.  -2.   6.   9. ]]
anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
        )
_scale_enum(anchor, scales) { //BEGIN
        /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
        Param:
        anchor: [-1.5  1.   8.5  6. ]
        scales: [4.         5.0396842  6.34960421]
        w, h, x_ctr, y_ctr = _whctrs(anchor)
        _whctrs(anchors) { //BEGIN
            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
            Param:
            anchor: [-1.5  1.   8.5  6. ]
            w = anchor[2] - anchor[0] + 1
            w: 11.0
            h = anchor[3] - anchor[1] + 1
            h: 6.0
            x_ctr = anchor[0] + 0.5 * (w - 1)
            x_ctr: 3.5
            y_ctr = anchor[1] + 0.5 * (h - 1)
            y_ctr: 3.5
            return w, h, x_ctr, y_ctr
            } // END _whctrs(anchors)
        w: 11.0, h: 6.0, x_ctr: 3.5, y_ctr: 3.5
        ws = w * scale
        ws: [44.         55.4365262  69.84564629]
        hs = h * scale
        hs: [24.         30.2381052  38.09762525]
        anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
        _mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
            Param:
            ws: [44.         55.4365262  69.84564629]
            hs: [24.         30.2381052  38.09762525]
            x_ctr: 3.5
            y_ctr: 3.5
            ws = ws[:, np.newaxis]
            ws: [[44.        ]
                [55.4365262 ]
                [69.84564629]]
            hs = hs[:, np.newaxis]
            hs: [[24.        ]
                [30.2381052 ]
                [38.09762525]]
            anchors = np.hstack(
                (
                    x_ctr - 0.5 * (ws - 1),
                    y_ctr - 0.5 * (hs - 1),
                    x_ctr + 0.5 * (ws - 1),
                    y_ctr + 0.5 * (hs - 1),
                    )
                )
            anchors: [[-18.          -8.          25.          15.        ]
                [-23.7182631  -11.1190526   30.7182631   18.1190526 ]
                [-30.92282314 -15.04881262  37.92282314  22.04881262]]
            return anchors
            } // END _mkanchors(ws, hs, x_ctr, y_ctr)
        anchors: [[-18.          -8.          25.          15.        ]
                [-23.7182631  -11.1190526   30.7182631   18.1190526 ]
                [-30.92282314 -15.04881262  37.92282314  22.04881262]]
        } // END _scale_enum(anchor, scales)
                    _scale_enum(anchor, scales) { //BEGIN
                            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                            Param:
                            anchor: [0. 0. 7. 7.]
                            scales: [4.         5.0396842  6.34960421]
                            w, h, x_ctr, y_ctr = _whctrs(anchor)
                            _whctrs(anchors) { //BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                anchor: [0. 0. 7. 7.]
                                w = anchor[2] - anchor[0] + 1
                                w: 8.0
                                h = anchor[3] - anchor[1] + 1
                                h: 8.0
                                x_ctr = anchor[0] + 0.5 * (w - 1)
                                x_ctr: 3.5
                                y_ctr = anchor[1] + 0.5 * (h - 1)
                                y_ctr: 3.5
                                return w, h, x_ctr, y_ctr
                                } // END _whctrs(anchors)
                            w: 8.0, h: 8.0, x_ctr: 3.5, y_ctr: 3.5
                            ws = w * scale
                            ws: [32.         40.3174736  50.79683366]
                            hs = h * scale
                            hs: [32.         40.3174736  50.79683366]
                            anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
                            _mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                ws: [32.         40.3174736  50.79683366]
                                hs: [32.         40.3174736  50.79683366]
                                x_ctr: 3.5
                                y_ctr: 3.5
                                ws = ws[:, np.newaxis]
                                ws: [[32.        ]
                                    [40.3174736 ]
                                    [50.79683366]]
                                hs = hs[:, np.newaxis]
                                hs: [[32.        ]
                                    [40.3174736 ]
                                    [50.79683366]]
                                anchors = np.hstack(
                                    (
                                        x_ctr - 0.5 * (ws - 1),
                                        y_ctr - 0.5 * (hs - 1),
                                        x_ctr + 0.5 * (ws - 1),
                                        y_ctr + 0.5 * (hs - 1),
                                        )
                                    )
                                anchors: [[-12.         -12.          19.          19.        ]
                                    [-16.1587368  -16.1587368   23.1587368   23.1587368 ]
                                    [-21.39841683 -21.39841683  28.39841683  28.39841683]]
                                return anchors
                                } // END _mkanchors(ws, hs, x_ctr, y_ctr)
                            anchors: [[-12.         -12.          19.          19.        ]
                                    [-16.1587368  -16.1587368   23.1587368   23.1587368 ]
                                    [-21.39841683 -21.39841683  28.39841683  28.39841683]]
                            } // END _scale_enum(anchor, scales)
                    _scale_enum(anchor, scales) { //BEGIN
                            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                            Param:
                            anchor: [ 1. -2.  6.  9.]
                            scales: [4.         5.0396842  6.34960421]
                            w, h, x_ctr, y_ctr = _whctrs(anchor)
                            _whctrs(anchors) { //BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                anchor: [ 1. -2.  6.  9.]
                                w = anchor[2] - anchor[0] + 1
                                w: 6.0
                                h = anchor[3] - anchor[1] + 1
                                h: 12.0
                                x_ctr = anchor[0] + 0.5 * (w - 1)
                                x_ctr: 3.5
                                y_ctr = anchor[1] + 0.5 * (h - 1)
                                y_ctr: 3.5
                                return w, h, x_ctr, y_ctr
                                } // END _whctrs(anchors)
                            w: 6.0, h: 12.0, x_ctr: 3.5, y_ctr: 3.5
                            ws = w * scale
                            ws: [24.         30.2381052  38.09762525]
                            hs = h * scale
                            hs: [48.         60.47621039 76.19525049]
                            anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
                            _mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                ws: [24.         30.2381052  38.09762525]
                                hs: [48.         60.47621039 76.19525049]
                                x_ctr: 3.5
                                y_ctr: 3.5
                                ws = ws[:, np.newaxis]
                                ws: [[24.        ]
                                    [30.2381052 ]
                                    [38.09762525]]
                                hs = hs[:, np.newaxis]
                                hs: [[48.        ]
                                    [60.47621039]
                                    [76.19525049]]
                                anchors = np.hstack(
                                    (
                                        x_ctr - 0.5 * (ws - 1),
                                        y_ctr - 0.5 * (hs - 1),
                                        x_ctr + 0.5 * (ws - 1),
                                        y_ctr + 0.5 * (hs - 1),
                                        )
                                    )
                                anchors: [[ -8.         -20.          15.          27.        ]
                                    [-11.1190526  -26.2381052   18.1190526   33.2381052 ]
                                    [-15.04881262 -34.09762525  22.04881262  41.09762525]]
                                return anchors
                                } // END _mkanchors(ws, hs, x_ctr, y_ctr)
                            anchors: [[ -8.         -20.          15.          27.        ]
                                    [-11.1190526  -26.2381052   18.1190526   33.2381052 ]
                                    [-15.04881262 -34.09762525  22.04881262  41.09762525]]
                            } // END _scale_enum(anchor, scales)
anchors: [[-18.          -8.          25.          15.        ]
        [-23.7182631  -11.1190526   30.7182631   18.1190526 ]
        [-30.92282314 -15.04881262  37.92282314  22.04881262]
        [-12.         -12.          19.          19.        ]
        [-16.1587368  -16.1587368   23.1587368   23.1587368 ]
        [-21.39841683 -21.39841683  28.39841683  28.39841683]
        [ -8.         -20.          15.          27.        ]
        [-11.1190526  -26.2381052   18.1190526   33.2381052 ]
        [-15.04881262 -34.09762525  22.04881262  41.09762525]]
return torch.from_numpy(anchors)
                        } // END _generate_anchors(base_size, scales, apect_ratios) END
                    } // END generate_anchors(stride, sizes, aspect_ratios)
                generate_anchors(stride, sizes, aspect_ratios) { //BEGIN
                        /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                        Params:
                        stride: 16
                        sizes: (64.0, 80.63494719327188, 101.59366732596476)
                        aspect_ratios: (0.5, 1.0, 2.0)
                        return _generate_anchors(stride,
                            np.array(sizes, dtype=np.float) / stride,
                            np.array(aspect_ratios, dtype=np.float),)
                        _generate_anchors(base_size, scales, aspect_ratios) { //BEGIN
                            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                            Params:
                            base_size: 16
                            scales: [4.         5.0396842  6.34960421]
                            aspect_ratios: [0.5 1.  2. ]
                            anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
                            anchor: [ 0.  0. 15. 15.]
                            anchors = _ratio_enum(anchor, aspect_ratios)
                            _ratio_enum(anchor, ratios) { //BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                anchor: [ 0.  0. 15. 15.]
                                ratios: [0.5 1.  2. ]
                                _whctrs(anchors) { //BEGIN
                                    /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                    Param:
                                    anchor: [ 0.  0. 15. 15.]
                                    w = anchor[2] - anchor[0] + 1
                                    w: 16.0
                                    h = anchor[3] - anchor[1] + 1
                                    h: 16.0
                                    x_ctr = anchor[0] + 0.5 * (w - 1)
                                    x_ctr: 7.5
                                    y_ctr = anchor[1] + 0.5 * (h - 1)
                                    y_ctr: 7.5
                                    return w, h, x_ctr, y_ctr
                                    } // END _whctrs(anchors)
                                w, h, x_ctr, y_ctr = _whctrs(anchor)
                                w: 16.0, h: 16.0, x_ctr: 7.5, y_ctr: 7.5
                                size = w * h
                                size: 256.0
                                size_ratios = size / ratios
                                size_ratios: [512. 256. 128.]
                                ws = np.round(np.sqrt(size_ratios))
                                ws: [23. 16. 11.]
                                hs = np.round(ws * ratios)
                                hs: [12. 16. 22.]
                                _mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
                                    /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                    Param:
                                    ws: [23. 16. 11.]
                                    hs: [12. 16. 22.]
                                    x_ctr: 7.5
                                    y_ctr: 7.5
                                    ws = ws[:, np.newaxis]
                                    ws: [[23.]
                                        [16.]
                                        [11.]]
                                    hs = hs[:, np.newaxis]
                                    hs: [[12.]
                                        [16.]
                                        [22.]]
                                    anchors = np.hstack(
                                        (
                                            x_ctr - 0.5 * (ws - 1),
                                            y_ctr - 0.5 * (hs - 1),
                                            x_ctr + 0.5 * (ws - 1),
                                            y_ctr + 0.5 * (hs - 1),
                                            )
                                        )
                                    anchors: [[-3.5  2.  18.5 13. ]
                                        [ 0.   0.  15.  15. ]
                                        [ 2.5 -3.  12.5 18. ]]
                                    return anchors
                                    } // END _mkanchors(ws, hs, x_ctr, y_ctr)
                                anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
                    anchors: [[-3.5  2.  18.5 13. ]
                            [ 0.   0.  15.  15. ]
                            [ 2.5 -3.  12.5 18. ]]
                    return anchors
                } // END _ratio_enum(anchor, ratios)
anchors: [[-3.5  2.  18.5 13. ]
        [ 0.   0.  15.  15. ]
        [ 2.5 -3.  12.5 18. ]]
anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
        )
_scale_enum(anchor, scales) { //BEGIN
        /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
        Param:
        anchor: [-3.5  2.  18.5 13. ]
        scales: [4.         5.0396842  6.34960421]
        w, h, x_ctr, y_ctr = _whctrs(anchor)
        _whctrs(anchors) { //BEGIN
            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
            Param:
            anchor: [-3.5  2.  18.5 13. ]
            w = anchor[2] - anchor[0] + 1
            w: 23.0
            h = anchor[3] - anchor[1] + 1
            h: 12.0
            x_ctr = anchor[0] + 0.5 * (w - 1)
            x_ctr: 7.5
            y_ctr = anchor[1] + 0.5 * (h - 1)
            y_ctr: 7.5
            return w, h, x_ctr, y_ctr
            } // END _whctrs(anchors)
        w: 23.0, h: 12.0, x_ctr: 7.5, y_ctr: 7.5
        ws = w * scale
        ws: [ 92.         115.91273659 146.04089678]
        hs = h * scale
        hs: [48.         60.47621039 76.19525049]
        anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
        _mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
            Param:
            ws: [ 92.         115.91273659 146.04089678]
            hs: [48.         60.47621039 76.19525049]
            x_ctr: 7.5
            y_ctr: 7.5
            ws = ws[:, np.newaxis]
            ws: [[ 92.        ]
                [115.91273659]
                [146.04089678]]
            hs = hs[:, np.newaxis]
            hs: [[48.        ]
                [60.47621039]
                [76.19525049]]
            anchors = np.hstack(
                (
                    x_ctr - 0.5 * (ws - 1),
                    y_ctr - 0.5 * (hs - 1),
                    x_ctr + 0.5 * (ws - 1),
                    y_ctr + 0.5 * (hs - 1),
                    )
                )
            anchors: [[-38.         -16.          53.          31.        ]
                [-49.9563683  -22.2381052   64.9563683   37.2381052 ]
                [-65.02044839 -30.09762525  80.02044839  45.09762525]]
            return anchors
            } // END _mkanchors(ws, hs, x_ctr, y_ctr)
        anchors: [[-38.         -16.          53.          31.        ]
                [-49.9563683  -22.2381052   64.9563683   37.2381052 ]
                [-65.02044839 -30.09762525  80.02044839  45.09762525]]
        } // END _scale_enum(anchor, scales)
                    _scale_enum(anchor, scales) { //BEGIN
                            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                            Param:
                            anchor: [ 0.  0. 15. 15.]
                            scales: [4.         5.0396842  6.34960421]
                            w, h, x_ctr, y_ctr = _whctrs(anchor)
                            _whctrs(anchors) { //BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                anchor: [ 0.  0. 15. 15.]
                                w = anchor[2] - anchor[0] + 1
                                w: 16.0
                                h = anchor[3] - anchor[1] + 1
                                h: 16.0
                                x_ctr = anchor[0] + 0.5 * (w - 1)
                                x_ctr: 7.5
                                y_ctr = anchor[1] + 0.5 * (h - 1)
                                y_ctr: 7.5
                                return w, h, x_ctr, y_ctr
                                } // END _whctrs(anchors)
                            w: 16.0, h: 16.0, x_ctr: 7.5, y_ctr: 7.5
                            ws = w * scale
                            ws: [ 64.          80.63494719 101.59366733]
                            hs = h * scale
                            hs: [ 64.          80.63494719 101.59366733]
                            anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
                            _mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                ws: [ 64.          80.63494719 101.59366733]
                                hs: [ 64.          80.63494719 101.59366733]
                                x_ctr: 7.5
                                y_ctr: 7.5
                                ws = ws[:, np.newaxis]
                                ws: [[ 64.        ]
                                    [ 80.63494719]
                                    [101.59366733]]
                                hs = hs[:, np.newaxis]
                                hs: [[ 64.        ]
                                    [ 80.63494719]
                                    [101.59366733]]
                                anchors = np.hstack(
                                    (
                                        x_ctr - 0.5 * (ws - 1),
                                        y_ctr - 0.5 * (hs - 1),
                                        x_ctr + 0.5 * (ws - 1),
                                        y_ctr + 0.5 * (hs - 1),
                                        )
                                    )
                                anchors: [[-24.         -24.          39.          39.        ]
                                    [-32.3174736  -32.3174736   47.3174736   47.3174736 ]
                                    [-42.79683366 -42.79683366  57.79683366  57.79683366]]
                                return anchors
                                } // END _mkanchors(ws, hs, x_ctr, y_ctr)
                            anchors: [[-24.         -24.          39.          39.        ]
                                    [-32.3174736  -32.3174736   47.3174736   47.3174736 ]
                                    [-42.79683366 -42.79683366  57.79683366  57.79683366]]
                            } // END _scale_enum(anchor, scales)
                    _scale_enum(anchor, scales) { //BEGIN
                            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                            Param:
                            anchor: [ 2.5 -3.  12.5 18. ]
                            scales: [4.         5.0396842  6.34960421]
                            w, h, x_ctr, y_ctr = _whctrs(anchor)
                            _whctrs(anchors) { //BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                anchor: [ 2.5 -3.  12.5 18. ]
                                w = anchor[2] - anchor[0] + 1
                                w: 11.0
                                h = anchor[3] - anchor[1] + 1
                                h: 22.0
                                x_ctr = anchor[0] + 0.5 * (w - 1)
                                x_ctr: 7.5
                                y_ctr = anchor[1] + 0.5 * (h - 1)
                                y_ctr: 7.5
                                return w, h, x_ctr, y_ctr
                                } // END _whctrs(anchors)
                            w: 11.0, h: 22.0, x_ctr: 7.5, y_ctr: 7.5
                            ws = w * scale
                            ws: [44.         55.4365262  69.84564629]
                            hs = h * scale
                            hs: [ 88.         110.87305239 139.69129257]
                            anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
                            _mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                ws: [44.         55.4365262  69.84564629]
                                hs: [ 88.         110.87305239 139.69129257]
                                x_ctr: 7.5
                                y_ctr: 7.5
                                ws = ws[:, np.newaxis]
                                ws: [[44.        ]
                                    [55.4365262 ]
                                    [69.84564629]]
                                hs = hs[:, np.newaxis]
                                hs: [[ 88.        ]
                                    [110.87305239]
                                    [139.69129257]]
                                anchors = np.hstack(
                                    (
                                        x_ctr - 0.5 * (ws - 1),
                                        y_ctr - 0.5 * (hs - 1),
                                        x_ctr + 0.5 * (ws - 1),
                                        y_ctr + 0.5 * (hs - 1),
                                        )
                                    )
                                anchors: [[-14.         -36.          29.          51.        ]
                                    [-19.7182631  -47.4365262   34.7182631   62.4365262 ]
                                    [-26.92282314 -61.84564629  41.92282314  76.84564629]]
                                return anchors
                                } // END _mkanchors(ws, hs, x_ctr, y_ctr)
                            anchors: [[-14.         -36.          29.          51.        ]
                                    [-19.7182631  -47.4365262   34.7182631   62.4365262 ]
                                    [-26.92282314 -61.84564629  41.92282314  76.84564629]]
                            } // END _scale_enum(anchor, scales)
anchors: [[-38.         -16.          53.          31.        ]
        [-49.9563683  -22.2381052   64.9563683   37.2381052 ]
        [-65.02044839 -30.09762525  80.02044839  45.09762525]
        [-24.         -24.          39.          39.        ]
        [-32.3174736  -32.3174736   47.3174736   47.3174736 ]
        [-42.79683366 -42.79683366  57.79683366  57.79683366]
        [-14.         -36.          29.          51.        ]
        [-19.7182631  -47.4365262   34.7182631   62.4365262 ]
        [-26.92282314 -61.84564629  41.92282314  76.84564629]]
return torch.from_numpy(anchors)
                        } // END _generate_anchors(base_size, scales, apect_ratios) END
                    } // END generate_anchors(stride, sizes, aspect_ratios)
                generate_anchors(stride, sizes, aspect_ratios) { //BEGIN
                        /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                        Params:
                        stride: 32
                        sizes: (128.0, 161.26989438654377, 203.18733465192952)
                        aspect_ratios: (0.5, 1.0, 2.0)
                        return _generate_anchors(stride,
                            np.array(sizes, dtype=np.float) / stride,
                            np.array(aspect_ratios, dtype=np.float),)
                        _generate_anchors(base_size, scales, aspect_ratios) { //BEGIN
                            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                            Params:
                            base_size: 32
                            scales: [4.         5.0396842  6.34960421]
                            aspect_ratios: [0.5 1.  2. ]
                            anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
                            anchor: [ 0.  0. 31. 31.]
                            anchors = _ratio_enum(anchor, aspect_ratios)
                            _ratio_enum(anchor, ratios) { //BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                anchor: [ 0.  0. 31. 31.]
                                ratios: [0.5 1.  2. ]
                                _whctrs(anchors) { //BEGIN
                                    /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                    Param:
                                    anchor: [ 0.  0. 31. 31.]
                                    w = anchor[2] - anchor[0] + 1
                                    w: 32.0
                                    h = anchor[3] - anchor[1] + 1
                                    h: 32.0
                                    x_ctr = anchor[0] + 0.5 * (w - 1)
                                    x_ctr: 15.5
                                    y_ctr = anchor[1] + 0.5 * (h - 1)
                                    y_ctr: 15.5
                                    return w, h, x_ctr, y_ctr
                                    } // END _whctrs(anchors)
                                w, h, x_ctr, y_ctr = _whctrs(anchor)
                                w: 32.0, h: 32.0, x_ctr: 15.5, y_ctr: 15.5
                                size = w * h
                                size: 1024.0
                                size_ratios = size / ratios
                                size_ratios: [2048. 1024.  512.]
                                ws = np.round(np.sqrt(size_ratios))
                                ws: [45. 32. 23.]
                                hs = np.round(ws * ratios)
                                hs: [22. 32. 46.]
                                _mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
                                    /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                    Param:
                                    ws: [45. 32. 23.]
                                    hs: [22. 32. 46.]
                                    x_ctr: 15.5
                                    y_ctr: 15.5
                                    ws = ws[:, np.newaxis]
                                    ws: [[45.]
                                        [32.]
                                        [23.]]
                                    hs = hs[:, np.newaxis]
                                    hs: [[22.]
                                        [32.]
                                        [46.]]
                                    anchors = np.hstack(
                                        (
                                            x_ctr - 0.5 * (ws - 1),
                                            y_ctr - 0.5 * (hs - 1),
                                            x_ctr + 0.5 * (ws - 1),
                                            y_ctr + 0.5 * (hs - 1),
                                            )
                                        )
                                    anchors: [[-6.5  5.  37.5 26. ]
                                        [ 0.   0.  31.  31. ]
                                        [ 4.5 -7.  26.5 38. ]]
                                    return anchors
                                    } // END _mkanchors(ws, hs, x_ctr, y_ctr)
                                anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
                    anchors: [[-6.5  5.  37.5 26. ]
                            [ 0.   0.  31.  31. ]
                            [ 4.5 -7.  26.5 38. ]]
                    return anchors
                } // END _ratio_enum(anchor, ratios)
anchors: [[-6.5  5.  37.5 26. ]
        [ 0.   0.  31.  31. ]
        [ 4.5 -7.  26.5 38. ]]
anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
        )
_scale_enum(anchor, scales) { //BEGIN
        /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
        Param:
        anchor: [-6.5  5.  37.5 26. ]
        scales: [4.         5.0396842  6.34960421]
        w, h, x_ctr, y_ctr = _whctrs(anchor)
        _whctrs(anchors) { //BEGIN
            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
            Param:
            anchor: [-6.5  5.  37.5 26. ]
            w = anchor[2] - anchor[0] + 1
            w: 45.0
            h = anchor[3] - anchor[1] + 1
            h: 22.0
            x_ctr = anchor[0] + 0.5 * (w - 1)
            x_ctr: 15.5
            y_ctr = anchor[1] + 0.5 * (h - 1)
            y_ctr: 15.5
            return w, h, x_ctr, y_ctr
            } // END _whctrs(anchors)
        w: 45.0, h: 22.0, x_ctr: 15.5, y_ctr: 15.5
        ws = w * scale
        ws: [180.         226.78578898 285.73218935]
        hs = h * scale
        hs: [ 88.         110.87305239 139.69129257]
        anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
        _mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
            Param:
            ws: [180.         226.78578898 285.73218935]
            hs: [ 88.         110.87305239 139.69129257]
            x_ctr: 15.5
            y_ctr: 15.5
            ws = ws[:, np.newaxis]
            ws: [[180.        ]
                [226.78578898]
                [285.73218935]]
            hs = hs[:, np.newaxis]
            hs: [[ 88.        ]
                [110.87305239]
                [139.69129257]]
            anchors = np.hstack(
                (
                    x_ctr - 0.5 * (ws - 1),
                    y_ctr - 0.5 * (hs - 1),
                    x_ctr + 0.5 * (ws - 1),
                    y_ctr + 0.5 * (hs - 1),
                    )
                )
            anchors: [[ -74.          -28.          105.           59.        ]
                [ -97.39289449  -39.4365262   128.39289449   70.4365262 ]
                [-126.86609468  -53.84564629  157.86609468   84.84564629]]
            return anchors
            } // END _mkanchors(ws, hs, x_ctr, y_ctr)
        anchors: [[ -74.          -28.          105.           59.        ]
                [ -97.39289449  -39.4365262   128.39289449   70.4365262 ]
                [-126.86609468  -53.84564629  157.86609468   84.84564629]]
        } // END _scale_enum(anchor, scales)
                    _scale_enum(anchor, scales) { //BEGIN
                            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                            Param:
                            anchor: [ 0.  0. 31. 31.]
                            scales: [4.         5.0396842  6.34960421]
                            w, h, x_ctr, y_ctr = _whctrs(anchor)
                            _whctrs(anchors) { //BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                anchor: [ 0.  0. 31. 31.]
                                w = anchor[2] - anchor[0] + 1
                                w: 32.0
                                h = anchor[3] - anchor[1] + 1
                                h: 32.0
                                x_ctr = anchor[0] + 0.5 * (w - 1)
                                x_ctr: 15.5
                                y_ctr = anchor[1] + 0.5 * (h - 1)
                                y_ctr: 15.5
                                return w, h, x_ctr, y_ctr
                                } // END _whctrs(anchors)
                            w: 32.0, h: 32.0, x_ctr: 15.5, y_ctr: 15.5
                            ws = w * scale
                            ws: [128.         161.26989439 203.18733465]
                            hs = h * scale
                            hs: [128.         161.26989439 203.18733465]
                            anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
                            _mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                ws: [128.         161.26989439 203.18733465]
                                hs: [128.         161.26989439 203.18733465]
                                x_ctr: 15.5
                                y_ctr: 15.5
                                ws = ws[:, np.newaxis]
                                ws: [[128.        ]
                                    [161.26989439]
                                    [203.18733465]]
                                hs = hs[:, np.newaxis]
                                hs: [[128.        ]
                                    [161.26989439]
                                    [203.18733465]]
                                anchors = np.hstack(
                                    (
                                        x_ctr - 0.5 * (ws - 1),
                                        y_ctr - 0.5 * (hs - 1),
                                        x_ctr + 0.5 * (ws - 1),
                                        y_ctr + 0.5 * (hs - 1),
                                        )
                                    )
                                anchors: [[-48.         -48.          79.          79.        ]
                                    [-64.63494719 -64.63494719  95.63494719  95.63494719]
                                    [-85.59366733 -85.59366733 116.59366733 116.59366733]]
                                return anchors
                                } // END _mkanchors(ws, hs, x_ctr, y_ctr)
                            anchors: [[-48.         -48.          79.          79.        ]
                                    [-64.63494719 -64.63494719  95.63494719  95.63494719]
                                    [-85.59366733 -85.59366733 116.59366733 116.59366733]]
                            } // END _scale_enum(anchor, scales)
                    _scale_enum(anchor, scales) { //BEGIN
                            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                            Param:
                            anchor: [ 4.5 -7.  26.5 38. ]
                            scales: [4.         5.0396842  6.34960421]
                            w, h, x_ctr, y_ctr = _whctrs(anchor)
                            _whctrs(anchors) { //BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                anchor: [ 4.5 -7.  26.5 38. ]
                                w = anchor[2] - anchor[0] + 1
                                w: 23.0
                                h = anchor[3] - anchor[1] + 1
                                h: 46.0
                                x_ctr = anchor[0] + 0.5 * (w - 1)
                                x_ctr: 15.5
                                y_ctr = anchor[1] + 0.5 * (h - 1)
                                y_ctr: 15.5
                                return w, h, x_ctr, y_ctr
                                } // END _whctrs(anchors)
                            w: 23.0, h: 46.0, x_ctr: 15.5, y_ctr: 15.5
                            ws = w * scale
                            ws: [ 92.         115.91273659 146.04089678]
                            hs = h * scale
                            hs: [184.         231.82547318 292.08179356]
                            anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
                            _mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                ws: [ 92.         115.91273659 146.04089678]
                                hs: [184.         231.82547318 292.08179356]
                                x_ctr: 15.5
                                y_ctr: 15.5
                                ws = ws[:, np.newaxis]
                                ws: [[ 92.        ]
                                    [115.91273659]
                                    [146.04089678]]
                                hs = hs[:, np.newaxis]
                                hs: [[184.        ]
                                    [231.82547318]
                                    [292.08179356]]
                                anchors = np.hstack(
                                    (
                                        x_ctr - 0.5 * (ws - 1),
                                        y_ctr - 0.5 * (hs - 1),
                                        x_ctr + 0.5 * (ws - 1),
                                        y_ctr + 0.5 * (hs - 1),
                                        )
                                    )
                                anchors: [[ -30.          -76.           61.          107.        ]
                                    [ -41.9563683   -99.91273659   72.9563683   130.91273659]
                                    [ -57.02044839 -130.04089678   88.02044839  161.04089678]]
                                return anchors
                                } // END _mkanchors(ws, hs, x_ctr, y_ctr)
                            anchors: [[ -30.          -76.           61.          107.        ]
                                    [ -41.9563683   -99.91273659   72.9563683   130.91273659]
                                    [ -57.02044839 -130.04089678   88.02044839  161.04089678]]
                            } // END _scale_enum(anchor, scales)
anchors: [[ -74.          -28.          105.           59.        ]
        [ -97.39289449  -39.4365262   128.39289449   70.4365262 ]
        [-126.86609468  -53.84564629  157.86609468   84.84564629]
        [ -48.          -48.           79.           79.        ]
        [ -64.63494719  -64.63494719   95.63494719   95.63494719]
        [ -85.59366733  -85.59366733  116.59366733  116.59366733]
        [ -30.          -76.           61.          107.        ]
        [ -41.9563683   -99.91273659   72.9563683   130.91273659]
        [ -57.02044839 -130.04089678   88.02044839  161.04089678]]
return torch.from_numpy(anchors)
                        } // END _generate_anchors(base_size, scales, apect_ratios) END
                    } // END generate_anchors(stride, sizes, aspect_ratios)
                generate_anchors(stride, sizes, aspect_ratios) { //BEGIN
                        /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                        Params:
                        stride: 64
                        sizes: (256.0, 322.53978877308754, 406.37466930385904)
                        aspect_ratios: (0.5, 1.0, 2.0)
                        return _generate_anchors(stride,
                            np.array(sizes, dtype=np.float) / stride,
                            np.array(aspect_ratios, dtype=np.float),)
                        _generate_anchors(base_size, scales, aspect_ratios) { //BEGIN
                            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                            Params:
                            base_size: 64
                            scales: [4.         5.0396842  6.34960421]
                            aspect_ratios: [0.5 1.  2. ]
                            anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
                            anchor: [ 0.  0. 63. 63.]
                            anchors = _ratio_enum(anchor, aspect_ratios)
                            _ratio_enum(anchor, ratios) { //BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                anchor: [ 0.  0. 63. 63.]
                                ratios: [0.5 1.  2. ]
                                _whctrs(anchors) { //BEGIN
                                    /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                    Param:
                                    anchor: [ 0.  0. 63. 63.]
                                    w = anchor[2] - anchor[0] + 1
                                    w: 64.0
                                    h = anchor[3] - anchor[1] + 1
                                    h: 64.0
                                    x_ctr = anchor[0] + 0.5 * (w - 1)
                                    x_ctr: 31.5
                                    y_ctr = anchor[1] + 0.5 * (h - 1)
                                    y_ctr: 31.5
                                    return w, h, x_ctr, y_ctr
                                    } // END _whctrs(anchors)
                                w, h, x_ctr, y_ctr = _whctrs(anchor)
                                w: 64.0, h: 64.0, x_ctr: 31.5, y_ctr: 31.5
                                size = w * h
                                size: 4096.0
                                size_ratios = size / ratios
                                size_ratios: [8192. 4096. 2048.]
                                ws = np.round(np.sqrt(size_ratios))
                                ws: [91. 64. 45.]
                                hs = np.round(ws * ratios)
                                hs: [46. 64. 90.]
                                _mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
                                    /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                    Param:
                                    ws: [91. 64. 45.]
                                    hs: [46. 64. 90.]
                                    x_ctr: 31.5
                                    y_ctr: 31.5
                                    ws = ws[:, np.newaxis]
                                    ws: [[91.]
                                        [64.]
                                        [45.]]
                                    hs = hs[:, np.newaxis]
                                    hs: [[46.]
                                        [64.]
                                        [90.]]
                                    anchors = np.hstack(
                                        (
                                            x_ctr - 0.5 * (ws - 1),
                                            y_ctr - 0.5 * (hs - 1),
                                            x_ctr + 0.5 * (ws - 1),
                                            y_ctr + 0.5 * (hs - 1),
                                            )
                                        )
                                    anchors: [[-13.5   9.   76.5  54. ]
                                        [  0.    0.   63.   63. ]
                                        [  9.5 -13.   53.5  76. ]]
                                    return anchors
                                    } // END _mkanchors(ws, hs, x_ctr, y_ctr)
                                anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
                    anchors: [[-13.5   9.   76.5  54. ]
                            [  0.    0.   63.   63. ]
                            [  9.5 -13.   53.5  76. ]]
                    return anchors
                } // END _ratio_enum(anchor, ratios)
anchors: [[-13.5   9.   76.5  54. ]
        [  0.    0.   63.   63. ]
        [  9.5 -13.   53.5  76. ]]
anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
        )
_scale_enum(anchor, scales) { //BEGIN
        /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
        Param:
        anchor: [-13.5   9.   76.5  54. ]
        scales: [4.         5.0396842  6.34960421]
        w, h, x_ctr, y_ctr = _whctrs(anchor)
        _whctrs(anchors) { //BEGIN
            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
            Param:
            anchor: [-13.5   9.   76.5  54. ]
            w = anchor[2] - anchor[0] + 1
            w: 91.0
            h = anchor[3] - anchor[1] + 1
            h: 46.0
            x_ctr = anchor[0] + 0.5 * (w - 1)
            x_ctr: 31.5
            y_ctr = anchor[1] + 0.5 * (h - 1)
            y_ctr: 31.5
            return w, h, x_ctr, y_ctr
            } // END _whctrs(anchors)
        w: 91.0, h: 46.0, x_ctr: 31.5, y_ctr: 31.5
        ws = w * scale
        ws: [364.         458.61126216 577.81398292]
        hs = h * scale
        hs: [184.         231.82547318 292.08179356]
        anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
        _mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
            Param:
            ws: [364.         458.61126216 577.81398292]
            hs: [184.         231.82547318 292.08179356]
            x_ctr: 31.5
            y_ctr: 31.5
            ws = ws[:, np.newaxis]
            ws: [[364.        ]
                [458.61126216]
                [577.81398292]]
            hs = hs[:, np.newaxis]
            hs: [[184.        ]
                [231.82547318]
                [292.08179356]]
            anchors = np.hstack(
                (
                    x_ctr - 0.5 * (ws - 1),
                    y_ctr - 0.5 * (hs - 1),
                    x_ctr + 0.5 * (ws - 1),
                    y_ctr + 0.5 * (hs - 1),
                    )
                )
            anchors: [[-150.          -60.          213.          123.        ]
                [-197.30563108  -83.91273659  260.30563108  146.91273659]
                [-256.90699146 -114.04089678  319.90699146  177.04089678]]
            return anchors
            } // END _mkanchors(ws, hs, x_ctr, y_ctr)
        anchors: [[-150.          -60.          213.          123.        ]
                [-197.30563108  -83.91273659  260.30563108  146.91273659]
                [-256.90699146 -114.04089678  319.90699146  177.04089678]]
        } // END _scale_enum(anchor, scales)
                    _scale_enum(anchor, scales) { //BEGIN
                            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                            Param:
                            anchor: [ 0.  0. 63. 63.]
                            scales: [4.         5.0396842  6.34960421]
                            w, h, x_ctr, y_ctr = _whctrs(anchor)
                            _whctrs(anchors) { //BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                anchor: [ 0.  0. 63. 63.]
                                w = anchor[2] - anchor[0] + 1
                                w: 64.0
                                h = anchor[3] - anchor[1] + 1
                                h: 64.0
                                x_ctr = anchor[0] + 0.5 * (w - 1)
                                x_ctr: 31.5
                                y_ctr = anchor[1] + 0.5 * (h - 1)
                                y_ctr: 31.5
                                return w, h, x_ctr, y_ctr
                                } // END _whctrs(anchors)
                            w: 64.0, h: 64.0, x_ctr: 31.5, y_ctr: 31.5
                            ws = w * scale
                            ws: [256.         322.53978877 406.3746693 ]
                            hs = h * scale
                            hs: [256.         322.53978877 406.3746693 ]
                            anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
                            _mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                ws: [256.         322.53978877 406.3746693 ]
                                hs: [256.         322.53978877 406.3746693 ]
                                x_ctr: 31.5
                                y_ctr: 31.5
                                ws = ws[:, np.newaxis]
                                ws: [[256.        ]
                                    [322.53978877]
                                    [406.3746693 ]]
                                hs = hs[:, np.newaxis]
                                hs: [[256.        ]
                                    [322.53978877]
                                    [406.3746693 ]]
                                anchors = np.hstack(
                                    (
                                        x_ctr - 0.5 * (ws - 1),
                                        y_ctr - 0.5 * (hs - 1),
                                        x_ctr + 0.5 * (ws - 1),
                                        y_ctr + 0.5 * (hs - 1),
                                        )
                                    )
                                anchors: [[ -96.          -96.          159.          159.        ]
                                    [-129.26989439 -129.26989439  192.26989439  192.26989439]
                                    [-171.18733465 -171.18733465  234.18733465  234.18733465]]
                                return anchors
                                } // END _mkanchors(ws, hs, x_ctr, y_ctr)
                            anchors: [[ -96.          -96.          159.          159.        ]
                                    [-129.26989439 -129.26989439  192.26989439  192.26989439]
                                    [-171.18733465 -171.18733465  234.18733465  234.18733465]]
                            } // END _scale_enum(anchor, scales)
                    _scale_enum(anchor, scales) { //BEGIN
                            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                            Param:
                            anchor: [  9.5 -13.   53.5  76. ]
                            scales: [4.         5.0396842  6.34960421]
                            w, h, x_ctr, y_ctr = _whctrs(anchor)
                            _whctrs(anchors) { //BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                anchor: [  9.5 -13.   53.5  76. ]
                                w = anchor[2] - anchor[0] + 1
                                w: 45.0
                                h = anchor[3] - anchor[1] + 1
                                h: 90.0
                                x_ctr = anchor[0] + 0.5 * (w - 1)
                                x_ctr: 31.5
                                y_ctr = anchor[1] + 0.5 * (h - 1)
                                y_ctr: 31.5
                                return w, h, x_ctr, y_ctr
                                } // END _whctrs(anchors)
                            w: 45.0, h: 90.0, x_ctr: 31.5, y_ctr: 31.5
                            ws = w * scale
                            ws: [180.         226.78578898 285.73218935]
                            hs = h * scale
                            hs: [360.         453.57157796 571.46437871]
                            anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
                            _mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                ws: [180.         226.78578898 285.73218935]
                                hs: [360.         453.57157796 571.46437871]
                                x_ctr: 31.5
                                y_ctr: 31.5
                                ws = ws[:, np.newaxis]
                                ws: [[180.        ]
                                    [226.78578898]
                                    [285.73218935]]
                                hs = hs[:, np.newaxis]
                                hs: [[360.        ]
                                    [453.57157796]
                                    [571.46437871]]
                                anchors = np.hstack(
                                    (
                                        x_ctr - 0.5 * (ws - 1),
                                        y_ctr - 0.5 * (hs - 1),
                                        x_ctr + 0.5 * (ws - 1),
                                        y_ctr + 0.5 * (hs - 1),
                                        )
                                    )
                                anchors: [[ -58.         -148.          121.          211.        ]
                                    [ -81.39289449 -194.78578898  144.39289449  257.78578898]
                                    [-110.86609468 -253.73218935  173.86609468  316.73218935]]
                                return anchors
                                } // END _mkanchors(ws, hs, x_ctr, y_ctr)
                            anchors: [[ -58.         -148.          121.          211.        ]
                                    [ -81.39289449 -194.78578898  144.39289449  257.78578898]
                                    [-110.86609468 -253.73218935  173.86609468  316.73218935]]
                            } // END _scale_enum(anchor, scales)
anchors: [[-150.          -60.          213.          123.        ]
        [-197.30563108  -83.91273659  260.30563108  146.91273659]
        [-256.90699146 -114.04089678  319.90699146  177.04089678]
        [ -96.          -96.          159.          159.        ]
        [-129.26989439 -129.26989439  192.26989439  192.26989439]
        [-171.18733465 -171.18733465  234.18733465  234.18733465]
        [ -58.         -148.          121.          211.        ]
        [ -81.39289449 -194.78578898  144.39289449  257.78578898]
        [-110.86609468 -253.73218935  173.86609468  316.73218935]]
return torch.from_numpy(anchors)
                        } // END _generate_anchors(base_size, scales, apect_ratios) END
                    } // END generate_anchors(stride, sizes, aspect_ratios)
                generate_anchors(stride, sizes, aspect_ratios) { //BEGIN
                        /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                        Params:
                        stride: 128
                        sizes: (512.0, 645.0795775461751, 812.7493386077181)
                        aspect_ratios: (0.5, 1.0, 2.0)
                        return _generate_anchors(stride,
                            np.array(sizes, dtype=np.float) / stride,
                            np.array(aspect_ratios, dtype=np.float),)
                        _generate_anchors(base_size, scales, aspect_ratios) { //BEGIN
                            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                            Params:
                            base_size: 128
                            scales: [4.         5.0396842  6.34960421]
                            aspect_ratios: [0.5 1.  2. ]
                            anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
                            anchor: [  0.   0. 127. 127.]
                            anchors = _ratio_enum(anchor, aspect_ratios)
                            _ratio_enum(anchor, ratios) { //BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                anchor: [  0.   0. 127. 127.]
                                ratios: [0.5 1.  2. ]
                                _whctrs(anchors) { //BEGIN
                                    /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                    Param:
                                    anchor: [  0.   0. 127. 127.]
                                    w = anchor[2] - anchor[0] + 1
                                    w: 128.0
                                    h = anchor[3] - anchor[1] + 1
                                    h: 128.0
                                    x_ctr = anchor[0] + 0.5 * (w - 1)
                                    x_ctr: 63.5
                                    y_ctr = anchor[1] + 0.5 * (h - 1)
                                    y_ctr: 63.5
                                    return w, h, x_ctr, y_ctr
                                    } // END _whctrs(anchors)
                                w, h, x_ctr, y_ctr = _whctrs(anchor)
                                w: 128.0, h: 128.0, x_ctr: 63.5, y_ctr: 63.5
                                size = w * h
                                size: 16384.0
                                size_ratios = size / ratios
                                size_ratios: [32768. 16384.  8192.]
                                ws = np.round(np.sqrt(size_ratios))
                                ws: [181. 128.  91.]
                                hs = np.round(ws * ratios)
                                hs: [ 90. 128. 182.]
                                _mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
                                    /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                    Param:
                                    ws: [181. 128.  91.]
                                    hs: [ 90. 128. 182.]
                                    x_ctr: 63.5
                                    y_ctr: 63.5
                                    ws = ws[:, np.newaxis]
                                    ws: [[181.]
                                        [128.]
                                        [ 91.]]
                                    hs = hs[:, np.newaxis]
                                    hs: [[ 90.]
                                        [128.]
                                        [182.]]
                                    anchors = np.hstack(
                                        (
                                            x_ctr - 0.5 * (ws - 1),
                                            y_ctr - 0.5 * (hs - 1),
                                            x_ctr + 0.5 * (ws - 1),
                                            y_ctr + 0.5 * (hs - 1),
                                            )
                                        )
                                    anchors: [[-26.5  19.  153.5 108. ]
                                        [  0.    0.  127.  127. ]
                                        [ 18.5 -27.  108.5 154. ]]
                                    return anchors
                                    } // END _mkanchors(ws, hs, x_ctr, y_ctr)
                                anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
                    anchors: [[-26.5  19.  153.5 108. ]
                            [  0.    0.  127.  127. ]
                            [ 18.5 -27.  108.5 154. ]]
                    return anchors
                } // END _ratio_enum(anchor, ratios)
anchors: [[-26.5  19.  153.5 108. ]
        [  0.    0.  127.  127. ]
        [ 18.5 -27.  108.5 154. ]]
anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
        )
_scale_enum(anchor, scales) { //BEGIN
        /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
        Param:
        anchor: [-26.5  19.  153.5 108. ]
        scales: [4.         5.0396842  6.34960421]
        w, h, x_ctr, y_ctr = _whctrs(anchor)
        _whctrs(anchors) { //BEGIN
            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
            Param:
            anchor: [-26.5  19.  153.5 108. ]
            w = anchor[2] - anchor[0] + 1
            w: 181.0
            h = anchor[3] - anchor[1] + 1
            h: 90.0
            x_ctr = anchor[0] + 0.5 * (w - 1)
            x_ctr: 63.5
            y_ctr = anchor[1] + 0.5 * (h - 1)
            y_ctr: 63.5
            return w, h, x_ctr, y_ctr
            } // END _whctrs(anchors)
        w: 181.0, h: 90.0, x_ctr: 63.5, y_ctr: 63.5
        ws = w * scale
        ws: [ 724.          912.18284012 1149.27836162]
        hs = h * scale
        hs: [360.         453.57157796 571.46437871]
        anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
        _mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
            Param:
            ws: [ 724.          912.18284012 1149.27836162]
            hs: [360.         453.57157796 571.46437871]
            x_ctr: 63.5
            y_ctr: 63.5
            ws = ws[:, np.newaxis]
            ws: [[ 724.        ]
                [ 912.18284012]
                [1149.27836162]]
            hs = hs[:, np.newaxis]
            hs: [[360.        ]
                [453.57157796]
                [571.46437871]]
            anchors = np.hstack(
                (
                    x_ctr - 0.5 * (ws - 1),
                    y_ctr - 0.5 * (hs - 1),
                    x_ctr + 0.5 * (ws - 1),
                    y_ctr + 0.5 * (hs - 1),
                    )
                )
            anchors: [[-298.         -116.          425.          243.        ]
                [-392.09142006 -162.78578898  519.09142006  289.78578898]
                [-510.63918081 -221.73218935  637.63918081  348.73218935]]
            return anchors
            } // END _mkanchors(ws, hs, x_ctr, y_ctr)
        anchors: [[-298.         -116.          425.          243.        ]
                [-392.09142006 -162.78578898  519.09142006  289.78578898]
                [-510.63918081 -221.73218935  637.63918081  348.73218935]]
        } // END _scale_enum(anchor, scales)
                    _scale_enum(anchor, scales) { //BEGIN
                            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                            Param:
                            anchor: [  0.   0. 127. 127.]
                            scales: [4.         5.0396842  6.34960421]
                            w, h, x_ctr, y_ctr = _whctrs(anchor)
                            _whctrs(anchors) { //BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                anchor: [  0.   0. 127. 127.]
                                w = anchor[2] - anchor[0] + 1
                                w: 128.0
                                h = anchor[3] - anchor[1] + 1
                                h: 128.0
                                x_ctr = anchor[0] + 0.5 * (w - 1)
                                x_ctr: 63.5
                                y_ctr = anchor[1] + 0.5 * (h - 1)
                                y_ctr: 63.5
                                return w, h, x_ctr, y_ctr
                                } // END _whctrs(anchors)
                            w: 128.0, h: 128.0, x_ctr: 63.5, y_ctr: 63.5
                            ws = w * scale
                            ws: [512.         645.07957755 812.74933861]
                            hs = h * scale
                            hs: [512.         645.07957755 812.74933861]
                            anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
                            _mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                ws: [512.         645.07957755 812.74933861]
                                hs: [512.         645.07957755 812.74933861]
                                x_ctr: 63.5
                                y_ctr: 63.5
                                ws = ws[:, np.newaxis]
                                ws: [[512.        ]
                                    [645.07957755]
                                    [812.74933861]]
                                hs = hs[:, np.newaxis]
                                hs: [[512.        ]
                                    [645.07957755]
                                    [812.74933861]]
                                anchors = np.hstack(
                                    (
                                        x_ctr - 0.5 * (ws - 1),
                                        y_ctr - 0.5 * (hs - 1),
                                        x_ctr + 0.5 * (ws - 1),
                                        y_ctr + 0.5 * (hs - 1),
                                        )
                                    )
                                anchors: [[-192.         -192.          319.          319.        ]
                                    [-258.53978877 -258.53978877  385.53978877  385.53978877]
                                    [-342.3746693  -342.3746693   469.3746693   469.3746693 ]]
                                return anchors
                                } // END _mkanchors(ws, hs, x_ctr, y_ctr)
                            anchors: [[-192.         -192.          319.          319.        ]
                                    [-258.53978877 -258.53978877  385.53978877  385.53978877]
                                    [-342.3746693  -342.3746693   469.3746693   469.3746693 ]]
                            } // END _scale_enum(anchor, scales)
                    _scale_enum(anchor, scales) { //BEGIN
                            /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                            Param:
                            anchor: [ 18.5 -27.  108.5 154. ]
                            scales: [4.         5.0396842  6.34960421]
                            w, h, x_ctr, y_ctr = _whctrs(anchor)
                            _whctrs(anchors) { //BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                anchor: [ 18.5 -27.  108.5 154. ]
                                w = anchor[2] - anchor[0] + 1
                                w: 91.0
                                h = anchor[3] - anchor[1] + 1
                                h: 182.0
                                x_ctr = anchor[0] + 0.5 * (w - 1)
                                x_ctr: 63.5
                                y_ctr = anchor[1] + 0.5 * (h - 1)
                                y_ctr: 63.5
                                return w, h, x_ctr, y_ctr
                                } // END _whctrs(anchors)
                            w: 91.0, h: 182.0, x_ctr: 63.5, y_ctr: 63.5
                            ws = w * scale
                            ws: [364.         458.61126216 577.81398292]
                            hs = h * scale
                            hs: [ 728.          917.22252432 1155.62796583]
                            anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
                            _mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
                                /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                                Param:
                                ws: [364.         458.61126216 577.81398292]
                                hs: [ 728.          917.22252432 1155.62796583]
                                x_ctr: 63.5
                                y_ctr: 63.5
                                ws = ws[:, np.newaxis]
                                ws: [[364.        ]
                                    [458.61126216]
                                    [577.81398292]]
                                hs = hs[:, np.newaxis]
                                hs: [[ 728.        ]
                                    [ 917.22252432]
                                    [1155.62796583]]
                                anchors = np.hstack(
                                    (
                                        x_ctr - 0.5 * (ws - 1),
                                        y_ctr - 0.5 * (hs - 1),
                                        x_ctr + 0.5 * (ws - 1),
                                        y_ctr + 0.5 * (hs - 1),
                                        )
                                    )
                                anchors: [[-118.         -300.          245.          427.        ]
                                    [-165.30563108 -394.61126216  292.30563108  521.61126216]
                                    [-224.90699146 -513.81398292  351.90699146  640.81398292]]
                                return anchors
                                } // END _mkanchors(ws, hs, x_ctr, y_ctr)
                            anchors: [[-118.         -300.          245.          427.        ]
                                    [-165.30563108 -394.61126216  292.30563108  521.61126216]
                                    [-224.90699146 -513.81398292  351.90699146  640.81398292]]
                            } // END _scale_enum(anchor, scales)
anchors: [[-298.         -116.          425.          243.        ]
        [-392.09142006 -162.78578898  519.09142006  289.78578898]
        [-510.63918081 -221.73218935  637.63918081  348.73218935]
        [-192.         -192.          319.          319.        ]
        [-258.53978877 -258.53978877  385.53978877  385.53978877]
        [-342.3746693  -342.3746693   469.3746693   469.3746693 ]
        [-118.         -300.          245.          427.        ]
        [-165.30563108 -394.61126216  292.30563108  521.61126216]
        [-224.90699146 -513.81398292  351.90699146  640.81398292]]
return torch.from_numpy(anchors)
                        } // END _generate_anchors(base_size, scales, apect_ratios) END
                    } // END generate_anchors(stride, sizes, aspect_ratios)
cell_anchors: [tensor([[-18.0000,  -8.0000,  25.0000,  15.0000],
    [-23.7183, -11.1191,  30.7183,  18.1191],
    [-30.9228, -15.0488,  37.9228,  22.0488],
    [-12.0000, -12.0000,  19.0000,  19.0000],
    [-16.1587, -16.1587,  23.1587,  23.1587],
    [-21.3984, -21.3984,  28.3984,  28.3984],
    [ -8.0000, -20.0000,  15.0000,  27.0000],
    [-11.1191, -26.2381,  18.1191,  33.2381],
    [-15.0488, -34.0976,  22.0488,  41.0976]]), tensor([[-38.0000, -16.0000,  53.0000,  31.0000],
        [-49.9564, -22.2381,  64.9564,  37.2381],
        [-65.0204, -30.0976,  80.0204,  45.0976],
        [-24.0000, -24.0000,  39.0000,  39.0000],
        [-32.3175, -32.3175,  47.3175,  47.3175],
        [-42.7968, -42.7968,  57.7968,  57.7968],
        [-14.0000, -36.0000,  29.0000,  51.0000],
        [-19.7183, -47.4365,  34.7183,  62.4365],
        [-26.9228, -61.8456,  41.9228,  76.8456]]), tensor([[ -74.0000,  -28.0000,  105.0000,   59.0000],
            [ -97.3929,  -39.4365,  128.3929,   70.4365],
            [-126.8661,  -53.8456,  157.8661,   84.8456],
            [ -48.0000,  -48.0000,   79.0000,   79.0000],
            [ -64.6349,  -64.6349,   95.6349,   95.6349],
            [ -85.5937,  -85.5937,  116.5937,  116.5937],
            [ -30.0000,  -76.0000,   61.0000,  107.0000],
            [ -41.9564,  -99.9127,   72.9564,  130.9127],
            [ -57.0204, -130.0409,   88.0204,  161.0409]]), tensor([[-150.0000,  -60.0000,  213.0000,  123.0000],
                [-197.3056,  -83.9127,  260.3056,  146.9127],
                [-256.9070, -114.0409,  319.9070,  177.0409],
                [ -96.0000,  -96.0000,  159.0000,  159.0000],
                [-129.2699, -129.2699,  192.2699,  192.2699],
                [-171.1873, -171.1873,  234.1873,  234.1873],
                [ -58.0000, -148.0000,  121.0000,  211.0000],
                [ -81.3929, -194.7858,  144.3929,  257.7858],
                [-110.8661, -253.7322,  173.8661,  316.7322]]), tensor([[-298.0000, -116.0000,  425.0000,  243.0000],
                    [-392.0914, -162.7858,  519.0914,  289.7858],
                    [-510.6392, -221.7322,  637.6392,  348.7322],
                    [-192.0000, -192.0000,  319.0000,  319.0000],
                    [-258.5398, -258.5398,  385.5398,  385.5398],
                    [-342.3747, -342.3747,  469.3747,  469.3747],
                    [-118.0000, -300.0000,  245.0000,  427.0000],
                    [-165.3056, -394.6113,  292.3056,  521.6113],
                    [-224.9070, -513.8140,  351.9070,  640.8140]])]
                BufferList.__init__(slef, buffers=None) { // BEGIN
                        // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                        Params
                        len(buffers): 5
                        super(BufferList, self).__init__()
                        if buffers is not None:
                        BufferList.extend(self, buffers) { // BEGIN
                            // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
                            Params
                            len(buffers): 5
                            offset = len(self)
                            len(buffers): 5
                            for i, buffer in enumerate(buffers) { // BEGIN
                                i: 0
                                buffer.shape: torch.Size([9, 4])
                                self.register_buffer(str(offset + i), buffer)
                                i: 1
                                buffer.shape: torch.Size([9, 4])
                                self.register_buffer(str(offset + i), buffer)
                                i: 2
                                buffer.shape: torch.Size([9, 4])
                                self.register_buffer(str(offset + i), buffer)
                                i: 3
                                buffer.shape: torch.Size([9, 4])
                                self.register_buffer(str(offset + i), buffer)
                                i: 4
                                buffer.shape: torch.Size([9, 4])
                                self.register_buffer(str(offset + i), buffer)
                                } // END for i, buffer in enumerate(buffers)
                            self: BufferList()
                            return self
                            } // END BufferList.extend(self, buffers)
                        self.extend(buffers)
                        len(buffers): 5
                        } // END BufferList.__init__(slef, buffers=None)
                self.strides: (8, 16, 32, 64, 128)
    self.cell_anchors: BufferList()
    self.straddle_thresh: -1
        } // END AnchorGenerator.__init__(sizes, aspect_ratios, anchor_strides, straddle_thresh)
        anchor_generator = AnchorGenerator(
                (cell_anchors): BufferList()
                )

        return anchor_generator
    } // make_anchor_generator_retinanet(config) END

    anchor_generator: AnchorGenerator(
            (cell_anchors): BufferList()
            )
    head = RetinaNetHead(cfg, in_channels=1024)


            RetinaNetHead.__init__(cfg, in_channels) { //BEGIN
                    // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/retinanet/retinanet.py
                    Params:
                    cfg:
                    in_channles: 1024:
                    num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
                    num_classes: 1
                    num_anchors = len(cfg.MODEL.RETINANET.ASPECT_RATIOS) \
                            * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE
                            cfg.MODEL.RETINANET.ASPECT_RATIOS: (0.5, 1.0, 2.0)
                            cfg.MODEL.RETINANET.SCALES_PER_OCTAVE: 3
                            num_anchors: 9


                            } // END RetinaNetHead._init__(cfg, in_channels)
            head: RetinaNetHead(
                    (cls_tower): Sequential(
                        (0): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                        (1): ReLU()
                        (2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                        (3): ReLU()
                        (4): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                        (5): ReLU()
                        (6): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                        (7): ReLU()
                        )
                    (bbox_tower): Sequential(
                        (0): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                        (1): ReLU()
                        (2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                        (3): ReLU()
                        (4): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                        (5): ReLU()
                        (6): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                        (7): ReLU()
                        )
                    (cls_logits): Conv2d(1024, 9, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (bbox_pred): Conv2d(1024, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    )
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
    box_coder: <maskrcnn_benchmark.modeling.box_coder.BoxCoder object at 0x7f492f589978>
    box_selector_test = make_retinanet_postprocessor(cfg, box_coder)
RPNPostProcessing.__init__() { //BEGIN
        // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/inference.py
        } // END RPNPostProcessing.__init__()
box_selector_test: RetinaNetPostProcessor()
    self.anchor_generator = anchor_generator
    self.head = head
    self.box_selector_test = box_selector_test


} // RetinaNetModule.__init__(self, cfg, in_channels) END
    } // END build_retinanet(cfg, in_channels)
    self.rpn = build_rpn(cfg, self.backbone.out_channels) // RETURNED
    self.rpn: RetinaNetModule(
            (anchor_generator): AnchorGenerator(
                (cell_anchors): BufferList()
                )
            (head): RetinaNetHead(
                (cls_tower): Sequential(
                    (0): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (1): ReLU()
                    (2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (3): ReLU()
                    (4): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (5): ReLU()
                    (6): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (7): ReLU()
                    )
                (bbox_tower): Sequential(
                    (0): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (1): ReLU()
                    (2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (3): ReLU()
                    (4): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (5): ReLU()
                    (6): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (7): ReLU()
                    )
                (cls_logits): Conv2d(1024, 9, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (bbox_pred): Conv2d(1024, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                )
            (box_selector_test): RetinaNetPostProcessor()
            )
    } END GeneralizedRCNN.__init__(self, cfg)
    self.model.eval()
    checkpointer = DetectronCheckpointer(cfg, self.model, save_dir='/dev/null')
    _ = checkpointer.load(weight)
INFO:maskrcnn_benchmark.utils.checkpoint:Loading checkpoint from ./model/detection/model_det_v2_200924_002_180k.pth
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.0.bn1.bias                  loaded from backbone.body.layer1.0.bn1.bias                  of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.0.bn1.running_mean          loaded from backbone.body.layer1.0.bn1.running_mean          of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.0.bn1.running_var           loaded from backbone.body.layer1.0.bn1.running_var           of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.0.bn1.weight                loaded from backbone.body.layer1.0.bn1.weight                of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.0.bn2.bias                  loaded from backbone.body.layer1.0.bn2.bias                  of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.0.bn2.running_mean          loaded from backbone.body.layer1.0.bn2.running_mean          of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.0.bn2.running_var           loaded from backbone.body.layer1.0.bn2.running_var           of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.0.bn2.weight                loaded from backbone.body.layer1.0.bn2.weight                of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.0.bn3.bias                  loaded from backbone.body.layer1.0.bn3.bias                  of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.0.bn3.running_mean          loaded from backbone.body.layer1.0.bn3.running_mean          of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.0.bn3.running_var           loaded from backbone.body.layer1.0.bn3.running_var           of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.0.bn3.weight                loaded from backbone.body.layer1.0.bn3.weight                of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.0.conv1.weight              loaded from backbone.body.layer1.0.conv1.weight              of shape (64, 64, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.0.conv2.weight              loaded from backbone.body.layer1.0.conv2.weight              of shape (64, 64, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.0.conv3.weight              loaded from backbone.body.layer1.0.conv3.weight              of shape (256, 64, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.0.downsample.0.weight       loaded from backbone.body.layer1.0.downsample.0.weight       of shape (256, 64, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.0.downsample.1.bias         loaded from backbone.body.layer1.0.downsample.1.bias         of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.0.downsample.1.running_mean loaded from backbone.body.layer1.0.downsample.1.running_mean of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.0.downsample.1.running_var  loaded from backbone.body.layer1.0.downsample.1.running_var  of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.0.downsample.1.weight       loaded from backbone.body.layer1.0.downsample.1.weight       of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.1.bn1.bias                  loaded from backbone.body.layer1.1.bn1.bias                  of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.1.bn1.running_mean          loaded from backbone.body.layer1.1.bn1.running_mean          of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.1.bn1.running_var           loaded from backbone.body.layer1.1.bn1.running_var           of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.1.bn1.weight                loaded from backbone.body.layer1.1.bn1.weight                of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.1.bn2.bias                  loaded from backbone.body.layer1.1.bn2.bias                  of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.1.bn2.running_mean          loaded from backbone.body.layer1.1.bn2.running_mean          of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.1.bn2.running_var           loaded from backbone.body.layer1.1.bn2.running_var           of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.1.bn2.weight                loaded from backbone.body.layer1.1.bn2.weight                of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.1.bn3.bias                  loaded from backbone.body.layer1.1.bn3.bias                  of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.1.bn3.running_mean          loaded from backbone.body.layer1.1.bn3.running_mean          of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.1.bn3.running_var           loaded from backbone.body.layer1.1.bn3.running_var           of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.1.bn3.weight                loaded from backbone.body.layer1.1.bn3.weight                of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.1.conv1.weight              loaded from backbone.body.layer1.1.conv1.weight              of shape (64, 256, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.1.conv2.weight              loaded from backbone.body.layer1.1.conv2.weight              of shape (64, 64, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.1.conv3.weight              loaded from backbone.body.layer1.1.conv3.weight              of shape (256, 64, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.2.bn1.bias                  loaded from backbone.body.layer1.2.bn1.bias                  of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.2.bn1.running_mean          loaded from backbone.body.layer1.2.bn1.running_mean          of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.2.bn1.running_var           loaded from backbone.body.layer1.2.bn1.running_var           of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.2.bn1.weight                loaded from backbone.body.layer1.2.bn1.weight                of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.2.bn2.bias                  loaded from backbone.body.layer1.2.bn2.bias                  of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.2.bn2.running_mean          loaded from backbone.body.layer1.2.bn2.running_mean          of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.2.bn2.running_var           loaded from backbone.body.layer1.2.bn2.running_var           of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.2.bn2.weight                loaded from backbone.body.layer1.2.bn2.weight                of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.2.bn3.bias                  loaded from backbone.body.layer1.2.bn3.bias                  of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.2.bn3.running_mean          loaded from backbone.body.layer1.2.bn3.running_mean          of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.2.bn3.running_var           loaded from backbone.body.layer1.2.bn3.running_var           of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.2.bn3.weight                loaded from backbone.body.layer1.2.bn3.weight                of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.2.conv1.weight              loaded from backbone.body.layer1.2.conv1.weight              of shape (64, 256, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.2.conv2.weight              loaded from backbone.body.layer1.2.conv2.weight              of shape (64, 64, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer1.2.conv3.weight              loaded from backbone.body.layer1.2.conv3.weight              of shape (256, 64, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.0.bn1.bias                  loaded from backbone.body.layer2.0.bn1.bias                  of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.0.bn1.running_mean          loaded from backbone.body.layer2.0.bn1.running_mean          of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.0.bn1.running_var           loaded from backbone.body.layer2.0.bn1.running_var           of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.0.bn1.weight                loaded from backbone.body.layer2.0.bn1.weight                of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.0.bn2.bias                  loaded from backbone.body.layer2.0.bn2.bias                  of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.0.bn2.running_mean          loaded from backbone.body.layer2.0.bn2.running_mean          of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.0.bn2.running_var           loaded from backbone.body.layer2.0.bn2.running_var           of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.0.bn2.weight                loaded from backbone.body.layer2.0.bn2.weight                of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.0.bn3.bias                  loaded from backbone.body.layer2.0.bn3.bias                  of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.0.bn3.running_mean          loaded from backbone.body.layer2.0.bn3.running_mean          of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.0.bn3.running_var           loaded from backbone.body.layer2.0.bn3.running_var           of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.0.bn3.weight                loaded from backbone.body.layer2.0.bn3.weight                of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.0.conv1.weight              loaded from backbone.body.layer2.0.conv1.weight              of shape (128, 256, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.0.conv2.weight              loaded from backbone.body.layer2.0.conv2.weight              of shape (128, 128, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.0.conv3.weight              loaded from backbone.body.layer2.0.conv3.weight              of shape (512, 128, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.0.downsample.0.weight       loaded from backbone.body.layer2.0.downsample.0.weight       of shape (512, 256, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.0.downsample.1.bias         loaded from backbone.body.layer2.0.downsample.1.bias         of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.0.downsample.1.running_mean loaded from backbone.body.layer2.0.downsample.1.running_mean of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.0.downsample.1.running_var  loaded from backbone.body.layer2.0.downsample.1.running_var  of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.0.downsample.1.weight       loaded from backbone.body.layer2.0.downsample.1.weight       of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.1.bn1.bias                  loaded from backbone.body.layer2.1.bn1.bias                  of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.1.bn1.running_mean          loaded from backbone.body.layer2.1.bn1.running_mean          of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.1.bn1.running_var           loaded from backbone.body.layer2.1.bn1.running_var           of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.1.bn1.weight                loaded from backbone.body.layer2.1.bn1.weight                of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.1.bn2.bias                  loaded from backbone.body.layer2.1.bn2.bias                  of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.1.bn2.running_mean          loaded from backbone.body.layer2.1.bn2.running_mean          of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.1.bn2.running_var           loaded from backbone.body.layer2.1.bn2.running_var           of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.1.bn2.weight                loaded from backbone.body.layer2.1.bn2.weight                of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.1.bn3.bias                  loaded from backbone.body.layer2.1.bn3.bias                  of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.1.bn3.running_mean          loaded from backbone.body.layer2.1.bn3.running_mean          of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.1.bn3.running_var           loaded from backbone.body.layer2.1.bn3.running_var           of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.1.bn3.weight                loaded from backbone.body.layer2.1.bn3.weight                of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.1.conv1.weight              loaded from backbone.body.layer2.1.conv1.weight              of shape (128, 512, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.1.conv2.weight              loaded from backbone.body.layer2.1.conv2.weight              of shape (128, 128, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.1.conv3.weight              loaded from backbone.body.layer2.1.conv3.weight              of shape (512, 128, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.2.bn1.bias                  loaded from backbone.body.layer2.2.bn1.bias                  of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.2.bn1.running_mean          loaded from backbone.body.layer2.2.bn1.running_mean          of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.2.bn1.running_var           loaded from backbone.body.layer2.2.bn1.running_var           of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.2.bn1.weight                loaded from backbone.body.layer2.2.bn1.weight                of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.2.bn2.bias                  loaded from backbone.body.layer2.2.bn2.bias                  of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.2.bn2.running_mean          loaded from backbone.body.layer2.2.bn2.running_mean          of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.2.bn2.running_var           loaded from backbone.body.layer2.2.bn2.running_var           of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.2.bn2.weight                loaded from backbone.body.layer2.2.bn2.weight                of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.2.bn3.bias                  loaded from backbone.body.layer2.2.bn3.bias                  of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.2.bn3.running_mean          loaded from backbone.body.layer2.2.bn3.running_mean          of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.2.bn3.running_var           loaded from backbone.body.layer2.2.bn3.running_var           of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.2.bn3.weight                loaded from backbone.body.layer2.2.bn3.weight                of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.2.conv1.weight              loaded from backbone.body.layer2.2.conv1.weight              of shape (128, 512, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.2.conv2.weight              loaded from backbone.body.layer2.2.conv2.weight              of shape (128, 128, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.2.conv3.weight              loaded from backbone.body.layer2.2.conv3.weight              of shape (512, 128, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.3.bn1.bias                  loaded from backbone.body.layer2.3.bn1.bias                  of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.3.bn1.running_mean          loaded from backbone.body.layer2.3.bn1.running_mean          of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.3.bn1.running_var           loaded from backbone.body.layer2.3.bn1.running_var           of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.3.bn1.weight                loaded from backbone.body.layer2.3.bn1.weight                of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.3.bn2.bias                  loaded from backbone.body.layer2.3.bn2.bias                  of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.3.bn2.running_mean          loaded from backbone.body.layer2.3.bn2.running_mean          of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.3.bn2.running_var           loaded from backbone.body.layer2.3.bn2.running_var           of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.3.bn2.weight                loaded from backbone.body.layer2.3.bn2.weight                of shape (128,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.3.bn3.bias                  loaded from backbone.body.layer2.3.bn3.bias                  of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.3.bn3.running_mean          loaded from backbone.body.layer2.3.bn3.running_mean          of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.3.bn3.running_var           loaded from backbone.body.layer2.3.bn3.running_var           of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.3.bn3.weight                loaded from backbone.body.layer2.3.bn3.weight                of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.3.conv1.weight              loaded from backbone.body.layer2.3.conv1.weight              of shape (128, 512, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.3.conv2.weight              loaded from backbone.body.layer2.3.conv2.weight              of shape (128, 128, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer2.3.conv3.weight              loaded from backbone.body.layer2.3.conv3.weight              of shape (512, 128, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.0.bn1.bias                  loaded from backbone.body.layer3.0.bn1.bias                  of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.0.bn1.running_mean          loaded from backbone.body.layer3.0.bn1.running_mean          of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.0.bn1.running_var           loaded from backbone.body.layer3.0.bn1.running_var           of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.0.bn1.weight                loaded from backbone.body.layer3.0.bn1.weight                of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.0.bn2.bias                  loaded from backbone.body.layer3.0.bn2.bias                  of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.0.bn2.running_mean          loaded from backbone.body.layer3.0.bn2.running_mean          of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.0.bn2.running_var           loaded from backbone.body.layer3.0.bn2.running_var           of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.0.bn2.weight                loaded from backbone.body.layer3.0.bn2.weight                of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.0.bn3.bias                  loaded from backbone.body.layer3.0.bn3.bias                  of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.0.bn3.running_mean          loaded from backbone.body.layer3.0.bn3.running_mean          of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.0.bn3.running_var           loaded from backbone.body.layer3.0.bn3.running_var           of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.0.bn3.weight                loaded from backbone.body.layer3.0.bn3.weight                of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.0.conv1.weight              loaded from backbone.body.layer3.0.conv1.weight              of shape (256, 512, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.0.conv2.weight              loaded from backbone.body.layer3.0.conv2.weight              of shape (256, 256, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.0.conv3.weight              loaded from backbone.body.layer3.0.conv3.weight              of shape (1024, 256, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.0.downsample.0.weight       loaded from backbone.body.layer3.0.downsample.0.weight       of shape (1024, 512, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.0.downsample.1.bias         loaded from backbone.body.layer3.0.downsample.1.bias         of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.0.downsample.1.running_mean loaded from backbone.body.layer3.0.downsample.1.running_mean of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.0.downsample.1.running_var  loaded from backbone.body.layer3.0.downsample.1.running_var  of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.0.downsample.1.weight       loaded from backbone.body.layer3.0.downsample.1.weight       of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.1.bn1.bias                  loaded from backbone.body.layer3.1.bn1.bias                  of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.1.bn1.running_mean          loaded from backbone.body.layer3.1.bn1.running_mean          of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.1.bn1.running_var           loaded from backbone.body.layer3.1.bn1.running_var           of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.1.bn1.weight                loaded from backbone.body.layer3.1.bn1.weight                of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.1.bn2.bias                  loaded from backbone.body.layer3.1.bn2.bias                  of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.1.bn2.running_mean          loaded from backbone.body.layer3.1.bn2.running_mean          of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.1.bn2.running_var           loaded from backbone.body.layer3.1.bn2.running_var           of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.1.bn2.weight                loaded from backbone.body.layer3.1.bn2.weight                of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.1.bn3.bias                  loaded from backbone.body.layer3.1.bn3.bias                  of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.1.bn3.running_mean          loaded from backbone.body.layer3.1.bn3.running_mean          of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.1.bn3.running_var           loaded from backbone.body.layer3.1.bn3.running_var           of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.1.bn3.weight                loaded from backbone.body.layer3.1.bn3.weight                of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.1.conv1.weight              loaded from backbone.body.layer3.1.conv1.weight              of shape (256, 1024, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.1.conv2.weight              loaded from backbone.body.layer3.1.conv2.weight              of shape (256, 256, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.1.conv3.weight              loaded from backbone.body.layer3.1.conv3.weight              of shape (1024, 256, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.2.bn1.bias                  loaded from backbone.body.layer3.2.bn1.bias                  of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.2.bn1.running_mean          loaded from backbone.body.layer3.2.bn1.running_mean          of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.2.bn1.running_var           loaded from backbone.body.layer3.2.bn1.running_var           of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.2.bn1.weight                loaded from backbone.body.layer3.2.bn1.weight                of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.2.bn2.bias                  loaded from backbone.body.layer3.2.bn2.bias                  of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.2.bn2.running_mean          loaded from backbone.body.layer3.2.bn2.running_mean          of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.2.bn2.running_var           loaded from backbone.body.layer3.2.bn2.running_var           of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.2.bn2.weight                loaded from backbone.body.layer3.2.bn2.weight                of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.2.bn3.bias                  loaded from backbone.body.layer3.2.bn3.bias                  of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.2.bn3.running_mean          loaded from backbone.body.layer3.2.bn3.running_mean          of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.2.bn3.running_var           loaded from backbone.body.layer3.2.bn3.running_var           of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.2.bn3.weight                loaded from backbone.body.layer3.2.bn3.weight                of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.2.conv1.weight              loaded from backbone.body.layer3.2.conv1.weight              of shape (256, 1024, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.2.conv2.weight              loaded from backbone.body.layer3.2.conv2.weight              of shape (256, 256, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.2.conv3.weight              loaded from backbone.body.layer3.2.conv3.weight              of shape (1024, 256, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.3.bn1.bias                  loaded from backbone.body.layer3.3.bn1.bias                  of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.3.bn1.running_mean          loaded from backbone.body.layer3.3.bn1.running_mean          of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.3.bn1.running_var           loaded from backbone.body.layer3.3.bn1.running_var           of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.3.bn1.weight                loaded from backbone.body.layer3.3.bn1.weight                of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.3.bn2.bias                  loaded from backbone.body.layer3.3.bn2.bias                  of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.3.bn2.running_mean          loaded from backbone.body.layer3.3.bn2.running_mean          of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.3.bn2.running_var           loaded from backbone.body.layer3.3.bn2.running_var           of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.3.bn2.weight                loaded from backbone.body.layer3.3.bn2.weight                of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.3.bn3.bias                  loaded from backbone.body.layer3.3.bn3.bias                  of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.3.bn3.running_mean          loaded from backbone.body.layer3.3.bn3.running_mean          of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.3.bn3.running_var           loaded from backbone.body.layer3.3.bn3.running_var           of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.3.bn3.weight                loaded from backbone.body.layer3.3.bn3.weight                of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.3.conv1.weight              loaded from backbone.body.layer3.3.conv1.weight              of shape (256, 1024, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.3.conv2.weight              loaded from backbone.body.layer3.3.conv2.weight              of shape (256, 256, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.3.conv3.weight              loaded from backbone.body.layer3.3.conv3.weight              of shape (1024, 256, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.4.bn1.bias                  loaded from backbone.body.layer3.4.bn1.bias                  of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.4.bn1.running_mean          loaded from backbone.body.layer3.4.bn1.running_mean          of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.4.bn1.running_var           loaded from backbone.body.layer3.4.bn1.running_var           of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.4.bn1.weight                loaded from backbone.body.layer3.4.bn1.weight                of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.4.bn2.bias                  loaded from backbone.body.layer3.4.bn2.bias                  of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.4.bn2.running_mean          loaded from backbone.body.layer3.4.bn2.running_mean          of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.4.bn2.running_var           loaded from backbone.body.layer3.4.bn2.running_var           of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.4.bn2.weight                loaded from backbone.body.layer3.4.bn2.weight                of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.4.bn3.bias                  loaded from backbone.body.layer3.4.bn3.bias                  of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.4.bn3.running_mean          loaded from backbone.body.layer3.4.bn3.running_mean          of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.4.bn3.running_var           loaded from backbone.body.layer3.4.bn3.running_var           of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.4.bn3.weight                loaded from backbone.body.layer3.4.bn3.weight                of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.4.conv1.weight              loaded from backbone.body.layer3.4.conv1.weight              of shape (256, 1024, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.4.conv2.weight              loaded from backbone.body.layer3.4.conv2.weight              of shape (256, 256, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.4.conv3.weight              loaded from backbone.body.layer3.4.conv3.weight              of shape (1024, 256, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.5.bn1.bias                  loaded from backbone.body.layer3.5.bn1.bias                  of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.5.bn1.running_mean          loaded from backbone.body.layer3.5.bn1.running_mean          of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.5.bn1.running_var           loaded from backbone.body.layer3.5.bn1.running_var           of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.5.bn1.weight                loaded from backbone.body.layer3.5.bn1.weight                of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.5.bn2.bias                  loaded from backbone.body.layer3.5.bn2.bias                  of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.5.bn2.running_mean          loaded from backbone.body.layer3.5.bn2.running_mean          of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.5.bn2.running_var           loaded from backbone.body.layer3.5.bn2.running_var           of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.5.bn2.weight                loaded from backbone.body.layer3.5.bn2.weight                of shape (256,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.5.bn3.bias                  loaded from backbone.body.layer3.5.bn3.bias                  of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.5.bn3.running_mean          loaded from backbone.body.layer3.5.bn3.running_mean          of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.5.bn3.running_var           loaded from backbone.body.layer3.5.bn3.running_var           of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.5.bn3.weight                loaded from backbone.body.layer3.5.bn3.weight                of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.5.conv1.weight              loaded from backbone.body.layer3.5.conv1.weight              of shape (256, 1024, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.5.conv2.weight              loaded from backbone.body.layer3.5.conv2.weight              of shape (256, 256, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer3.5.conv3.weight              loaded from backbone.body.layer3.5.conv3.weight              of shape (1024, 256, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.0.bn1.bias                  loaded from backbone.body.layer4.0.bn1.bias                  of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.0.bn1.running_mean          loaded from backbone.body.layer4.0.bn1.running_mean          of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.0.bn1.running_var           loaded from backbone.body.layer4.0.bn1.running_var           of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.0.bn1.weight                loaded from backbone.body.layer4.0.bn1.weight                of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.0.bn2.bias                  loaded from backbone.body.layer4.0.bn2.bias                  of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.0.bn2.running_mean          loaded from backbone.body.layer4.0.bn2.running_mean          of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.0.bn2.running_var           loaded from backbone.body.layer4.0.bn2.running_var           of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.0.bn2.weight                loaded from backbone.body.layer4.0.bn2.weight                of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.0.bn3.bias                  loaded from backbone.body.layer4.0.bn3.bias                  of shape (2048,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.0.bn3.running_mean          loaded from backbone.body.layer4.0.bn3.running_mean          of shape (2048,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.0.bn3.running_var           loaded from backbone.body.layer4.0.bn3.running_var           of shape (2048,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.0.bn3.weight                loaded from backbone.body.layer4.0.bn3.weight                of shape (2048,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.0.conv1.weight              loaded from backbone.body.layer4.0.conv1.weight              of shape (512, 1024, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.0.conv2.weight              loaded from backbone.body.layer4.0.conv2.weight              of shape (512, 512, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.0.conv3.weight              loaded from backbone.body.layer4.0.conv3.weight              of shape (2048, 512, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.0.downsample.0.weight       loaded from backbone.body.layer4.0.downsample.0.weight       of shape (2048, 1024, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.0.downsample.1.bias         loaded from backbone.body.layer4.0.downsample.1.bias         of shape (2048,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.0.downsample.1.running_mean loaded from backbone.body.layer4.0.downsample.1.running_mean of shape (2048,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.0.downsample.1.running_var  loaded from backbone.body.layer4.0.downsample.1.running_var  of shape (2048,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.0.downsample.1.weight       loaded from backbone.body.layer4.0.downsample.1.weight       of shape (2048,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.1.bn1.bias                  loaded from backbone.body.layer4.1.bn1.bias                  of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.1.bn1.running_mean          loaded from backbone.body.layer4.1.bn1.running_mean          of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.1.bn1.running_var           loaded from backbone.body.layer4.1.bn1.running_var           of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.1.bn1.weight                loaded from backbone.body.layer4.1.bn1.weight                of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.1.bn2.bias                  loaded from backbone.body.layer4.1.bn2.bias                  of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.1.bn2.running_mean          loaded from backbone.body.layer4.1.bn2.running_mean          of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.1.bn2.running_var           loaded from backbone.body.layer4.1.bn2.running_var           of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.1.bn2.weight                loaded from backbone.body.layer4.1.bn2.weight                of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.1.bn3.bias                  loaded from backbone.body.layer4.1.bn3.bias                  of shape (2048,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.1.bn3.running_mean          loaded from backbone.body.layer4.1.bn3.running_mean          of shape (2048,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.1.bn3.running_var           loaded from backbone.body.layer4.1.bn3.running_var           of shape (2048,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.1.bn3.weight                loaded from backbone.body.layer4.1.bn3.weight                of shape (2048,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.1.conv1.weight              loaded from backbone.body.layer4.1.conv1.weight              of shape (512, 2048, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.1.conv2.weight              loaded from backbone.body.layer4.1.conv2.weight              of shape (512, 512, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.1.conv3.weight              loaded from backbone.body.layer4.1.conv3.weight              of shape (2048, 512, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.2.bn1.bias                  loaded from backbone.body.layer4.2.bn1.bias                  of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.2.bn1.running_mean          loaded from backbone.body.layer4.2.bn1.running_mean          of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.2.bn1.running_var           loaded from backbone.body.layer4.2.bn1.running_var           of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.2.bn1.weight                loaded from backbone.body.layer4.2.bn1.weight                of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.2.bn2.bias                  loaded from backbone.body.layer4.2.bn2.bias                  of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.2.bn2.running_mean          loaded from backbone.body.layer4.2.bn2.running_mean          of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.2.bn2.running_var           loaded from backbone.body.layer4.2.bn2.running_var           of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.2.bn2.weight                loaded from backbone.body.layer4.2.bn2.weight                of shape (512,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.2.bn3.bias                  loaded from backbone.body.layer4.2.bn3.bias                  of shape (2048,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.2.bn3.running_mean          loaded from backbone.body.layer4.2.bn3.running_mean          of shape (2048,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.2.bn3.running_var           loaded from backbone.body.layer4.2.bn3.running_var           of shape (2048,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.2.bn3.weight                loaded from backbone.body.layer4.2.bn3.weight                of shape (2048,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.2.conv1.weight              loaded from backbone.body.layer4.2.conv1.weight              of shape (512, 2048, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.2.conv2.weight              loaded from backbone.body.layer4.2.conv2.weight              of shape (512, 512, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.layer4.2.conv3.weight              loaded from backbone.body.layer4.2.conv3.weight              of shape (2048, 512, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.stem.bn1.bias                      loaded from backbone.body.stem.bn1.bias                      of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.stem.bn1.running_mean              loaded from backbone.body.stem.bn1.running_mean              of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.stem.bn1.running_var               loaded from backbone.body.stem.bn1.running_var               of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.stem.bn1.weight                    loaded from backbone.body.stem.bn1.weight                    of shape (64,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.body.stem.conv1.weight                  loaded from backbone.body.stem.conv1.weight                  of shape (64, 3, 7, 7)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.fpn.fpn_inner2.bias                     loaded from backbone.fpn.fpn_inner2.bias                     of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.fpn.fpn_inner2.weight                   loaded from backbone.fpn.fpn_inner2.weight                   of shape (1024, 512, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.fpn.fpn_inner3.bias                     loaded from backbone.fpn.fpn_inner3.bias                     of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.fpn.fpn_inner3.weight                   loaded from backbone.fpn.fpn_inner3.weight                   of shape (1024, 1024, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.fpn.fpn_inner4.bias                     loaded from backbone.fpn.fpn_inner4.bias                     of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.fpn.fpn_inner4.weight                   loaded from backbone.fpn.fpn_inner4.weight                   of shape (1024, 2048, 1, 1)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.fpn.fpn_layer2.bias                     loaded from backbone.fpn.fpn_layer2.bias                     of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.fpn.fpn_layer2.weight                   loaded from backbone.fpn.fpn_layer2.weight                   of shape (1024, 1024, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.fpn.fpn_layer3.bias                     loaded from backbone.fpn.fpn_layer3.bias                     of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.fpn.fpn_layer3.weight                   loaded from backbone.fpn.fpn_layer3.weight                   of shape (1024, 1024, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.fpn.fpn_layer4.bias                     loaded from backbone.fpn.fpn_layer4.bias                     of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.fpn.fpn_layer4.weight                   loaded from backbone.fpn.fpn_layer4.weight                   of shape (1024, 1024, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.fpn.top_blocks.p6.bias                  loaded from backbone.fpn.top_blocks.p6.bias                  of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.fpn.top_blocks.p6.weight                loaded from backbone.fpn.top_blocks.p6.weight                of shape (1024, 2048, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.fpn.top_blocks.p7.bias                  loaded from backbone.fpn.top_blocks.p7.bias                  of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:backbone.fpn.top_blocks.p7.weight                loaded from backbone.fpn.top_blocks.p7.weight                of shape (1024, 1024, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.anchor_generator.cell_anchors.0              loaded from rpn.anchor_generator.cell_anchors.0              of shape (9, 4)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.anchor_generator.cell_anchors.1              loaded from rpn.anchor_generator.cell_anchors.1              of shape (9, 4)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.anchor_generator.cell_anchors.2              loaded from rpn.anchor_generator.cell_anchors.2              of shape (9, 4)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.anchor_generator.cell_anchors.3              loaded from rpn.anchor_generator.cell_anchors.3              of shape (9, 4)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.anchor_generator.cell_anchors.4              loaded from rpn.anchor_generator.cell_anchors.4              of shape (9, 4)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.head.bbox_pred.bias                          loaded from rpn.head.bbox_pred.bias                          of shape (36,)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.head.bbox_pred.weight                        loaded from rpn.head.bbox_pred.weight                        of shape (36, 1024, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.head.bbox_tower.0.bias                       loaded from rpn.head.bbox_tower.0.bias                       of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.head.bbox_tower.0.weight                     loaded from rpn.head.bbox_tower.0.weight                     of shape (1024, 1024, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.head.bbox_tower.2.bias                       loaded from rpn.head.bbox_tower.2.bias                       of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.head.bbox_tower.2.weight                     loaded from rpn.head.bbox_tower.2.weight                     of shape (1024, 1024, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.head.bbox_tower.4.bias                       loaded from rpn.head.bbox_tower.4.bias                       of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.head.bbox_tower.4.weight                     loaded from rpn.head.bbox_tower.4.weight                     of shape (1024, 1024, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.head.bbox_tower.6.bias                       loaded from rpn.head.bbox_tower.6.bias                       of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.head.bbox_tower.6.weight                     loaded from rpn.head.bbox_tower.6.weight                     of shape (1024, 1024, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.head.cls_logits.bias                         loaded from rpn.head.cls_logits.bias                         of shape (9,)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.head.cls_logits.weight                       loaded from rpn.head.cls_logits.weight                       of shape (9, 1024, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.head.cls_tower.0.bias                        loaded from rpn.head.cls_tower.0.bias                        of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.head.cls_tower.0.weight                      loaded from rpn.head.cls_tower.0.weight                      of shape (1024, 1024, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.head.cls_tower.2.bias                        loaded from rpn.head.cls_tower.2.bias                        of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.head.cls_tower.2.weight                      loaded from rpn.head.cls_tower.2.weight                      of shape (1024, 1024, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.head.cls_tower.4.bias                        loaded from rpn.head.cls_tower.4.bias                        of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.head.cls_tower.4.weight                      loaded from rpn.head.cls_tower.4.weight                      of shape (1024, 1024, 3, 3)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.head.cls_tower.6.bias                        loaded from rpn.head.cls_tower.6.bias                        of shape (1024,)
INFO:maskrcnn_benchmark.utils.model_serialization:rpn.head.cls_tower.6.weight                      loaded from rpn.head.cls_tower.6.weight                      of shape (1024, 1024, 3, 3)
    self.transforms = build_transforms(self.cfg, self.is_recognition)
} // END DetectionDemo.__init__
compute_prediction(self, image) { // BEGIN
    // defined in detection_model_debug.py

    Params:
        image: H, W=(438,512)

        transforms.py Compose class __call__  ====== BEGIN

        for t in self.transforms:
            image = <maskrcnn_benchmark.data.transforms.transforms.Resize object at 0x7f4932d0c080>(image)
            image = <maskrcnn_benchmark.data.transforms.transforms.ToTensor object at 0x7f4932d0c128>(image)
            image = <maskrcnn_benchmark.data.transforms.transforms.Normalize object at 0x7f4932d0c0b8>(image)

        return image
        transforms.py Compose class __call__  ====== END
    image_tensor = self.transforms(image)

    image_tensor.shape: torch.Size([3, 480, 561])

    padding images for 32 divisible size on width and height
    image_list = to_image_list(image_tensor, 32).to(self.device)

    to_image_list(tensors, size_divisible=32) ====== BEGIN
        type(batched_imgs): <class 'torch.Tensor'>
        batched_imgs.shape: torch.Size([1, 3, 480, 576])
        image_sizes: [torch.Size([480, 561])]
        return ImageList(batched_imgs, image_sizes)
    to_image_list(tensors, size_divisible=32) ====== END

    image_list.image_sizes: [torch.Size([480, 561])]
    image_list.tensors.shape: torch.Size([1, 3, 480, 576])
    pred = self.model(image_list)


    GeneralizedRCNN.forward(self, images, targets=None) { //BEGIN
    // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/detector/generalized_rcnn.py
        Params:
            images:
                type(images): <class 'maskrcnn_benchmark.structures.image_list.ImageList'>
            targets: None
    if self.training: False: 
    images = to_image_list(images)

    to_image_list(tensors, size_divisible=0) ====== BEGIN
        if isinstance(tensors, ImageList):
        return tensors
    to_image_list(tensors, size_divisible=0) ====== END

    images.image_sizes: [torch.Size([480, 561])]
    images.tensors.shape: torch.Size([1, 3, 480, 576])
    model.backbone.forward(images.tensors) BEFORE

    Resnet.forward(self, x) { //BEGIN
        Param
            x.shape=torch.Size([1, 3, 480, 576])

        x = self.stem(x)
        x.shape: torch.Size([1, 64, 120, 144])
        stem output of shape (1, 64, 120, 144) saved into ./npy_save/stem_output.npy


        for stage_name in self.stages:
            stage_name: layer1
                output shape of layer1: torch.Size([1, 256, 120, 144])
            outputs.append(x) stage_name: layer1
            x.shape: torch.Size([1, 256, 120, 144])
        layer1 output of shape (1, 256, 120, 144) saved into ./npy_save/layer1_output.npy


            stage_name: layer2
                output shape of layer2: torch.Size([1, 512, 60, 72])
            outputs.append(x) stage_name: layer2
            x.shape: torch.Size([1, 512, 60, 72])
        layer2 output of shape (1, 512, 60, 72) saved into ./npy_save/layer2_output.npy


            stage_name: layer3
                output shape of layer3: torch.Size([1, 1024, 30, 36])
            outputs.append(x) stage_name: layer3
            x.shape: torch.Size([1, 1024, 30, 36])
        layer3 output of shape (1, 1024, 30, 36) saved into ./npy_save/layer3_output.npy


            stage_name: layer4
                output shape of layer4: torch.Size([1, 2048, 15, 18])
            outputs.append(x) stage_name: layer4
            x.shape: torch.Size([1, 2048, 15, 18])
        layer4 output of shape (1, 2048, 15, 18) saved into ./npy_save/layer4_output.npy



        ResNet::forward return value
            outputs[0]: torch.Size([1, 256, 120, 144])
            outputs[1]: torch.Size([1, 512, 60, 72])
            outputs[2]: torch.Size([1, 1024, 30, 36])
            outputs[3]: torch.Size([1, 2048, 15, 18])

        return outputs

    } // END Resnet.forward()


    FPN.forward(self,x) { // BEGIN
        Param: x  = [C2, C3, C4, C5] 
            len(x) = 4
            C[1].shape : torch.Size([1, 256, 120, 144])
            C[2].shape : torch.Size([1, 512, 60, 72])
            C[3].shape : torch.Size([1, 1024, 30, 36])
            C[4].shape : torch.Size([1, 2048, 15, 18])

            x[-1].shape = torch.Size([1, 2048, 15, 18])

            ===========================================================================
            FPN block info
            self.inner_blocks: ['fpn_inner2', 'fpn_inner3', 'fpn_inner4'])
            self.layer_blocks: ['fpn_layer2', 'fpn_layer3', 'fpn_layer4'])
            ===========================================================================

    last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        self.innerblocks[-1] = Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))
        x[-1].shape = torch.Size([1, 2048, 15, 18])
        last_inner.shape = torch.Size([1, 1024, 15, 18])

    fpn_inner4 output of shape (1, 1024, 15, 18) saved into ./npy_save/fpn_inner4_output.npy



    results.append(fpn_layer4(last_inner))
        self.layer_blocks[-1]: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        results[0].shape: torch.Size([1, 1024, 15, 18])

        fpn_layer4 output of shape (1, 1024, 15, 18) saved into ./npy_save/fpn_layer4_output.npy


    for feature, inner_block, layer_block
            in zip[(x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]):

        ====================================
        iteration 0 summary
        ====================================
        feature.shape: torch.Size([1, 1024, 30, 36])
        inner_block: fpn_inner3 ==> Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
        layer_block: fpn_layer3 ==> Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        last_inner.shape: torch.Size([1, 1024, 15, 18])
        ====================================

        --------------------------------------------------
        0.1 Upsample : replace with Decovolution in caffe
        layer name in caffe: fpn_inner3_upsample = Deconvolution(last_inner)
        --------------------------------------------------
        inner_top_down = F.interpolate(last_inner, scale_factor=2, mode='nearest'
        last_inner.shape: torch.Size([1, 1024, 15, 18])
        inner_top_down.shape : torch.Size([1, 1024, 30, 36])
        --------------------------------------------------

        inner_top_down of shape (1, 1024, 30, 36) saved into ./npy_save/inner_top_down_forfpn_inner3.npy


        --------------------------------------------------
        0.2 inner_lateral = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))(feature)
        layer name in caffe: fpn_inner3_lateral=Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))(feature)
        --------------------------------------------------
            inner_block: fpn_inner3 ==> Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
            input: feature.shape: torch.Size([1, 1024, 30, 36])
            output: inner_lateral.shape: torch.Size([1, 1024, 30, 36])

        --------------------------------------------------

        fpn_inner3 output of shape (1, 1024, 30, 36) saved into ./npy_save/fpn_inner3_output.npy


        --------------------------------------------------
        0.3 Elementwise Addition: replaced with eltwise in caffe
        layer in caffe: eltwise_3 = eltwise(fpn_inner3_lateral, fpn_inner3_upsample )
        --------------------------------------------------
        last_inner = inner_lateral + inner_top_down
            inner_lateral.shape: torch.Size([1, 1024, 30, 36])
            inner_top_down.shape: torch.Size([1, 1024, 30, 36])
            last_inner.shape : torch.Size([1, 1024, 30, 36])
        --------------------------------------------------

        superimposing result of fpn_inner3 output plus inner topdown of shape (1, 1024, 30, 36) saved into ./npy_save/fpn_inner3_ouptut_plus_inner_topdown.npy


        --------------------------------------------------
        0.4 results.insert(0, Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(last_inner)
        layer in caffe: fpn_layer3 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(eltwise_3)
        --------------------------------------------------
            layer_block: fpn_layer3 ==> Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            input: last_inner.shape = torch.Size([1, 1024, 30, 36])
        --------------------------------------------------

        fpn_layer3 output of shape (1, 1024, 30, 36) saved into ./npy_save/fpn_layer3_ouptut.npy


        --------------------------------------------------
        results after iteration 0
        --------------------------------------------------
            results[0].shape: torch.Size([1, 1024, 30, 36])
            results[1].shape: torch.Size([1, 1024, 15, 18])
        --------------------------------------------------

        ====================================
        iteration 1 summary
        ====================================
        feature.shape: torch.Size([1, 512, 60, 72])
        inner_block: fpn_inner2 ==> Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
        layer_block: fpn_layer2 ==> Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        last_inner.shape: torch.Size([1, 1024, 30, 36])
        ====================================

        --------------------------------------------------
        1.1 Upsample : replace with Decovolution in caffe
        layer name in caffe: fpn_inner2_upsample = Deconvolution(last_inner)
        --------------------------------------------------
        inner_top_down = F.interpolate(last_inner, scale_factor=2, mode='nearest'
        last_inner.shape: torch.Size([1, 1024, 30, 36])
        inner_top_down.shape : torch.Size([1, 1024, 60, 72])
        --------------------------------------------------

        inner_top_down of shape (1, 1024, 60, 72) saved into ./npy_save/inner_top_down_forfpn_inner2.npy


        --------------------------------------------------
        1.2 inner_lateral = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))(feature)
        layer name in caffe: fpn_inner2_lateral=Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))(feature)
        --------------------------------------------------
            inner_block: fpn_inner2 ==> Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
            input: feature.shape: torch.Size([1, 512, 60, 72])
            output: inner_lateral.shape: torch.Size([1, 1024, 60, 72])

        --------------------------------------------------

        fpn_inner2 output of shape (1, 1024, 60, 72) saved into ./npy_save/fpn_inner2_output.npy


        --------------------------------------------------
        1.3 Elementwise Addition: replaced with eltwise in caffe
        layer in caffe: eltwise_2 = eltwise(fpn_inner2_lateral, fpn_inner2_upsample )
        --------------------------------------------------
        last_inner = inner_lateral + inner_top_down
            inner_lateral.shape: torch.Size([1, 1024, 60, 72])
            inner_top_down.shape: torch.Size([1, 1024, 60, 72])
            last_inner.shape : torch.Size([1, 1024, 60, 72])
        --------------------------------------------------

        superimposing result of fpn_inner2 output plus inner topdown of shape (1, 1024, 60, 72) saved into ./npy_save/fpn_inner2_ouptut_plus_inner_topdown.npy


        --------------------------------------------------
        1.4 results.insert(0, Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(last_inner)
        layer in caffe: fpn_layer2 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(eltwise_2)
        --------------------------------------------------
            layer_block: fpn_layer2 ==> Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            input: last_inner.shape = torch.Size([1, 1024, 60, 72])
        --------------------------------------------------

        fpn_layer2 output of shape (1, 1024, 60, 72) saved into ./npy_save/fpn_layer2_ouptut.npy


        --------------------------------------------------
        results after iteration 1
        --------------------------------------------------
            results[0].shape: torch.Size([1, 1024, 60, 72])
            results[1].shape: torch.Size([1, 1024, 30, 36])
            results[2].shape: torch.Size([1, 1024, 15, 18])
        --------------------------------------------------

    for loop END


    if isinstance(self.top_blocks, LastLevelP6P7):
            self.top_blocks: LastLevelP6P7(
  (p6): Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (p7): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
)
            len(x): 4
            x[0].shape : torch.Size([1, 256, 120, 144])
            x[1].shape : torch.Size([1, 512, 60, 72])
            x[2].shape : torch.Size([1, 1024, 30, 36])
            x[3].shape : torch.Size([1, 2048, 15, 18])
            x[-1].shape: torch.Size([1, 2048, 15, 18])


            len(results): 3
            results[0].shape : torch.Size([1, 1024, 60, 72])
            results[1].shape : torch.Size([1, 1024, 30, 36])
            results[2].shape : torch.Size([1, 1024, 15, 18])
            results[-1].shape: torch.Size([1, 1024, 15, 18])



            LastLevelP6P7.forward(self, c5, p5) { // BEGIN 

                Param:
                    c5.shape: torch.Size([1, 2048, 15, 18])
                    p5.shape: torch.Size([1, 1024, 15, 18])

                if (self.use_P5 == False)
                    x=c5
                x.shape = torch.Size([1, 2048, 15, 18])
                p6 = self.p6(x)
                    self.p6: Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                    p6.shape: torch.Size([1, 1024, 8, 9])

                LastLevelP6P7::forward() self.p6 ==> Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) output of shape (1, 1024, 8, 9) saved into ./npy_save/P6.npy


                p7 = self.p7(F.relu(p6))
                    self.p7: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                    p7.shape: torch.Size([1, 1024, 4, 5])

            LastLevelP6P7::forward() self.p7 Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))(F.relu(p6)) output of shape (1, 1024, 4, 5) saved into ./npy_save/P7.npy


                returns [p6, p7]
            } // END LastLevelP6P7.forward(self, c5, p5)


        last_result = self.top_blocks(x[-1], results[-1])
            x[-1] => c5, results[-1])=> p5
            len(last_results):2
            last_results[0].shape : torch.Size([1, 1024, 8, 9])
            last_results[1].shape : torch.Size([1, 1024, 4, 5])
        results.extend(last_results)
            len(results): 5
            results[0].shape : torch.Size([1, 1024, 60, 72])
            results[1].shape : torch.Size([1, 1024, 30, 36])
            results[2].shape : torch.Size([1, 1024, 15, 18])
            results[3].shape : torch.Size([1, 1024, 8, 9])
            results[4].shape : torch.Size([1, 1024, 4, 5])




        results
        result[0].shape: torch.Size([1, 1024, 60, 72])
        result[1].shape: torch.Size([1, 1024, 30, 36])
        result[2].shape: torch.Size([1, 1024, 15, 18])
        result[3].shape: torch.Size([1, 1024, 8, 9])
        result[4].shape: torch.Size([1, 1024, 4, 5])

    return tuple(results)


    } // END FPN.forward(self,x)
    model.backbone.forward(images.tensors) DONE
proposals, proposal_losses = self.rpn(images, features, targets) BEFORE


RetinaNetModule.forward(self, images, features, targets=None) { // BEGIN
// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/retinanet/retinanet.py
    Params:
        type(images.image_size): <class 'list'>
        type(images.tensors): <class 'torch.Tensor'>
        len(features)): 5
            feature[0].shape: torch.Size([1, 1024, 60, 72])
            feature[1].shape: torch.Size([1, 1024, 30, 36])
            feature[2].shape: torch.Size([1, 1024, 15, 18])
            feature[3].shape: torch.Size([1, 1024, 8, 9])
            feature[4].shape: torch.Size([1, 1024, 4, 5])


    RetinaNetHead.forward(self, x) { // BEGIN
    // // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/retinanet/retinanet.py
        Param:
            len(x)): 5
                x[0].shape: torch.Size([1, 1024, 60, 72])
                x[1].shape: torch.Size([1, 1024, 30, 36])
                x[2].shape: torch.Size([1, 1024, 15, 18])
                x[3].shape: torch.Size([1, 1024, 8, 9])
                x[4].shape: torch.Size([1, 1024, 4, 5])
        logits = []
        bbox_reg = []

        self.cls_tower:
Sequential(
  (0): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU()
  (2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU()
  (4): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (5): ReLU()
  (6): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (7): ReLU()
)

        self.cls_logits:
Conv2d(1024, 9, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.bbox_tower:
Sequential(
  (0): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU()
  (2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU()
  (4): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (5): ReLU()
  (6): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (7): ReLU()
)

        self.bbox_pred:
Conv2d(1024, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))



        for idx, feature in enumerate(x) {
            ===== iteration: 0 ====
            feature[0].shape: torch.Size([1, 1024, 60, 72])

            logits.append(self.cls_logits(self.cls_tower(feature)))
                len(logits): 1

            bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
                len(bbox_reg): 1

            ===== iteration: 1 ====
            feature[1].shape: torch.Size([1, 1024, 30, 36])

            logits.append(self.cls_logits(self.cls_tower(feature)))
                len(logits): 2

            bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
                len(bbox_reg): 2

            ===== iteration: 2 ====
            feature[2].shape: torch.Size([1, 1024, 15, 18])

            logits.append(self.cls_logits(self.cls_tower(feature)))
                len(logits): 3

            bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
                len(bbox_reg): 3

            ===== iteration: 3 ====
            feature[3].shape: torch.Size([1, 1024, 8, 9])

            logits.append(self.cls_logits(self.cls_tower(feature)))
                len(logits): 4

            bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
                len(bbox_reg): 4

            ===== iteration: 4 ====
            feature[4].shape: torch.Size([1, 1024, 4, 5])

            logits.append(self.cls_logits(self.cls_tower(feature)))
                len(logits): 5

            bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
                len(bbox_reg): 5



        }// END for idx, feature n enumerate(x)
 ==== logits ====
logits[0].shape: torch.Size([1, 9, 60, 72])
logits[1].shape: torch.Size([1, 9, 30, 36])
logits[2].shape: torch.Size([1, 9, 15, 18])
logits[3].shape: torch.Size([1, 9, 8, 9])
logits[4].shape: torch.Size([1, 9, 4, 5])

 ==== bbox_reg ====
bbox_reg[0].shape: torch.Size([1, 36, 60, 72])
bbox_reg[1].shape: torch.Size([1, 36, 30, 36])
bbox_reg[2].shape: torch.Size([1, 36, 15, 18])
bbox_reg[3].shape: torch.Size([1, 36, 8, 9])
bbox_reg[4].shape: torch.Size([1, 36, 4, 5])

return logits, bbox_reg
    } // END RetinaNetHead.forward(self, x)
self.head: RetinaNetHead(
  (cls_tower): Sequential(
    (0): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): ReLU()
    (6): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU()
  )
  (bbox_tower): Sequential(
    (0): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): ReLU()
    (6): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU()
  )
  (cls_logits): Conv2d(1024, 9, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bbox_pred): Conv2d(1024, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
)
box_cls, box_regression = self.head(features)
    len(box_cls): 5
    box_cls[0].shape: torch.Size([1, 9, 60, 72])
    box_cls[1].shape: torch.Size([1, 9, 30, 36])
    box_cls[2].shape: torch.Size([1, 9, 15, 18])
    box_cls[3].shape: torch.Size([1, 9, 8, 9])
    box_cls[4].shape: torch.Size([1, 9, 4, 5])
    len(box_regression): 5
    box_regression[0].shape: torch.Size([1, 36, 60, 72])
    box_regression[1].shape: torch.Size([1, 36, 30, 36])
    box_regression[2].shape: torch.Size([1, 36, 15, 18])
    box_regression[3].shape: torch.Size([1, 36, 8, 9])
    box_regression[4].shape: torch.Size([1, 36, 4, 5])

    AnchorGenerator.forward(image_list, feature_maps) { //BEGIN
    // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
        Params:
            image_list:
                len(image_list.image_sizes): 1
                image_list.image_sizes[0]: torch.Size([480, 561])
                len(image_list.tensors): 1
                image_list.tensors[0].shape: torch.Size([3, 480, 576])
            feature_maps:
                feature_maps[0].shape: torch.Size([1, 1024, 60, 72])
                feature_maps[1].shape: torch.Size([1, 1024, 30, 36])
                feature_maps[2].shape: torch.Size([1, 1024, 15, 18])
                feature_maps[3].shape: torch.Size([1, 1024, 8, 9])
                feature_maps[4].shape: torch.Size([1, 1024, 4, 5])

grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        AnchorGenerator.grid_anchors(grid_sizes) { // BEGIN
            Param:
            grid_sizes: [torch.Size([60, 72]), torch.Size([30, 36]), torch.Size([15, 18]), torch.Size([8, 9]), torch.Size([4, 5])]
return anchors
        } // END AnchorGenerator.grid_anchors(grid_sizes)

anchors = []
for i, (image_height, image_width) in enumerate(image_list.image_sizes) {

        anchors_in_image = []

        for anchors_per_feature_map in anchors_over_all_feature_maps {

        ========================
        anchors_per_feature_map.shape: torch.Size([38880, 4])
        ========================
        boxlist = BoxList( anchors_per_feature_map, (image_width, image_height), mode="xyxy" )
        boxlist:
            BoxList(num_boxes=38880, image_width=561, image_height=480, mode=xyxy)

        self.add_visibility_to(boxlist)

AnchorGenerator.add_visibitity_to(boxlist) { // BEGIN
// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
        } // END AnchorGenerator.add_visibitity_to(boxlist)

        boxlist:
            BoxList(num_boxes=38880, image_width=561, image_height=480, mode=xyxy)

        anchors_in_image.append(boxlist)

        ========================
        anchors_per_feature_map.shape: torch.Size([9720, 4])
        ========================
        boxlist = BoxList( anchors_per_feature_map, (image_width, image_height), mode="xyxy" )
        boxlist:
            BoxList(num_boxes=9720, image_width=561, image_height=480, mode=xyxy)

        self.add_visibility_to(boxlist)

AnchorGenerator.add_visibitity_to(boxlist) { // BEGIN
// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
        } // END AnchorGenerator.add_visibitity_to(boxlist)

        boxlist:
            BoxList(num_boxes=9720, image_width=561, image_height=480, mode=xyxy)

        anchors_in_image.append(boxlist)

        ========================
        anchors_per_feature_map.shape: torch.Size([2430, 4])
        ========================
        boxlist = BoxList( anchors_per_feature_map, (image_width, image_height), mode="xyxy" )
        boxlist:
            BoxList(num_boxes=2430, image_width=561, image_height=480, mode=xyxy)

        self.add_visibility_to(boxlist)

AnchorGenerator.add_visibitity_to(boxlist) { // BEGIN
// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
        } // END AnchorGenerator.add_visibitity_to(boxlist)

        boxlist:
            BoxList(num_boxes=2430, image_width=561, image_height=480, mode=xyxy)

        anchors_in_image.append(boxlist)

        ========================
        anchors_per_feature_map.shape: torch.Size([648, 4])
        ========================
        boxlist = BoxList( anchors_per_feature_map, (image_width, image_height), mode="xyxy" )
        boxlist:
            BoxList(num_boxes=648, image_width=561, image_height=480, mode=xyxy)

        self.add_visibility_to(boxlist)

AnchorGenerator.add_visibitity_to(boxlist) { // BEGIN
// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
        } // END AnchorGenerator.add_visibitity_to(boxlist)

        boxlist:
            BoxList(num_boxes=648, image_width=561, image_height=480, mode=xyxy)

        anchors_in_image.append(boxlist)

        ========================
        anchors_per_feature_map.shape: torch.Size([180, 4])
        ========================
        boxlist = BoxList( anchors_per_feature_map, (image_width, image_height), mode="xyxy" )
        boxlist:
            BoxList(num_boxes=180, image_width=561, image_height=480, mode=xyxy)

        self.add_visibility_to(boxlist)

AnchorGenerator.add_visibitity_to(boxlist) { // BEGIN
// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
        } // END AnchorGenerator.add_visibitity_to(boxlist)

        boxlist:
            BoxList(num_boxes=180, image_width=561, image_height=480, mode=xyxy)

        anchors_in_image.append(boxlist)

        } // END for anchors_per_feature_map in anchors_over_all_feature_maps

        anchors_in_image:
            [BoxList(num_boxes=38880, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=9720, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=2430, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=648, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=180, image_width=561, image_height=480, mode=xyxy)]

        anchors.append(anchors_in_image)

} // END for i, (image_height, image_width) in enumerate(image_list.image_sizes)

        anchors:
            [[BoxList(num_boxes=38880, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=9720, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=2430, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=648, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=180, image_width=561, image_height=480, mode=xyxy)]]

return anchors
    } // END AnchorGenerator.forward(image_list, feature_maps)
anchors = self.anchor_generator(images, features)
self.anchor_generator: AnchorGenerator(
  (cell_anchors): BufferList()
)
anchors: [[BoxList(num_boxes=38880, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=9720, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=2430, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=648, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=180, image_width=561, image_height=480, mode=xyxy)]]
if self.training: False
    return self._forward_test(anchors, box_cls, box_regression)


RetinaNetModule._forward_test(self, anchors, box_cls, box_regression) { // BEGIN
// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/retinanet/retinanet.py
    params:
    len(anchors): 1
    len(box_cls): 5
    len(box_regression): 5
    self.box_selector_test: RetinaNetPostProcessor()
    boxes = self.box_selector_test(anchors, box_cls, box_regression)
        RPNPostProcessing.forward(self. anchors, objectness, box_regression, targets=None) { // BEGIN
        Params:
        tanchors: len(anchors) : 1
        tobjectness: len(objectness) : 5
        tbox_regression: type(box_regression) : <class 'list'>
        ttarget: None
return boxlists
        } // END RPNPostProcessing.forward(self. anchors, objectness, box_regression, targets=None)
len(boxes): 1
(boxes): [BoxList(num_boxes=72, image_width=561, image_height=480, mode=xyxy)]
return boxes, {} # {} is just empty dictionayr


} // RetinaNetModule._forward_test(self, anchors, box_cls, box_regression): END
} // END RetinaNetModule.forward(self, images, features, targets=None)
proposals, proposal_losses = self.rpn(images, features, targets) DONE
x = features
result = proposals
return result
} // END GeneralizedRCNN.forward(self, images, targets=None)
return pred
    pred: BoxList(num_boxes=72, image_width=561, image_height=480, mode=xyxy)
} // END compute_prediction(self, image)




