DEBUG:root:DetectionDemo.__init__ { // BEGIN
DEBUG:root:	self.model = build_detection_model(self.cfg)
DEBUG:root:GeneralizedRCNN.__init__(self, cfg) { //BEGIN
DEBUG:root:	// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/detector/generalized_rcnn.py

DEBUG:root:		Params:
DEBUG:root:			cfg:
DEBUG:root:	super(GeneralizedRCNN, self).__init__()

DEBUG:root:	self.backbone = build_backbone(cfg) // CALL
DEBUG:root:
	build_backbone(cfg) { // BEGIN
DEBUG:root:		defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/backbone/backbone.py

DEBUG:root:		Params:
DEBUG:root:			cfg:
DEBUG:root:
	build_resnet_fpn_p3p7_backbone(cfg) { // BEGIN
DEBUG:root:	// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/backbone/backbone.py

DEBUG:root:		Params:
DEBUG:root:			cfg:
DEBUG:root:		body = resnet.ResNet(cfg) // CALL
DEBUG:root:
	Resnet.__init__ { //BEGIN
DEBUG:root:	// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/backbone/resnet.py

DEBUG:root:		Params:
DEBUG:root:			cfg:
DEBUG:root:		_STEM_MODULES: {'StemWithFixedBatchNorm': <class 'maskrcnn_benchmark.modeling.backbone.resnet.StemWithFixedBatchNorm'>}
DEBUG:root:		_cfg.MODEL.RESNETS.STEM_FUNC: StemWithFixedBatchNorm
DEBUG:root:		stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
DEBUG:root:			stem_module: <class 'maskrcnn_benchmark.modeling.backbone.resnet.StemWithFixedBatchNorm'>
DEBUG:root:		cfg.MODEL.BACKBONE.CONV_BODY: R-50-FPN-RETINANET
DEBUG:root:		stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY=R-50-FPN-RETINANET]
DEBUG:root:			stage_specs: (StageSpec(index=1, block_count=3, return_features=True), StageSpec(index=2, block_count=4, return_features=True), StageSpec(index=3, block_count=6, return_features=True), StageSpec(index=4, block_count=3, return_features=True))
DEBUG:root:		_TRANSFORMATION_MODULES: {'BottleneckWithFixedBatchNorm': <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>}
DEBUG:root:		cfg.MODEL.RESNETS.TRANS_FUNC: BottleneckWithFixedBatchNorm
DEBUG:root:		transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC=BottleneckWithFixedBatchNorm]
DEBUG:root:			transformation_module: <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>
DEBUG:root:		self.stem = stem_module(cfg)
DEBUG:root:			self.stem: StemWithFixedBatchNorm(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): FrozenBatchNorm2d()
)
DEBUG:root:		num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
DEBUG:root:		num_groups.stem: 1
DEBUG:root:		width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
DEBUG:root:		width_per_group: 64
DEBUG:root:		in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
DEBUG:root:		in_channels: 64
DEBUG:root:		stage2_bottleneck_channels = num_groups * width_per_group
DEBUG:root:		stage2_bottleneck_channels: 64
DEBUG:root:		stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
DEBUG:root:		stage2_out_channels: 256
DEBUG:root:		self.stages = []
DEBUG:root:		self.return_features = {}
DEBUG:root:		for stage_spec in stage_specs {
DEBUG:root:

			--------------------------------------------------------
DEBUG:root:

			stage_spec: StageSpec(index=1, block_count=3, return_features=True)
DEBUG:root:			stage_spec.index: 1
DEBUG:root:

			--------------------------------------------------------
DEBUG:root:			name = "layer" + str(stage_spec.index)
DEBUG:root:			name: layer1
DEBUG:root:			stage2_relative_factor = 2 ** (stage_spec.index - 1)
DEBUG:root:			stage2_relative_factor: 1
DEBUG:root:			bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
DEBUG:root:			bottlenec_channels: 64
DEBUG:root:			out_channels = stage2_out_channels * stage2_relative_factor
DEBUG:root:			out_channels: 256
DEBUG:root:			stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]
DEBUG:root:			stage_with_dcn: False
DEBUG:root:			module = _make_stage(
DEBUG:root:				transformation_module = <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>,
DEBUG:root:				in_channels = 64,
DEBUG:root:				bottleneck_channels = 64,
DEBUG:root:				out_channels = 256,
DEBUG:root:				stage_spec.block_count = 3,
DEBUG:root:				num_groups = 1,
DEBUG:root:				cfg.MODEL.RESNETS.STRIDE_IN_1X1 : True,
DEBUG:root:				first_stride=int(stage_spec.index > 1) + 1: 1,
DEBUG:root:				dcn_config={
DEBUG:root:					'stage_with_dcn': False,
DEBUG:root:					'with_modulated_dcn': False,
DEBUG:root:					'deformable_groups': 1,
DEBUG:root:					}
DEBUG:root:				)
DEBUG:root:			in_channels = out_channels
DEBUG:root:			in_channels: 256
DEBUG:root:			self.add_module(name=layer1, module)
DEBUG:root:			self.stages.append(name=layer1)
DEBUG:root:			stage_spec.return_features: True
DEBUG:root:			self.return_features[name] = stage_spec.return_features
DEBUG:root:

			--------------------------------------------------------
DEBUG:root:

			stage_spec: StageSpec(index=2, block_count=4, return_features=True)
DEBUG:root:			stage_spec.index: 2
DEBUG:root:

			--------------------------------------------------------
DEBUG:root:			name = "layer" + str(stage_spec.index)
DEBUG:root:			name: layer2
DEBUG:root:			stage2_relative_factor = 2 ** (stage_spec.index - 1)
DEBUG:root:			stage2_relative_factor: 2
DEBUG:root:			bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
DEBUG:root:			bottlenec_channels: 128
DEBUG:root:			out_channels = stage2_out_channels * stage2_relative_factor
DEBUG:root:			out_channels: 512
DEBUG:root:			stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]
DEBUG:root:			stage_with_dcn: False
DEBUG:root:			module = _make_stage(
DEBUG:root:				transformation_module = <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>,
DEBUG:root:				in_channels = 256,
DEBUG:root:				bottleneck_channels = 128,
DEBUG:root:				out_channels = 512,
DEBUG:root:				stage_spec.block_count = 4,
DEBUG:root:				num_groups = 1,
DEBUG:root:				cfg.MODEL.RESNETS.STRIDE_IN_1X1 : True,
DEBUG:root:				first_stride=int(stage_spec.index > 1) + 1: 2,
DEBUG:root:				dcn_config={
DEBUG:root:					'stage_with_dcn': False,
DEBUG:root:					'with_modulated_dcn': False,
DEBUG:root:					'deformable_groups': 1,
DEBUG:root:					}
DEBUG:root:				)
DEBUG:root:			in_channels = out_channels
DEBUG:root:			in_channels: 512
DEBUG:root:			self.add_module(name=layer2, module)
DEBUG:root:			self.stages.append(name=layer2)
DEBUG:root:			stage_spec.return_features: True
DEBUG:root:			self.return_features[name] = stage_spec.return_features
DEBUG:root:

			--------------------------------------------------------
DEBUG:root:

			stage_spec: StageSpec(index=3, block_count=6, return_features=True)
DEBUG:root:			stage_spec.index: 3
DEBUG:root:

			--------------------------------------------------------
DEBUG:root:			name = "layer" + str(stage_spec.index)
DEBUG:root:			name: layer3
DEBUG:root:			stage2_relative_factor = 2 ** (stage_spec.index - 1)
DEBUG:root:			stage2_relative_factor: 4
DEBUG:root:			bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
DEBUG:root:			bottlenec_channels: 256
DEBUG:root:			out_channels = stage2_out_channels * stage2_relative_factor
DEBUG:root:			out_channels: 1024
DEBUG:root:			stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]
DEBUG:root:			stage_with_dcn: False
DEBUG:root:			module = _make_stage(
DEBUG:root:				transformation_module = <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>,
DEBUG:root:				in_channels = 512,
DEBUG:root:				bottleneck_channels = 256,
DEBUG:root:				out_channels = 1024,
DEBUG:root:				stage_spec.block_count = 6,
DEBUG:root:				num_groups = 1,
DEBUG:root:				cfg.MODEL.RESNETS.STRIDE_IN_1X1 : True,
DEBUG:root:				first_stride=int(stage_spec.index > 1) + 1: 2,
DEBUG:root:				dcn_config={
DEBUG:root:					'stage_with_dcn': False,
DEBUG:root:					'with_modulated_dcn': False,
DEBUG:root:					'deformable_groups': 1,
DEBUG:root:					}
DEBUG:root:				)
DEBUG:root:			in_channels = out_channels
DEBUG:root:			in_channels: 1024
DEBUG:root:			self.add_module(name=layer3, module)
DEBUG:root:			self.stages.append(name=layer3)
DEBUG:root:			stage_spec.return_features: True
DEBUG:root:			self.return_features[name] = stage_spec.return_features
DEBUG:root:

			--------------------------------------------------------
DEBUG:root:

			stage_spec: StageSpec(index=4, block_count=3, return_features=True)
DEBUG:root:			stage_spec.index: 4
DEBUG:root:

			--------------------------------------------------------
DEBUG:root:			name = "layer" + str(stage_spec.index)
DEBUG:root:			name: layer4
DEBUG:root:			stage2_relative_factor = 2 ** (stage_spec.index - 1)
DEBUG:root:			stage2_relative_factor: 8
DEBUG:root:			bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
DEBUG:root:			bottlenec_channels: 512
DEBUG:root:			out_channels = stage2_out_channels * stage2_relative_factor
DEBUG:root:			out_channels: 2048
DEBUG:root:			stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]
DEBUG:root:			stage_with_dcn: False
DEBUG:root:			module = _make_stage(
DEBUG:root:				transformation_module = <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>,
DEBUG:root:				in_channels = 1024,
DEBUG:root:				bottleneck_channels = 512,
DEBUG:root:				out_channels = 2048,
DEBUG:root:				stage_spec.block_count = 3,
DEBUG:root:				num_groups = 1,
DEBUG:root:				cfg.MODEL.RESNETS.STRIDE_IN_1X1 : True,
DEBUG:root:				first_stride=int(stage_spec.index > 1) + 1: 2,
DEBUG:root:				dcn_config={
DEBUG:root:					'stage_with_dcn': False,
DEBUG:root:					'with_modulated_dcn': False,
DEBUG:root:					'deformable_groups': 1,
DEBUG:root:					}
DEBUG:root:				)
DEBUG:root:			in_channels = out_channels
DEBUG:root:			in_channels: 2048
DEBUG:root:			self.add_module(name=layer4, module)
DEBUG:root:			self.stages.append(name=layer4)
DEBUG:root:			stage_spec.return_features: True
DEBUG:root:			self.return_features[name] = stage_spec.return_features
DEBUG:root:} // END for stage_spec in stage_specs:
DEBUG:root:			cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT: 2)
DEBUG:root:			self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)
DEBUG:root:
	Resnet.__freeze_backbone(self, freeze_at) { // BEGIN
DEBUG:root:
		Params:
DEBUG:root:
			freeze_at: 2
DEBUG:root:	} // END Resnet.__freeze_backbone(self, freeze_at)

DEBUG:root:} // END Resnet.__init__ END

DEBUG:root:		body = resnet.ResNet(cfg) // RETURNED
DEBUG:root:		cfg.MODEL.RESNETS.RES2_OUT_CHANNELS: 256
DEBUG:root:		in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
DEBUG:root:		in_channels_stage2 = 256
DEBUG:root:		cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS:1024
DEBUG:root:		out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
DEBUG:root:		out_channels = 1024
DEBUG:root:		in_channels_stage2: 256
DEBUG:root:		out_channels: 1024
DEBUG:root:		cfg.MODEL.RETINANET.USE_C5: True
DEBUG:root:		in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 else out_channels
DEBUG:root:		in_channels_p6p7 = 2048
DEBUG:root:
		fpn = fpn_module.FPN(
DEBUG:root:
				in_channels_list = [0, 512, 1024, 2048],
DEBUG:root:
				out_channels = 1024, 
DEBUG:root:
				conv_block=conv_with_kaiming_uniform( cfg.MODEL.FPN.USE_GN =False, cfg.MODEL.FPN.USE_RELU =False ),
DEBUG:root:
				top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7=2048, out_channels=1024,) // CALL
DEBUG:root:
			conv_with_kaiming_uniform(use_gn=False, use_relut=False) { //BEGIN
DEBUG:root:			// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
DEBUG:root:			} // END conv_with_kaiming_uniform(use_gn=False, use_relu=False)

DEBUG:root:

			LastLevelP6P7.__init__(self, in_channels, out_channels) { //BEGIN
DEBUG:root:				Param:
DEBUG:root:					in_channels: 2048
DEBUG:root:					out_channels: 1024
DEBUG:root:				super(LastLevelP6P7, self).__init__()
DEBUG:root:				self.p6 = nn.Conv2d(in_channels=2048, out_channels=1024, 3, 2, 1)
DEBUG:root:				self.p7 = nn.Conv2d(out_channels=1024, out_channels=1024, 3, 2, 1)
DEBUG:root:				for module in [self.p6, self.p7] {
DEBUG:root:					module=Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
DEBUG:root:					nn.init.kaiming_uniform_(module.weight=module.weight, a=1)
DEBUG:root:					nn.init.constant_(module.bias=module.bias, 0)
DEBUG:root:					module=Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
DEBUG:root:					nn.init.kaiming_uniform_(module.weight=module.weight, a=1)
DEBUG:root:					nn.init.constant_(module.bias=module.bias, 0)
DEBUG:root:				} // END for module in [self.p6, self.p7]
DEBUG:root:				self.use_p5 : False
DEBUG:root:	
		} // END LastLevelP6P7.__init__(self, in_channels, out_channels)


DEBUG:root:

FPN.__init__ { // BEGIN
DEBUG:root:		defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/backbone/fpn.py
DEBUG:root:		Params
DEBUG:root:			in_channels_list: [0, 512, 1024, 2048]
DEBUG:root:			out_channels: 1024
DEBUG:root:			conv_block: <function conv_with_kaiming_uniform.<locals>.make_conv at 0x7f492f576268>
DEBUG:root:			top_blocks: LastLevelP6P7(
  (p6): Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (p7): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
)
DEBUG:root:	super(FPN, self).__init__()

DEBUG:root:	for idx, in_channels in enumerate(in_channels_list, 1) {
DEBUG:root:
		-----------------------------------------------------
DEBUG:root:
		iteration with idx:1, in_channels:0
DEBUG:root:
		-----------------------------------------------------
DEBUG:root:		if in_channels ==0, skip

DEBUG:root:
		-----------------------------------------------------
DEBUG:root:
		iteration with idx:2, in_channels:512
DEBUG:root:
		-----------------------------------------------------
DEBUG:root:		inner_block: fpn_inner2
DEBUG:root:		layer_block: fpn_layer2
DEBUG:root:		inner_block_module = conv_block(in_channels=512, out_channels=1024, 1)
DEBUG:root:
				make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
DEBUG:root:				// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
DEBUG:root:				Param
DEBUG:root:					in_channels: 512
DEBUG:root:					out_channels: 1024
DEBUG:root:					kernel_size: 1
DEBUG:root:					stride: 1
DEBUG:root:					dilation: 1
DEBUG:root:				conv = Conv2d(in_channles=512, out_channels=1024, kernel_size=1, stride=1
DEBUG:root:				       padding=0, dilation=1, bias=True, )
DEBUG:root:					conv: Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
DEBUG:root:				nn.init.kaiming_uniform_(conv.weight, a=1)
DEBUG:root:				if not use_gn:
DEBUG:root:					nn.init.constant_(conv.bias, 0)
DEBUG:root:				module = [conv,]
DEBUG:root:					module: [Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))]
DEBUG:root:				conv: {conv}
DEBUG:root:				return conv

DEBUG:root:				} // END conv_with_kaiming_uniform().make_conv()

DEBUG:root:		layer_block_module = conv_block(out_channels=1024, out_channels=1024, 3,1)
DEBUG:root:
				make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
DEBUG:root:				// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
DEBUG:root:				Param
DEBUG:root:					in_channels: 1024
DEBUG:root:					out_channels: 1024
DEBUG:root:					kernel_size: 3
DEBUG:root:					stride: 1
DEBUG:root:					dilation: 1
DEBUG:root:				conv = Conv2d(in_channles=1024, out_channels=1024, kernel_size=3, stride=1
DEBUG:root:				       padding=1, dilation=1, bias=True, )
DEBUG:root:					conv: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
DEBUG:root:				nn.init.kaiming_uniform_(conv.weight, a=1)
DEBUG:root:				if not use_gn:
DEBUG:root:					nn.init.constant_(conv.bias, 0)
DEBUG:root:				module = [conv,]
DEBUG:root:					module: [Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
DEBUG:root:				conv: {conv}
DEBUG:root:				return conv

DEBUG:root:				} // END conv_with_kaiming_uniform().make_conv()

DEBUG:root:		self.add_module(fpn_inner2, inner_block_module)
DEBUG:root:		self.add_module(fpn_layer2, layer_block_module)
DEBUG:root:		self.inner_blocks.append(fpn_inner2)
DEBUG:root:		self.layer_blocks.append(fpn_layer2)
DEBUG:root:
		-----------------------------------------------------
DEBUG:root:
		iteration with idx:3, in_channels:1024
DEBUG:root:
		-----------------------------------------------------
DEBUG:root:		inner_block: fpn_inner3
DEBUG:root:		layer_block: fpn_layer3
DEBUG:root:		inner_block_module = conv_block(in_channels=1024, out_channels=1024, 1)
DEBUG:root:
				make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
DEBUG:root:				// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
DEBUG:root:				Param
DEBUG:root:					in_channels: 1024
DEBUG:root:					out_channels: 1024
DEBUG:root:					kernel_size: 1
DEBUG:root:					stride: 1
DEBUG:root:					dilation: 1
DEBUG:root:				conv = Conv2d(in_channles=1024, out_channels=1024, kernel_size=1, stride=1
DEBUG:root:				       padding=0, dilation=1, bias=True, )
DEBUG:root:					conv: Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
DEBUG:root:				nn.init.kaiming_uniform_(conv.weight, a=1)
DEBUG:root:				if not use_gn:
DEBUG:root:					nn.init.constant_(conv.bias, 0)
DEBUG:root:				module = [conv,]
DEBUG:root:					module: [Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))]
DEBUG:root:				conv: {conv}
DEBUG:root:				return conv

DEBUG:root:				} // END conv_with_kaiming_uniform().make_conv()

DEBUG:root:		layer_block_module = conv_block(out_channels=1024, out_channels=1024, 3,1)
DEBUG:root:
				make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
DEBUG:root:				// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
DEBUG:root:				Param
DEBUG:root:					in_channels: 1024
DEBUG:root:					out_channels: 1024
DEBUG:root:					kernel_size: 3
DEBUG:root:					stride: 1
DEBUG:root:					dilation: 1
DEBUG:root:				conv = Conv2d(in_channles=1024, out_channels=1024, kernel_size=3, stride=1
DEBUG:root:				       padding=1, dilation=1, bias=True, )
DEBUG:root:					conv: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
DEBUG:root:				nn.init.kaiming_uniform_(conv.weight, a=1)
DEBUG:root:				if not use_gn:
DEBUG:root:					nn.init.constant_(conv.bias, 0)
DEBUG:root:				module = [conv,]
DEBUG:root:					module: [Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
DEBUG:root:				conv: {conv}
DEBUG:root:				return conv

DEBUG:root:				} // END conv_with_kaiming_uniform().make_conv()

DEBUG:root:		self.add_module(fpn_inner3, inner_block_module)
DEBUG:root:		self.add_module(fpn_layer3, layer_block_module)
DEBUG:root:		self.inner_blocks.append(fpn_inner3)
DEBUG:root:		self.layer_blocks.append(fpn_layer3)
DEBUG:root:
		-----------------------------------------------------
DEBUG:root:
		iteration with idx:4, in_channels:2048
DEBUG:root:
		-----------------------------------------------------
DEBUG:root:		inner_block: fpn_inner4
DEBUG:root:		layer_block: fpn_layer4
DEBUG:root:		inner_block_module = conv_block(in_channels=2048, out_channels=1024, 1)
DEBUG:root:
				make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
DEBUG:root:				// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
DEBUG:root:				Param
DEBUG:root:					in_channels: 2048
DEBUG:root:					out_channels: 1024
DEBUG:root:					kernel_size: 1
DEBUG:root:					stride: 1
DEBUG:root:					dilation: 1
DEBUG:root:				conv = Conv2d(in_channles=2048, out_channels=1024, kernel_size=1, stride=1
DEBUG:root:				       padding=0, dilation=1, bias=True, )
DEBUG:root:					conv: Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))
DEBUG:root:				nn.init.kaiming_uniform_(conv.weight, a=1)
DEBUG:root:				if not use_gn:
DEBUG:root:					nn.init.constant_(conv.bias, 0)
DEBUG:root:				module = [conv,]
DEBUG:root:					module: [Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))]
DEBUG:root:				conv: {conv}
DEBUG:root:				return conv

DEBUG:root:				} // END conv_with_kaiming_uniform().make_conv()

DEBUG:root:		layer_block_module = conv_block(out_channels=1024, out_channels=1024, 3,1)
DEBUG:root:
				make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
DEBUG:root:				// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
DEBUG:root:				Param
DEBUG:root:					in_channels: 1024
DEBUG:root:					out_channels: 1024
DEBUG:root:					kernel_size: 3
DEBUG:root:					stride: 1
DEBUG:root:					dilation: 1
DEBUG:root:				conv = Conv2d(in_channles=1024, out_channels=1024, kernel_size=3, stride=1
DEBUG:root:				       padding=1, dilation=1, bias=True, )
DEBUG:root:					conv: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
DEBUG:root:				nn.init.kaiming_uniform_(conv.weight, a=1)
DEBUG:root:				if not use_gn:
DEBUG:root:					nn.init.constant_(conv.bias, 0)
DEBUG:root:				module = [conv,]
DEBUG:root:					module: [Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
DEBUG:root:				conv: {conv}
DEBUG:root:				return conv

DEBUG:root:				} // END conv_with_kaiming_uniform().make_conv()

DEBUG:root:		self.add_module(fpn_inner4, inner_block_module)
DEBUG:root:		self.add_module(fpn_layer4, layer_block_module)
DEBUG:root:		self.inner_blocks.append(fpn_inner4)
DEBUG:root:		self.layer_blocks.append(fpn_layer4)
DEBUG:root:	} // END for idx, in_channels in enumerate(in_channels_list, 1)
DEBUG:root:	self.top_blocks = top_blocks
DEBUG:root:
	self.inner_blocks: ['fpn_inner2', 'fpn_inner3', 'fpn_inner4']
DEBUG:root:		self.fpn_inner2: Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
DEBUG:root:		self.fpn_inner3: Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
DEBUG:root:		self.fpn_inner4: Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))
DEBUG:root:
	self.layer_blocks: ['fpn_layer2', 'fpn_layer3', 'fpn_layer4']
DEBUG:root:		self.fpn_layer2: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
DEBUG:root:		self.fpn_layer3: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
DEBUG:root:		self.fpn_layer4: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
DEBUG:root:
	self.top_blocks: LastLevelP6P7(
  (p6): Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (p7): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
)
DEBUG:root:} // END FPN.__init__


DEBUG:root:
		fpn = fpn_module.FPN(
DEBUG:root:
				in_channels_list = [0, 512, 1024, 2048],
DEBUG:root:
				out_channels = 1024, 
DEBUG:root:
				conv_block=conv_with_kaiming_uniform( cfg.MODEL.FPN.USE_GN =False, cfg.MODEL.FPN.USE_RELU =False ),
DEBUG:root:
				top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7=2048, out_channels=1024,) // RETURNED
DEBUG:root:			fpn: {fpn}
DEBUG:root:		model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)])) // CALL
DEBUG:root:		model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)])) // RETURNED
DEBUG:root:		model: Sequential(
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
DEBUG:root:		model.out_channels = out_channels
DEBUG:root:			model.out_channels: 1024
DEBUG:root:		return model
DEBUG:root:	} // END build_resnet_fpn_p3p7_backbone(cfg) 


DEBUG:root:	registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg): Sequential(
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
DEBUG:root:		return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
DEBUG:root:	} // END build_backbone(cfg)

DEBUG:root:
	build_resnet_fpn_p3p7_backbone(cfg) { // BEGIN
DEBUG:root:	// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/backbone/backbone.py

DEBUG:root:		Params:
DEBUG:root:			cfg:
DEBUG:root:		body = resnet.ResNet(cfg) // CALL
DEBUG:root:
	Resnet.__init__ { //BEGIN
DEBUG:root:	// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/backbone/resnet.py

DEBUG:root:		Params:
DEBUG:root:			cfg:
DEBUG:root:		_STEM_MODULES: {'StemWithFixedBatchNorm': <class 'maskrcnn_benchmark.modeling.backbone.resnet.StemWithFixedBatchNorm'>}
DEBUG:root:		_cfg.MODEL.RESNETS.STEM_FUNC: StemWithFixedBatchNorm
DEBUG:root:		stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
DEBUG:root:			stem_module: <class 'maskrcnn_benchmark.modeling.backbone.resnet.StemWithFixedBatchNorm'>
DEBUG:root:		cfg.MODEL.BACKBONE.CONV_BODY: R-50-FPN-RETINANET
DEBUG:root:		stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY=R-50-FPN-RETINANET]
DEBUG:root:			stage_specs: (StageSpec(index=1, block_count=3, return_features=True), StageSpec(index=2, block_count=4, return_features=True), StageSpec(index=3, block_count=6, return_features=True), StageSpec(index=4, block_count=3, return_features=True))
DEBUG:root:		_TRANSFORMATION_MODULES: {'BottleneckWithFixedBatchNorm': <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>}
DEBUG:root:		cfg.MODEL.RESNETS.TRANS_FUNC: BottleneckWithFixedBatchNorm
DEBUG:root:		transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC=BottleneckWithFixedBatchNorm]
DEBUG:root:			transformation_module: <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>
DEBUG:root:		self.stem = stem_module(cfg)
DEBUG:root:			self.stem: StemWithFixedBatchNorm(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): FrozenBatchNorm2d()
)
DEBUG:root:		num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
DEBUG:root:		num_groups.stem: 1
DEBUG:root:		width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
DEBUG:root:		width_per_group: 64
DEBUG:root:		in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
DEBUG:root:		in_channels: 64
DEBUG:root:		stage2_bottleneck_channels = num_groups * width_per_group
DEBUG:root:		stage2_bottleneck_channels: 64
DEBUG:root:		stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
DEBUG:root:		stage2_out_channels: 256
DEBUG:root:		self.stages = []
DEBUG:root:		self.return_features = {}
DEBUG:root:		for stage_spec in stage_specs {
DEBUG:root:

			--------------------------------------------------------
DEBUG:root:

			stage_spec: StageSpec(index=1, block_count=3, return_features=True)
DEBUG:root:			stage_spec.index: 1
DEBUG:root:

			--------------------------------------------------------
DEBUG:root:			name = "layer" + str(stage_spec.index)
DEBUG:root:			name: layer1
DEBUG:root:			stage2_relative_factor = 2 ** (stage_spec.index - 1)
DEBUG:root:			stage2_relative_factor: 1
DEBUG:root:			bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
DEBUG:root:			bottlenec_channels: 64
DEBUG:root:			out_channels = stage2_out_channels * stage2_relative_factor
DEBUG:root:			out_channels: 256
DEBUG:root:			stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]
DEBUG:root:			stage_with_dcn: False
DEBUG:root:			module = _make_stage(
DEBUG:root:				transformation_module = <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>,
DEBUG:root:				in_channels = 64,
DEBUG:root:				bottleneck_channels = 64,
DEBUG:root:				out_channels = 256,
DEBUG:root:				stage_spec.block_count = 3,
DEBUG:root:				num_groups = 1,
DEBUG:root:				cfg.MODEL.RESNETS.STRIDE_IN_1X1 : True,
DEBUG:root:				first_stride=int(stage_spec.index > 1) + 1: 1,
DEBUG:root:				dcn_config={
DEBUG:root:					'stage_with_dcn': False,
DEBUG:root:					'with_modulated_dcn': False,
DEBUG:root:					'deformable_groups': 1,
DEBUG:root:					}
DEBUG:root:				)
DEBUG:root:			in_channels = out_channels
DEBUG:root:			in_channels: 256
DEBUG:root:			self.add_module(name=layer1, module)
DEBUG:root:			self.stages.append(name=layer1)
DEBUG:root:			stage_spec.return_features: True
DEBUG:root:			self.return_features[name] = stage_spec.return_features
DEBUG:root:

			--------------------------------------------------------
DEBUG:root:

			stage_spec: StageSpec(index=2, block_count=4, return_features=True)
DEBUG:root:			stage_spec.index: 2
DEBUG:root:

			--------------------------------------------------------
DEBUG:root:			name = "layer" + str(stage_spec.index)
DEBUG:root:			name: layer2
DEBUG:root:			stage2_relative_factor = 2 ** (stage_spec.index - 1)
DEBUG:root:			stage2_relative_factor: 2
DEBUG:root:			bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
DEBUG:root:			bottlenec_channels: 128
DEBUG:root:			out_channels = stage2_out_channels * stage2_relative_factor
DEBUG:root:			out_channels: 512
DEBUG:root:			stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]
DEBUG:root:			stage_with_dcn: False
DEBUG:root:			module = _make_stage(
DEBUG:root:				transformation_module = <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>,
DEBUG:root:				in_channels = 256,
DEBUG:root:				bottleneck_channels = 128,
DEBUG:root:				out_channels = 512,
DEBUG:root:				stage_spec.block_count = 4,
DEBUG:root:				num_groups = 1,
DEBUG:root:				cfg.MODEL.RESNETS.STRIDE_IN_1X1 : True,
DEBUG:root:				first_stride=int(stage_spec.index > 1) + 1: 2,
DEBUG:root:				dcn_config={
DEBUG:root:					'stage_with_dcn': False,
DEBUG:root:					'with_modulated_dcn': False,
DEBUG:root:					'deformable_groups': 1,
DEBUG:root:					}
DEBUG:root:				)
DEBUG:root:			in_channels = out_channels
DEBUG:root:			in_channels: 512
DEBUG:root:			self.add_module(name=layer2, module)
DEBUG:root:			self.stages.append(name=layer2)
DEBUG:root:			stage_spec.return_features: True
DEBUG:root:			self.return_features[name] = stage_spec.return_features
DEBUG:root:

			--------------------------------------------------------
DEBUG:root:

			stage_spec: StageSpec(index=3, block_count=6, return_features=True)
DEBUG:root:			stage_spec.index: 3
DEBUG:root:

			--------------------------------------------------------
DEBUG:root:			name = "layer" + str(stage_spec.index)
DEBUG:root:			name: layer3
DEBUG:root:			stage2_relative_factor = 2 ** (stage_spec.index - 1)
DEBUG:root:			stage2_relative_factor: 4
DEBUG:root:			bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
DEBUG:root:			bottlenec_channels: 256
DEBUG:root:			out_channels = stage2_out_channels * stage2_relative_factor
DEBUG:root:			out_channels: 1024
DEBUG:root:			stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]
DEBUG:root:			stage_with_dcn: False
DEBUG:root:			module = _make_stage(
DEBUG:root:				transformation_module = <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>,
DEBUG:root:				in_channels = 512,
DEBUG:root:				bottleneck_channels = 256,
DEBUG:root:				out_channels = 1024,
DEBUG:root:				stage_spec.block_count = 6,
DEBUG:root:				num_groups = 1,
DEBUG:root:				cfg.MODEL.RESNETS.STRIDE_IN_1X1 : True,
DEBUG:root:				first_stride=int(stage_spec.index > 1) + 1: 2,
DEBUG:root:				dcn_config={
DEBUG:root:					'stage_with_dcn': False,
DEBUG:root:					'with_modulated_dcn': False,
DEBUG:root:					'deformable_groups': 1,
DEBUG:root:					}
DEBUG:root:				)
DEBUG:root:			in_channels = out_channels
DEBUG:root:			in_channels: 1024
DEBUG:root:			self.add_module(name=layer3, module)
DEBUG:root:			self.stages.append(name=layer3)
DEBUG:root:			stage_spec.return_features: True
DEBUG:root:			self.return_features[name] = stage_spec.return_features
DEBUG:root:

			--------------------------------------------------------
DEBUG:root:

			stage_spec: StageSpec(index=4, block_count=3, return_features=True)
DEBUG:root:			stage_spec.index: 4
DEBUG:root:

			--------------------------------------------------------
DEBUG:root:			name = "layer" + str(stage_spec.index)
DEBUG:root:			name: layer4
DEBUG:root:			stage2_relative_factor = 2 ** (stage_spec.index - 1)
DEBUG:root:			stage2_relative_factor: 8
DEBUG:root:			bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
DEBUG:root:			bottlenec_channels: 512
DEBUG:root:			out_channels = stage2_out_channels * stage2_relative_factor
DEBUG:root:			out_channels: 2048
DEBUG:root:			stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]
DEBUG:root:			stage_with_dcn: False
DEBUG:root:			module = _make_stage(
DEBUG:root:				transformation_module = <class 'maskrcnn_benchmark.modeling.backbone.resnet.BottleneckWithFixedBatchNorm'>,
DEBUG:root:				in_channels = 1024,
DEBUG:root:				bottleneck_channels = 512,
DEBUG:root:				out_channels = 2048,
DEBUG:root:				stage_spec.block_count = 3,
DEBUG:root:				num_groups = 1,
DEBUG:root:				cfg.MODEL.RESNETS.STRIDE_IN_1X1 : True,
DEBUG:root:				first_stride=int(stage_spec.index > 1) + 1: 2,
DEBUG:root:				dcn_config={
DEBUG:root:					'stage_with_dcn': False,
DEBUG:root:					'with_modulated_dcn': False,
DEBUG:root:					'deformable_groups': 1,
DEBUG:root:					}
DEBUG:root:				)
DEBUG:root:			in_channels = out_channels
DEBUG:root:			in_channels: 2048
DEBUG:root:			self.add_module(name=layer4, module)
DEBUG:root:			self.stages.append(name=layer4)
DEBUG:root:			stage_spec.return_features: True
DEBUG:root:			self.return_features[name] = stage_spec.return_features
DEBUG:root:} // END for stage_spec in stage_specs:
DEBUG:root:			cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT: 2)
DEBUG:root:			self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)
DEBUG:root:
	Resnet.__freeze_backbone(self, freeze_at) { // BEGIN
DEBUG:root:
		Params:
DEBUG:root:
			freeze_at: 2
DEBUG:root:	} // END Resnet.__freeze_backbone(self, freeze_at)

DEBUG:root:} // END Resnet.__init__ END

DEBUG:root:		body = resnet.ResNet(cfg) // RETURNED
DEBUG:root:		cfg.MODEL.RESNETS.RES2_OUT_CHANNELS: 256
DEBUG:root:		in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
DEBUG:root:		in_channels_stage2 = 256
DEBUG:root:		cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS:1024
DEBUG:root:		out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
DEBUG:root:		out_channels = 1024
DEBUG:root:		in_channels_stage2: 256
DEBUG:root:		out_channels: 1024
DEBUG:root:		cfg.MODEL.RETINANET.USE_C5: True
DEBUG:root:		in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 else out_channels
DEBUG:root:		in_channels_p6p7 = 2048
DEBUG:root:
		fpn = fpn_module.FPN(
DEBUG:root:
				in_channels_list = [0, 512, 1024, 2048],
DEBUG:root:
				out_channels = 1024, 
DEBUG:root:
				conv_block=conv_with_kaiming_uniform( cfg.MODEL.FPN.USE_GN =False, cfg.MODEL.FPN.USE_RELU =False ),
DEBUG:root:
				top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7=2048, out_channels=1024,) // CALL
DEBUG:root:
			conv_with_kaiming_uniform(use_gn=False, use_relut=False) { //BEGIN
DEBUG:root:			// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
DEBUG:root:			} // END conv_with_kaiming_uniform(use_gn=False, use_relu=False)

DEBUG:root:

			LastLevelP6P7.__init__(self, in_channels, out_channels) { //BEGIN
DEBUG:root:				Param:
DEBUG:root:					in_channels: 2048
DEBUG:root:					out_channels: 1024
DEBUG:root:				super(LastLevelP6P7, self).__init__()
DEBUG:root:				self.p6 = nn.Conv2d(in_channels=2048, out_channels=1024, 3, 2, 1)
DEBUG:root:				self.p7 = nn.Conv2d(out_channels=1024, out_channels=1024, 3, 2, 1)
DEBUG:root:				for module in [self.p6, self.p7] {
DEBUG:root:					module=Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
DEBUG:root:					nn.init.kaiming_uniform_(module.weight=module.weight, a=1)
DEBUG:root:					nn.init.constant_(module.bias=module.bias, 0)
DEBUG:root:					module=Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
DEBUG:root:					nn.init.kaiming_uniform_(module.weight=module.weight, a=1)
DEBUG:root:					nn.init.constant_(module.bias=module.bias, 0)
DEBUG:root:				} // END for module in [self.p6, self.p7]
DEBUG:root:				self.use_p5 : False
DEBUG:root:	
		} // END LastLevelP6P7.__init__(self, in_channels, out_channels)


DEBUG:root:

FPN.__init__ { // BEGIN
DEBUG:root:		defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/backbone/fpn.py
DEBUG:root:		Params
DEBUG:root:			in_channels_list: [0, 512, 1024, 2048]
DEBUG:root:			out_channels: 1024
DEBUG:root:			conv_block: <function conv_with_kaiming_uniform.<locals>.make_conv at 0x7f492f579d90>
DEBUG:root:			top_blocks: LastLevelP6P7(
  (p6): Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (p7): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
)
DEBUG:root:	super(FPN, self).__init__()

DEBUG:root:	for idx, in_channels in enumerate(in_channels_list, 1) {
DEBUG:root:
		-----------------------------------------------------
DEBUG:root:
		iteration with idx:1, in_channels:0
DEBUG:root:
		-----------------------------------------------------
DEBUG:root:		if in_channels ==0, skip

DEBUG:root:
		-----------------------------------------------------
DEBUG:root:
		iteration with idx:2, in_channels:512
DEBUG:root:
		-----------------------------------------------------
DEBUG:root:		inner_block: fpn_inner2
DEBUG:root:		layer_block: fpn_layer2
DEBUG:root:		inner_block_module = conv_block(in_channels=512, out_channels=1024, 1)
DEBUG:root:
				make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
DEBUG:root:				// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
DEBUG:root:				Param
DEBUG:root:					in_channels: 512
DEBUG:root:					out_channels: 1024
DEBUG:root:					kernel_size: 1
DEBUG:root:					stride: 1
DEBUG:root:					dilation: 1
DEBUG:root:				conv = Conv2d(in_channles=512, out_channels=1024, kernel_size=1, stride=1
DEBUG:root:				       padding=0, dilation=1, bias=True, )
DEBUG:root:					conv: Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
DEBUG:root:				nn.init.kaiming_uniform_(conv.weight, a=1)
DEBUG:root:				if not use_gn:
DEBUG:root:					nn.init.constant_(conv.bias, 0)
DEBUG:root:				module = [conv,]
DEBUG:root:					module: [Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))]
DEBUG:root:				conv: {conv}
DEBUG:root:				return conv

DEBUG:root:				} // END conv_with_kaiming_uniform().make_conv()

DEBUG:root:		layer_block_module = conv_block(out_channels=1024, out_channels=1024, 3,1)
DEBUG:root:
				make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
DEBUG:root:				// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
DEBUG:root:				Param
DEBUG:root:					in_channels: 1024
DEBUG:root:					out_channels: 1024
DEBUG:root:					kernel_size: 3
DEBUG:root:					stride: 1
DEBUG:root:					dilation: 1
DEBUG:root:				conv = Conv2d(in_channles=1024, out_channels=1024, kernel_size=3, stride=1
DEBUG:root:				       padding=1, dilation=1, bias=True, )
DEBUG:root:					conv: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
DEBUG:root:				nn.init.kaiming_uniform_(conv.weight, a=1)
DEBUG:root:				if not use_gn:
DEBUG:root:					nn.init.constant_(conv.bias, 0)
DEBUG:root:				module = [conv,]
DEBUG:root:					module: [Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
DEBUG:root:				conv: {conv}
DEBUG:root:				return conv

DEBUG:root:				} // END conv_with_kaiming_uniform().make_conv()

DEBUG:root:		self.add_module(fpn_inner2, inner_block_module)
DEBUG:root:		self.add_module(fpn_layer2, layer_block_module)
DEBUG:root:		self.inner_blocks.append(fpn_inner2)
DEBUG:root:		self.layer_blocks.append(fpn_layer2)
DEBUG:root:
		-----------------------------------------------------
DEBUG:root:
		iteration with idx:3, in_channels:1024
DEBUG:root:
		-----------------------------------------------------
DEBUG:root:		inner_block: fpn_inner3
DEBUG:root:		layer_block: fpn_layer3
DEBUG:root:		inner_block_module = conv_block(in_channels=1024, out_channels=1024, 1)
DEBUG:root:
				make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
DEBUG:root:				// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
DEBUG:root:				Param
DEBUG:root:					in_channels: 1024
DEBUG:root:					out_channels: 1024
DEBUG:root:					kernel_size: 1
DEBUG:root:					stride: 1
DEBUG:root:					dilation: 1
DEBUG:root:				conv = Conv2d(in_channles=1024, out_channels=1024, kernel_size=1, stride=1
DEBUG:root:				       padding=0, dilation=1, bias=True, )
DEBUG:root:					conv: Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
DEBUG:root:				nn.init.kaiming_uniform_(conv.weight, a=1)
DEBUG:root:				if not use_gn:
DEBUG:root:					nn.init.constant_(conv.bias, 0)
DEBUG:root:				module = [conv,]
DEBUG:root:					module: [Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))]
DEBUG:root:				conv: {conv}
DEBUG:root:				return conv

DEBUG:root:				} // END conv_with_kaiming_uniform().make_conv()

DEBUG:root:		layer_block_module = conv_block(out_channels=1024, out_channels=1024, 3,1)
DEBUG:root:
				make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
DEBUG:root:				// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
DEBUG:root:				Param
DEBUG:root:					in_channels: 1024
DEBUG:root:					out_channels: 1024
DEBUG:root:					kernel_size: 3
DEBUG:root:					stride: 1
DEBUG:root:					dilation: 1
DEBUG:root:				conv = Conv2d(in_channles=1024, out_channels=1024, kernel_size=3, stride=1
DEBUG:root:				       padding=1, dilation=1, bias=True, )
DEBUG:root:					conv: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
DEBUG:root:				nn.init.kaiming_uniform_(conv.weight, a=1)
DEBUG:root:				if not use_gn:
DEBUG:root:					nn.init.constant_(conv.bias, 0)
DEBUG:root:				module = [conv,]
DEBUG:root:					module: [Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
DEBUG:root:				conv: {conv}
DEBUG:root:				return conv

DEBUG:root:				} // END conv_with_kaiming_uniform().make_conv()

DEBUG:root:		self.add_module(fpn_inner3, inner_block_module)
DEBUG:root:		self.add_module(fpn_layer3, layer_block_module)
DEBUG:root:		self.inner_blocks.append(fpn_inner3)
DEBUG:root:		self.layer_blocks.append(fpn_layer3)
DEBUG:root:
		-----------------------------------------------------
DEBUG:root:
		iteration with idx:4, in_channels:2048
DEBUG:root:
		-----------------------------------------------------
DEBUG:root:		inner_block: fpn_inner4
DEBUG:root:		layer_block: fpn_layer4
DEBUG:root:		inner_block_module = conv_block(in_channels=2048, out_channels=1024, 1)
DEBUG:root:
				make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
DEBUG:root:				// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
DEBUG:root:				Param
DEBUG:root:					in_channels: 2048
DEBUG:root:					out_channels: 1024
DEBUG:root:					kernel_size: 1
DEBUG:root:					stride: 1
DEBUG:root:					dilation: 1
DEBUG:root:				conv = Conv2d(in_channles=2048, out_channels=1024, kernel_size=1, stride=1
DEBUG:root:				       padding=0, dilation=1, bias=True, )
DEBUG:root:					conv: Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))
DEBUG:root:				nn.init.kaiming_uniform_(conv.weight, a=1)
DEBUG:root:				if not use_gn:
DEBUG:root:					nn.init.constant_(conv.bias, 0)
DEBUG:root:				module = [conv,]
DEBUG:root:					module: [Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))]
DEBUG:root:				conv: {conv}
DEBUG:root:				return conv

DEBUG:root:				} // END conv_with_kaiming_uniform().make_conv()

DEBUG:root:		layer_block_module = conv_block(out_channels=1024, out_channels=1024, 3,1)
DEBUG:root:
				make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) { //BEGIN
DEBUG:root:				// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/make_layers.py
DEBUG:root:				Param
DEBUG:root:					in_channels: 1024
DEBUG:root:					out_channels: 1024
DEBUG:root:					kernel_size: 3
DEBUG:root:					stride: 1
DEBUG:root:					dilation: 1
DEBUG:root:				conv = Conv2d(in_channles=1024, out_channels=1024, kernel_size=3, stride=1
DEBUG:root:				       padding=1, dilation=1, bias=True, )
DEBUG:root:					conv: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
DEBUG:root:				nn.init.kaiming_uniform_(conv.weight, a=1)
DEBUG:root:				if not use_gn:
DEBUG:root:					nn.init.constant_(conv.bias, 0)
DEBUG:root:				module = [conv,]
DEBUG:root:					module: [Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
DEBUG:root:				conv: {conv}
DEBUG:root:				return conv

DEBUG:root:				} // END conv_with_kaiming_uniform().make_conv()

DEBUG:root:		self.add_module(fpn_inner4, inner_block_module)
DEBUG:root:		self.add_module(fpn_layer4, layer_block_module)
DEBUG:root:		self.inner_blocks.append(fpn_inner4)
DEBUG:root:		self.layer_blocks.append(fpn_layer4)
DEBUG:root:	} // END for idx, in_channels in enumerate(in_channels_list, 1)
DEBUG:root:	self.top_blocks = top_blocks
DEBUG:root:
	self.inner_blocks: ['fpn_inner2', 'fpn_inner3', 'fpn_inner4']
DEBUG:root:		self.fpn_inner2: Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
DEBUG:root:		self.fpn_inner3: Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
DEBUG:root:		self.fpn_inner4: Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))
DEBUG:root:
	self.layer_blocks: ['fpn_layer2', 'fpn_layer3', 'fpn_layer4']
DEBUG:root:		self.fpn_layer2: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
DEBUG:root:		self.fpn_layer3: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
DEBUG:root:		self.fpn_layer4: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
DEBUG:root:
	self.top_blocks: LastLevelP6P7(
  (p6): Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (p7): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
)
DEBUG:root:} // END FPN.__init__


DEBUG:root:
		fpn = fpn_module.FPN(
DEBUG:root:
				in_channels_list = [0, 512, 1024, 2048],
DEBUG:root:
				out_channels = 1024, 
DEBUG:root:
				conv_block=conv_with_kaiming_uniform( cfg.MODEL.FPN.USE_GN =False, cfg.MODEL.FPN.USE_RELU =False ),
DEBUG:root:
				top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7=2048, out_channels=1024,) // RETURNED
DEBUG:root:			fpn: {fpn}
DEBUG:root:		model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)])) // CALL
DEBUG:root:		model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)])) // RETURNED
DEBUG:root:		model: Sequential(
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
DEBUG:root:		model.out_channels = out_channels
DEBUG:root:			model.out_channels: 1024
DEBUG:root:		return model
DEBUG:root:	} // END build_resnet_fpn_p3p7_backbone(cfg) 


DEBUG:root:	self.backbone = build_backbone(cfg) // RETURNED
DEBUG:root:		self.backbone: Sequential(
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
DEBUG:root:	self.backbone.out_channels: 1024
DEBUG:root:	self.rpn = build_rpn(cfg, self.backbone.out_channels) // CALL
DEBUG:root:	build_retinanet(cfg, in_channels) { // BEGIN
DEBUG:root:	// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/retinanet/retinanet.py
DEBUG:root:		Param:
DEBUG:root:			cfg:
DEBUG:root:			in_channels: 1024
DEBUG:root:	return RetinaNetModule(cfg, in_channels)
DEBUG:root:

RetinaNetModule.__init__(self, cfg, in_channels) { // BEGIN
DEBUG:root:// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/retinanet/retinanet.py
DEBUG:root:	Params:
DEBUG:root:		cfg:
DEBUG:root:		in_channels: 1024

DEBUG:root:	super(RetinaNetModule, self).__init__()
DEBUG:root:	self.cfg = cfg.clone()
DEBUG:root:	anchor_generator = make_anchor_generator_retinanet(cfg)

DEBUG:root:
		make_anchor_generator_retinanet(config) { // BEGIN
DEBUG:root:/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:		config params
DEBUG:root:			anchor_sizes: (32, 64, 128, 256, 512)
DEBUG:root:			aspect_ratios: (0.5, 1.0, 2.0)
DEBUG:root:			anchor_strides: (8, 16, 32, 64, 128)
DEBUG:root:			straddle_thresh: -1
DEBUG:root:			octave: 2.0
DEBUG:root:			scales_per_octave: 3
DEBUG:root:		new_anchor_sizes = []
DEBUG:root:
		for size in anchor_sizes {
DEBUG:root:			----------- 
DEBUG:root:			size: 32
DEBUG:root:			----------- 
DEBUG:root:			per_layer_anchor_sizes = []
DEBUG:root:
			for scale_per_octave in range(scales_per_octave) { // BEGIN
DEBUG:root:				------------------------
DEBUG:root:				scale_per_octave : 0
DEBUG:root:				------------------------
DEBUG:root:					octave : 2.0

DEBUG:root:				octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

DEBUG:root:					octave_scale: 1.0
DEBUG:root:					size: 32

DEBUG:root:				per_layer_anchor_sizes.append(octave_scale * size)
DEBUG:root:					per_layer_anchor_sizes: [32.0]

DEBUG:root:				------------------------
DEBUG:root:				scale_per_octave : 1
DEBUG:root:				------------------------
DEBUG:root:					octave : 2.0

DEBUG:root:				octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

DEBUG:root:					octave_scale: 1.2599210498948732
DEBUG:root:					size: 32

DEBUG:root:				per_layer_anchor_sizes.append(octave_scale * size)
DEBUG:root:					per_layer_anchor_sizes: [32.0, 40.31747359663594]

DEBUG:root:				------------------------
DEBUG:root:				scale_per_octave : 2
DEBUG:root:				------------------------
DEBUG:root:					octave : 2.0

DEBUG:root:				octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

DEBUG:root:					octave_scale: 1.5874010519681994
DEBUG:root:					size: 32

DEBUG:root:				per_layer_anchor_sizes.append(octave_scale * size)
DEBUG:root:					per_layer_anchor_sizes: [32.0, 40.31747359663594, 50.79683366298238]

DEBUG:root:			} // EDN for scale_per_octave in range(scales_per_octave)


DEBUG:root:			new_anchor_sizes.append(tuple(per_layer_anchor_sizes))
DEBUG:root:			new_anchor_sizes: [(32.0, 40.31747359663594, 50.79683366298238)]
DEBUG:root:			----------- 
DEBUG:root:			size: 64
DEBUG:root:			----------- 
DEBUG:root:			per_layer_anchor_sizes = []
DEBUG:root:
			for scale_per_octave in range(scales_per_octave) { // BEGIN
DEBUG:root:				------------------------
DEBUG:root:				scale_per_octave : 0
DEBUG:root:				------------------------
DEBUG:root:					octave : 2.0

DEBUG:root:				octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

DEBUG:root:					octave_scale: 1.0
DEBUG:root:					size: 64

DEBUG:root:				per_layer_anchor_sizes.append(octave_scale * size)
DEBUG:root:					per_layer_anchor_sizes: [64.0]

DEBUG:root:				------------------------
DEBUG:root:				scale_per_octave : 1
DEBUG:root:				------------------------
DEBUG:root:					octave : 2.0

DEBUG:root:				octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

DEBUG:root:					octave_scale: 1.2599210498948732
DEBUG:root:					size: 64

DEBUG:root:				per_layer_anchor_sizes.append(octave_scale * size)
DEBUG:root:					per_layer_anchor_sizes: [64.0, 80.63494719327188]

DEBUG:root:				------------------------
DEBUG:root:				scale_per_octave : 2
DEBUG:root:				------------------------
DEBUG:root:					octave : 2.0

DEBUG:root:				octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

DEBUG:root:					octave_scale: 1.5874010519681994
DEBUG:root:					size: 64

DEBUG:root:				per_layer_anchor_sizes.append(octave_scale * size)
DEBUG:root:					per_layer_anchor_sizes: [64.0, 80.63494719327188, 101.59366732596476]

DEBUG:root:			} // EDN for scale_per_octave in range(scales_per_octave)


DEBUG:root:			new_anchor_sizes.append(tuple(per_layer_anchor_sizes))
DEBUG:root:			new_anchor_sizes: [(32.0, 40.31747359663594, 50.79683366298238), (64.0, 80.63494719327188, 101.59366732596476)]
DEBUG:root:			----------- 
DEBUG:root:			size: 128
DEBUG:root:			----------- 
DEBUG:root:			per_layer_anchor_sizes = []
DEBUG:root:
			for scale_per_octave in range(scales_per_octave) { // BEGIN
DEBUG:root:				------------------------
DEBUG:root:				scale_per_octave : 0
DEBUG:root:				------------------------
DEBUG:root:					octave : 2.0

DEBUG:root:				octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

DEBUG:root:					octave_scale: 1.0
DEBUG:root:					size: 128

DEBUG:root:				per_layer_anchor_sizes.append(octave_scale * size)
DEBUG:root:					per_layer_anchor_sizes: [128.0]

DEBUG:root:				------------------------
DEBUG:root:				scale_per_octave : 1
DEBUG:root:				------------------------
DEBUG:root:					octave : 2.0

DEBUG:root:				octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

DEBUG:root:					octave_scale: 1.2599210498948732
DEBUG:root:					size: 128

DEBUG:root:				per_layer_anchor_sizes.append(octave_scale * size)
DEBUG:root:					per_layer_anchor_sizes: [128.0, 161.26989438654377]

DEBUG:root:				------------------------
DEBUG:root:				scale_per_octave : 2
DEBUG:root:				------------------------
DEBUG:root:					octave : 2.0

DEBUG:root:				octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

DEBUG:root:					octave_scale: 1.5874010519681994
DEBUG:root:					size: 128

DEBUG:root:				per_layer_anchor_sizes.append(octave_scale * size)
DEBUG:root:					per_layer_anchor_sizes: [128.0, 161.26989438654377, 203.18733465192952]

DEBUG:root:			} // EDN for scale_per_octave in range(scales_per_octave)


DEBUG:root:			new_anchor_sizes.append(tuple(per_layer_anchor_sizes))
DEBUG:root:			new_anchor_sizes: [(32.0, 40.31747359663594, 50.79683366298238), (64.0, 80.63494719327188, 101.59366732596476), (128.0, 161.26989438654377, 203.18733465192952)]
DEBUG:root:			----------- 
DEBUG:root:			size: 256
DEBUG:root:			----------- 
DEBUG:root:			per_layer_anchor_sizes = []
DEBUG:root:
			for scale_per_octave in range(scales_per_octave) { // BEGIN
DEBUG:root:				------------------------
DEBUG:root:				scale_per_octave : 0
DEBUG:root:				------------------------
DEBUG:root:					octave : 2.0

DEBUG:root:				octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

DEBUG:root:					octave_scale: 1.0
DEBUG:root:					size: 256

DEBUG:root:				per_layer_anchor_sizes.append(octave_scale * size)
DEBUG:root:					per_layer_anchor_sizes: [256.0]

DEBUG:root:				------------------------
DEBUG:root:				scale_per_octave : 1
DEBUG:root:				------------------------
DEBUG:root:					octave : 2.0

DEBUG:root:				octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

DEBUG:root:					octave_scale: 1.2599210498948732
DEBUG:root:					size: 256

DEBUG:root:				per_layer_anchor_sizes.append(octave_scale * size)
DEBUG:root:					per_layer_anchor_sizes: [256.0, 322.53978877308754]

DEBUG:root:				------------------------
DEBUG:root:				scale_per_octave : 2
DEBUG:root:				------------------------
DEBUG:root:					octave : 2.0

DEBUG:root:				octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

DEBUG:root:					octave_scale: 1.5874010519681994
DEBUG:root:					size: 256

DEBUG:root:				per_layer_anchor_sizes.append(octave_scale * size)
DEBUG:root:					per_layer_anchor_sizes: [256.0, 322.53978877308754, 406.37466930385904]

DEBUG:root:			} // EDN for scale_per_octave in range(scales_per_octave)


DEBUG:root:			new_anchor_sizes.append(tuple(per_layer_anchor_sizes))
DEBUG:root:			new_anchor_sizes: [(32.0, 40.31747359663594, 50.79683366298238), (64.0, 80.63494719327188, 101.59366732596476), (128.0, 161.26989438654377, 203.18733465192952), (256.0, 322.53978877308754, 406.37466930385904)]
DEBUG:root:			----------- 
DEBUG:root:			size: 512
DEBUG:root:			----------- 
DEBUG:root:			per_layer_anchor_sizes = []
DEBUG:root:
			for scale_per_octave in range(scales_per_octave) { // BEGIN
DEBUG:root:				------------------------
DEBUG:root:				scale_per_octave : 0
DEBUG:root:				------------------------
DEBUG:root:					octave : 2.0

DEBUG:root:				octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

DEBUG:root:					octave_scale: 1.0
DEBUG:root:					size: 512

DEBUG:root:				per_layer_anchor_sizes.append(octave_scale * size)
DEBUG:root:					per_layer_anchor_sizes: [512.0]

DEBUG:root:				------------------------
DEBUG:root:				scale_per_octave : 1
DEBUG:root:				------------------------
DEBUG:root:					octave : 2.0

DEBUG:root:				octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

DEBUG:root:					octave_scale: 1.2599210498948732
DEBUG:root:					size: 512

DEBUG:root:				per_layer_anchor_sizes.append(octave_scale * size)
DEBUG:root:					per_layer_anchor_sizes: [512.0, 645.0795775461751]

DEBUG:root:				------------------------
DEBUG:root:				scale_per_octave : 2
DEBUG:root:				------------------------
DEBUG:root:					octave : 2.0

DEBUG:root:				octave_scale = octave ** (scale_per_octave / float(scales_per_octave))

DEBUG:root:					octave_scale: 1.5874010519681994
DEBUG:root:					size: 512

DEBUG:root:				per_layer_anchor_sizes.append(octave_scale * size)
DEBUG:root:					per_layer_anchor_sizes: [512.0, 645.0795775461751, 812.7493386077181]

DEBUG:root:			} // EDN for scale_per_octave in range(scales_per_octave)


DEBUG:root:			new_anchor_sizes.append(tuple(per_layer_anchor_sizes))
DEBUG:root:			new_anchor_sizes: [(32.0, 40.31747359663594, 50.79683366298238), (64.0, 80.63494719327188, 101.59366732596476), (128.0, 161.26989438654377, 203.18733465192952), (256.0, 322.53978877308754, 406.37466930385904), (512.0, 645.0795775461751, 812.7493386077181)]
DEBUG:root:		} // END for size in anchor_sizes

DEBUG:root:		new_anchor_sizes:
			[(32.0, 40.31747359663594, 50.79683366298238), (64.0, 80.63494719327188, 101.59366732596476), (128.0, 161.26989438654377, 203.18733465192952), (256.0, 322.53978877308754, 406.37466930385904), (512.0, 645.0795775461751, 812.7493386077181)]
DEBUG:root:		aspect_ratios:
			(0.5, 1.0, 2.0)
DEBUG:root:		anchor_strides:
			(8, 16, 32, 64, 128)
DEBUG:root:		straddle_thresh:
			-1
DEBUG:root:		anchor_generator = AnchorGenerator( tuple(new_anchor_sizes), aspect_ratios, anchor_strides, straddle_thresh )
DEBUG:root:		AnchorGenerator.__init__(sizes, aspect_ratios, anchor_strides, straddle_thresh) { //BEGIN
DEBUG:root:		// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:			Params
DEBUG:root:				sizes: ((32.0, 40.31747359663594, 50.79683366298238), (64.0, 80.63494719327188, 101.59366732596476), (128.0, 161.26989438654377, 203.18733465192952), (256.0, 322.53978877308754, 406.37466930385904), (512.0, 645.0795775461751, 812.7493386077181))
DEBUG:root:				aspect_ratios: (0.5, 1.0, 2.0)
DEBUG:root:				anchor_strides: (8, 16, 32, 64, 128)
DEBUG:root:				straddle_thresh: -1
DEBUG:root:		else: i.e, len(anchor_strides) !=1
DEBUG:root:			anchor_stride = anchor_strides[0]
DEBUG:root:			len(anchor_strides):5, len(size): 5
DEBUG:root:		else: i.e, len(anchor_strides) == len(sizes)
DEBUG:root:		cell_anchors = [ generate_anchors( anchor_stride,
DEBUG:root:		                 size if isinstance(size, (tuple, list)) else (size,), 
DEBUG:root:		                 aspect_ratios).float()
DEBUG:root:		for anchor_stride, size in zip(anchor_strides, sizes)
DEBUG:root:				generate_anchors(stride, sizes, aspect_ratios) { //BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Params:
DEBUG:root:						stride: 8
DEBUG:root:						sizes: (32.0, 40.31747359663594, 50.79683366298238)
DEBUG:root:						aspect_ratios: (0.5, 1.0, 2.0)
DEBUG:root:					return _generate_anchors(stride,
DEBUG:root:						     np.array(sizes, dtype=np.float) / stride,
DEBUG:root:						     np.array(aspect_ratios, dtype=np.float),)
DEBUG:root:						_generate_anchors(base_size, scales, aspect_ratios) { //BEGIN
DEBUG:root:						/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:							Params:
DEBUG:root:								base_size: 8
DEBUG:root:								scales: [4.         5.0396842  6.34960421]
DEBUG:root:								aspect_ratios: [0.5 1.  2. ]
DEBUG:root:anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
DEBUG:root:anchor: [0. 0. 7. 7.]
DEBUG:root:anchors = _ratio_enum(anchor, aspect_ratios)
DEBUG:root:				_ratio_enum(anchor, ratios) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [0. 0. 7. 7.]
DEBUG:root:						ratios: [0.5 1.  2. ]
DEBUG:root:				_whctrs(anchors) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [0. 0. 7. 7.]
DEBUG:root:						w = anchor[2] - anchor[0] + 1
DEBUG:root:						w: 8.0
DEBUG:root:						h = anchor[3] - anchor[1] + 1
DEBUG:root:						h: 8.0
DEBUG:root:						x_ctr = anchor[0] + 0.5 * (w - 1)
DEBUG:root:						x_ctr: 3.5
DEBUG:root:						y_ctr = anchor[1] + 0.5 * (h - 1)
DEBUG:root:						y_ctr: 3.5
DEBUG:root:						return w, h, x_ctr, y_ctr
DEBUG:root:					} // END _whctrs(anchors)
DEBUG:root:					w, h, x_ctr, y_ctr = _whctrs(anchor)
DEBUG:root:					w: 8.0, h: 8.0, x_ctr: 3.5, y_ctr: 3.5
DEBUG:root:					size = w * h
DEBUG:root:					size: 64.0
DEBUG:root:					size_ratios = size / ratios
DEBUG:root:					size_ratios: [128.  64.  32.]
DEBUG:root:					ws = np.round(np.sqrt(size_ratios))
DEBUG:root:					ws: [11.  8.  6.]
DEBUG:root:					hs = np.round(ws * ratios)
DEBUG:root:					hs: [ 6.  8. 12.]
DEBUG:root:			_mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:				Param:
DEBUG:root:					ws: [11.  8.  6.]
DEBUG:root:					hs: [ 6.  8. 12.]
DEBUG:root:					x_ctr: 3.5
DEBUG:root:					y_ctr: 3.5
DEBUG:root:				ws = ws[:, np.newaxis]
DEBUG:root:					ws: [[11.]
 [ 8.]
 [ 6.]]
DEBUG:root:				hs = hs[:, np.newaxis]
DEBUG:root:					hs: [[ 6.]
 [ 8.]
 [12.]]
DEBUG:root:				anchors = np.hstack(
DEBUG:root:				    (
DEBUG:root:				        x_ctr - 0.5 * (ws - 1),
DEBUG:root:				        y_ctr - 0.5 * (hs - 1),
DEBUG:root:				        x_ctr + 0.5 * (ws - 1),
DEBUG:root:				        y_ctr + 0.5 * (hs - 1),
DEBUG:root:				    )
DEBUG:root:				)
DEBUG:root:				anchors: [[-1.5  1.   8.5  6. ]
 [ 0.   0.   7.   7. ]
 [ 1.  -2.   6.   9. ]]
DEBUG:root:				return anchors
DEBUG:root:			} // END _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors: [[-1.5  1.   8.5  6. ]
 [ 0.   0.   7.   7. ]
 [ 1.  -2.   6.   9. ]]
DEBUG:root:					return anchors
DEBUG:root:				} // END _ratio_enum(anchor, ratios)
DEBUG:root:anchors: [[-1.5  1.   8.5  6. ]
 [ 0.   0.   7.   7. ]
 [ 1.  -2.   6.   9. ]]
DEBUG:root:anchors = np.vstack(
DEBUG:root:[_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
DEBUG:root:)
DEBUG:root:					_scale_enum(anchor, scales) { //BEGIN
DEBUG:root:					/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [-1.5  1.   8.5  6. ]
DEBUG:root:						scales: [4.         5.0396842  6.34960421]
DEBUG:root:					w, h, x_ctr, y_ctr = _whctrs(anchor)
DEBUG:root:				_whctrs(anchors) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [-1.5  1.   8.5  6. ]
DEBUG:root:						w = anchor[2] - anchor[0] + 1
DEBUG:root:						w: 11.0
DEBUG:root:						h = anchor[3] - anchor[1] + 1
DEBUG:root:						h: 6.0
DEBUG:root:						x_ctr = anchor[0] + 0.5 * (w - 1)
DEBUG:root:						x_ctr: 3.5
DEBUG:root:						y_ctr = anchor[1] + 0.5 * (h - 1)
DEBUG:root:						y_ctr: 3.5
DEBUG:root:						return w, h, x_ctr, y_ctr
DEBUG:root:					} // END _whctrs(anchors)
DEBUG:root:					w: 11.0, h: 6.0, x_ctr: 3.5, y_ctr: 3.5
DEBUG:root:					ws = w * scale
DEBUG:root:					ws: [44.         55.4365262  69.84564629]
DEBUG:root:					hs = h * scale
DEBUG:root:					hs: [24.         30.2381052  38.09762525]
DEBUG:root:					anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:			_mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:				Param:
DEBUG:root:					ws: [44.         55.4365262  69.84564629]
DEBUG:root:					hs: [24.         30.2381052  38.09762525]
DEBUG:root:					x_ctr: 3.5
DEBUG:root:					y_ctr: 3.5
DEBUG:root:				ws = ws[:, np.newaxis]
DEBUG:root:					ws: [[44.        ]
 [55.4365262 ]
 [69.84564629]]
DEBUG:root:				hs = hs[:, np.newaxis]
DEBUG:root:					hs: [[24.        ]
 [30.2381052 ]
 [38.09762525]]
DEBUG:root:				anchors = np.hstack(
DEBUG:root:				    (
DEBUG:root:				        x_ctr - 0.5 * (ws - 1),
DEBUG:root:				        y_ctr - 0.5 * (hs - 1),
DEBUG:root:				        x_ctr + 0.5 * (ws - 1),
DEBUG:root:				        y_ctr + 0.5 * (hs - 1),
DEBUG:root:				    )
DEBUG:root:				)
DEBUG:root:				anchors: [[-18.          -8.          25.          15.        ]
 [-23.7182631  -11.1190526   30.7182631   18.1190526 ]
 [-30.92282314 -15.04881262  37.92282314  22.04881262]]
DEBUG:root:				return anchors
DEBUG:root:			} // END _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors: [[-18.          -8.          25.          15.        ]
 [-23.7182631  -11.1190526   30.7182631   18.1190526 ]
 [-30.92282314 -15.04881262  37.92282314  22.04881262]]
DEBUG:root:				} // END _scale_enum(anchor, scales)
DEBUG:root:					_scale_enum(anchor, scales) { //BEGIN
DEBUG:root:					/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [0. 0. 7. 7.]
DEBUG:root:						scales: [4.         5.0396842  6.34960421]
DEBUG:root:					w, h, x_ctr, y_ctr = _whctrs(anchor)
DEBUG:root:				_whctrs(anchors) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [0. 0. 7. 7.]
DEBUG:root:						w = anchor[2] - anchor[0] + 1
DEBUG:root:						w: 8.0
DEBUG:root:						h = anchor[3] - anchor[1] + 1
DEBUG:root:						h: 8.0
DEBUG:root:						x_ctr = anchor[0] + 0.5 * (w - 1)
DEBUG:root:						x_ctr: 3.5
DEBUG:root:						y_ctr = anchor[1] + 0.5 * (h - 1)
DEBUG:root:						y_ctr: 3.5
DEBUG:root:						return w, h, x_ctr, y_ctr
DEBUG:root:					} // END _whctrs(anchors)
DEBUG:root:					w: 8.0, h: 8.0, x_ctr: 3.5, y_ctr: 3.5
DEBUG:root:					ws = w * scale
DEBUG:root:					ws: [32.         40.3174736  50.79683366]
DEBUG:root:					hs = h * scale
DEBUG:root:					hs: [32.         40.3174736  50.79683366]
DEBUG:root:					anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:			_mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:				Param:
DEBUG:root:					ws: [32.         40.3174736  50.79683366]
DEBUG:root:					hs: [32.         40.3174736  50.79683366]
DEBUG:root:					x_ctr: 3.5
DEBUG:root:					y_ctr: 3.5
DEBUG:root:				ws = ws[:, np.newaxis]
DEBUG:root:					ws: [[32.        ]
 [40.3174736 ]
 [50.79683366]]
DEBUG:root:				hs = hs[:, np.newaxis]
DEBUG:root:					hs: [[32.        ]
 [40.3174736 ]
 [50.79683366]]
DEBUG:root:				anchors = np.hstack(
DEBUG:root:				    (
DEBUG:root:				        x_ctr - 0.5 * (ws - 1),
DEBUG:root:				        y_ctr - 0.5 * (hs - 1),
DEBUG:root:				        x_ctr + 0.5 * (ws - 1),
DEBUG:root:				        y_ctr + 0.5 * (hs - 1),
DEBUG:root:				    )
DEBUG:root:				)
DEBUG:root:				anchors: [[-12.         -12.          19.          19.        ]
 [-16.1587368  -16.1587368   23.1587368   23.1587368 ]
 [-21.39841683 -21.39841683  28.39841683  28.39841683]]
DEBUG:root:				return anchors
DEBUG:root:			} // END _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors: [[-12.         -12.          19.          19.        ]
 [-16.1587368  -16.1587368   23.1587368   23.1587368 ]
 [-21.39841683 -21.39841683  28.39841683  28.39841683]]
DEBUG:root:				} // END _scale_enum(anchor, scales)
DEBUG:root:					_scale_enum(anchor, scales) { //BEGIN
DEBUG:root:					/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [ 1. -2.  6.  9.]
DEBUG:root:						scales: [4.         5.0396842  6.34960421]
DEBUG:root:					w, h, x_ctr, y_ctr = _whctrs(anchor)
DEBUG:root:				_whctrs(anchors) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [ 1. -2.  6.  9.]
DEBUG:root:						w = anchor[2] - anchor[0] + 1
DEBUG:root:						w: 6.0
DEBUG:root:						h = anchor[3] - anchor[1] + 1
DEBUG:root:						h: 12.0
DEBUG:root:						x_ctr = anchor[0] + 0.5 * (w - 1)
DEBUG:root:						x_ctr: 3.5
DEBUG:root:						y_ctr = anchor[1] + 0.5 * (h - 1)
DEBUG:root:						y_ctr: 3.5
DEBUG:root:						return w, h, x_ctr, y_ctr
DEBUG:root:					} // END _whctrs(anchors)
DEBUG:root:					w: 6.0, h: 12.0, x_ctr: 3.5, y_ctr: 3.5
DEBUG:root:					ws = w * scale
DEBUG:root:					ws: [24.         30.2381052  38.09762525]
DEBUG:root:					hs = h * scale
DEBUG:root:					hs: [48.         60.47621039 76.19525049]
DEBUG:root:					anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:			_mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:				Param:
DEBUG:root:					ws: [24.         30.2381052  38.09762525]
DEBUG:root:					hs: [48.         60.47621039 76.19525049]
DEBUG:root:					x_ctr: 3.5
DEBUG:root:					y_ctr: 3.5
DEBUG:root:				ws = ws[:, np.newaxis]
DEBUG:root:					ws: [[24.        ]
 [30.2381052 ]
 [38.09762525]]
DEBUG:root:				hs = hs[:, np.newaxis]
DEBUG:root:					hs: [[48.        ]
 [60.47621039]
 [76.19525049]]
DEBUG:root:				anchors = np.hstack(
DEBUG:root:				    (
DEBUG:root:				        x_ctr - 0.5 * (ws - 1),
DEBUG:root:				        y_ctr - 0.5 * (hs - 1),
DEBUG:root:				        x_ctr + 0.5 * (ws - 1),
DEBUG:root:				        y_ctr + 0.5 * (hs - 1),
DEBUG:root:				    )
DEBUG:root:				)
DEBUG:root:				anchors: [[ -8.         -20.          15.          27.        ]
 [-11.1190526  -26.2381052   18.1190526   33.2381052 ]
 [-15.04881262 -34.09762525  22.04881262  41.09762525]]
DEBUG:root:				return anchors
DEBUG:root:			} // END _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors: [[ -8.         -20.          15.          27.        ]
 [-11.1190526  -26.2381052   18.1190526   33.2381052 ]
 [-15.04881262 -34.09762525  22.04881262  41.09762525]]
DEBUG:root:				} // END _scale_enum(anchor, scales)
DEBUG:root:anchors: [[-18.          -8.          25.          15.        ]
 [-23.7182631  -11.1190526   30.7182631   18.1190526 ]
 [-30.92282314 -15.04881262  37.92282314  22.04881262]
 [-12.         -12.          19.          19.        ]
 [-16.1587368  -16.1587368   23.1587368   23.1587368 ]
 [-21.39841683 -21.39841683  28.39841683  28.39841683]
 [ -8.         -20.          15.          27.        ]
 [-11.1190526  -26.2381052   18.1190526   33.2381052 ]
 [-15.04881262 -34.09762525  22.04881262  41.09762525]]
DEBUG:root:							return torch.from_numpy(anchors)
DEBUG:root:						} // END _generate_anchors(base_size, scales, apect_ratios) END
DEBUG:root:					} // END generate_anchors(stride, sizes, aspect_ratios)
DEBUG:root:				generate_anchors(stride, sizes, aspect_ratios) { //BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Params:
DEBUG:root:						stride: 16
DEBUG:root:						sizes: (64.0, 80.63494719327188, 101.59366732596476)
DEBUG:root:						aspect_ratios: (0.5, 1.0, 2.0)
DEBUG:root:					return _generate_anchors(stride,
DEBUG:root:						     np.array(sizes, dtype=np.float) / stride,
DEBUG:root:						     np.array(aspect_ratios, dtype=np.float),)
DEBUG:root:						_generate_anchors(base_size, scales, aspect_ratios) { //BEGIN
DEBUG:root:						/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:							Params:
DEBUG:root:								base_size: 16
DEBUG:root:								scales: [4.         5.0396842  6.34960421]
DEBUG:root:								aspect_ratios: [0.5 1.  2. ]
DEBUG:root:anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
DEBUG:root:anchor: [ 0.  0. 15. 15.]
DEBUG:root:anchors = _ratio_enum(anchor, aspect_ratios)
DEBUG:root:				_ratio_enum(anchor, ratios) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [ 0.  0. 15. 15.]
DEBUG:root:						ratios: [0.5 1.  2. ]
DEBUG:root:				_whctrs(anchors) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [ 0.  0. 15. 15.]
DEBUG:root:						w = anchor[2] - anchor[0] + 1
DEBUG:root:						w: 16.0
DEBUG:root:						h = anchor[3] - anchor[1] + 1
DEBUG:root:						h: 16.0
DEBUG:root:						x_ctr = anchor[0] + 0.5 * (w - 1)
DEBUG:root:						x_ctr: 7.5
DEBUG:root:						y_ctr = anchor[1] + 0.5 * (h - 1)
DEBUG:root:						y_ctr: 7.5
DEBUG:root:						return w, h, x_ctr, y_ctr
DEBUG:root:					} // END _whctrs(anchors)
DEBUG:root:					w, h, x_ctr, y_ctr = _whctrs(anchor)
DEBUG:root:					w: 16.0, h: 16.0, x_ctr: 7.5, y_ctr: 7.5
DEBUG:root:					size = w * h
DEBUG:root:					size: 256.0
DEBUG:root:					size_ratios = size / ratios
DEBUG:root:					size_ratios: [512. 256. 128.]
DEBUG:root:					ws = np.round(np.sqrt(size_ratios))
DEBUG:root:					ws: [23. 16. 11.]
DEBUG:root:					hs = np.round(ws * ratios)
DEBUG:root:					hs: [12. 16. 22.]
DEBUG:root:			_mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:				Param:
DEBUG:root:					ws: [23. 16. 11.]
DEBUG:root:					hs: [12. 16. 22.]
DEBUG:root:					x_ctr: 7.5
DEBUG:root:					y_ctr: 7.5
DEBUG:root:				ws = ws[:, np.newaxis]
DEBUG:root:					ws: [[23.]
 [16.]
 [11.]]
DEBUG:root:				hs = hs[:, np.newaxis]
DEBUG:root:					hs: [[12.]
 [16.]
 [22.]]
DEBUG:root:				anchors = np.hstack(
DEBUG:root:				    (
DEBUG:root:				        x_ctr - 0.5 * (ws - 1),
DEBUG:root:				        y_ctr - 0.5 * (hs - 1),
DEBUG:root:				        x_ctr + 0.5 * (ws - 1),
DEBUG:root:				        y_ctr + 0.5 * (hs - 1),
DEBUG:root:				    )
DEBUG:root:				)
DEBUG:root:				anchors: [[-3.5  2.  18.5 13. ]
 [ 0.   0.  15.  15. ]
 [ 2.5 -3.  12.5 18. ]]
DEBUG:root:				return anchors
DEBUG:root:			} // END _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors: [[-3.5  2.  18.5 13. ]
 [ 0.   0.  15.  15. ]
 [ 2.5 -3.  12.5 18. ]]
DEBUG:root:					return anchors
DEBUG:root:				} // END _ratio_enum(anchor, ratios)
DEBUG:root:anchors: [[-3.5  2.  18.5 13. ]
 [ 0.   0.  15.  15. ]
 [ 2.5 -3.  12.5 18. ]]
DEBUG:root:anchors = np.vstack(
DEBUG:root:[_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
DEBUG:root:)
DEBUG:root:					_scale_enum(anchor, scales) { //BEGIN
DEBUG:root:					/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [-3.5  2.  18.5 13. ]
DEBUG:root:						scales: [4.         5.0396842  6.34960421]
DEBUG:root:					w, h, x_ctr, y_ctr = _whctrs(anchor)
DEBUG:root:				_whctrs(anchors) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [-3.5  2.  18.5 13. ]
DEBUG:root:						w = anchor[2] - anchor[0] + 1
DEBUG:root:						w: 23.0
DEBUG:root:						h = anchor[3] - anchor[1] + 1
DEBUG:root:						h: 12.0
DEBUG:root:						x_ctr = anchor[0] + 0.5 * (w - 1)
DEBUG:root:						x_ctr: 7.5
DEBUG:root:						y_ctr = anchor[1] + 0.5 * (h - 1)
DEBUG:root:						y_ctr: 7.5
DEBUG:root:						return w, h, x_ctr, y_ctr
DEBUG:root:					} // END _whctrs(anchors)
DEBUG:root:					w: 23.0, h: 12.0, x_ctr: 7.5, y_ctr: 7.5
DEBUG:root:					ws = w * scale
DEBUG:root:					ws: [ 92.         115.91273659 146.04089678]
DEBUG:root:					hs = h * scale
DEBUG:root:					hs: [48.         60.47621039 76.19525049]
DEBUG:root:					anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:			_mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:				Param:
DEBUG:root:					ws: [ 92.         115.91273659 146.04089678]
DEBUG:root:					hs: [48.         60.47621039 76.19525049]
DEBUG:root:					x_ctr: 7.5
DEBUG:root:					y_ctr: 7.5
DEBUG:root:				ws = ws[:, np.newaxis]
DEBUG:root:					ws: [[ 92.        ]
 [115.91273659]
 [146.04089678]]
DEBUG:root:				hs = hs[:, np.newaxis]
DEBUG:root:					hs: [[48.        ]
 [60.47621039]
 [76.19525049]]
DEBUG:root:				anchors = np.hstack(
DEBUG:root:				    (
DEBUG:root:				        x_ctr - 0.5 * (ws - 1),
DEBUG:root:				        y_ctr - 0.5 * (hs - 1),
DEBUG:root:				        x_ctr + 0.5 * (ws - 1),
DEBUG:root:				        y_ctr + 0.5 * (hs - 1),
DEBUG:root:				    )
DEBUG:root:				)
DEBUG:root:				anchors: [[-38.         -16.          53.          31.        ]
 [-49.9563683  -22.2381052   64.9563683   37.2381052 ]
 [-65.02044839 -30.09762525  80.02044839  45.09762525]]
DEBUG:root:				return anchors
DEBUG:root:			} // END _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors: [[-38.         -16.          53.          31.        ]
 [-49.9563683  -22.2381052   64.9563683   37.2381052 ]
 [-65.02044839 -30.09762525  80.02044839  45.09762525]]
DEBUG:root:				} // END _scale_enum(anchor, scales)
DEBUG:root:					_scale_enum(anchor, scales) { //BEGIN
DEBUG:root:					/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [ 0.  0. 15. 15.]
DEBUG:root:						scales: [4.         5.0396842  6.34960421]
DEBUG:root:					w, h, x_ctr, y_ctr = _whctrs(anchor)
DEBUG:root:				_whctrs(anchors) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [ 0.  0. 15. 15.]
DEBUG:root:						w = anchor[2] - anchor[0] + 1
DEBUG:root:						w: 16.0
DEBUG:root:						h = anchor[3] - anchor[1] + 1
DEBUG:root:						h: 16.0
DEBUG:root:						x_ctr = anchor[0] + 0.5 * (w - 1)
DEBUG:root:						x_ctr: 7.5
DEBUG:root:						y_ctr = anchor[1] + 0.5 * (h - 1)
DEBUG:root:						y_ctr: 7.5
DEBUG:root:						return w, h, x_ctr, y_ctr
DEBUG:root:					} // END _whctrs(anchors)
DEBUG:root:					w: 16.0, h: 16.0, x_ctr: 7.5, y_ctr: 7.5
DEBUG:root:					ws = w * scale
DEBUG:root:					ws: [ 64.          80.63494719 101.59366733]
DEBUG:root:					hs = h * scale
DEBUG:root:					hs: [ 64.          80.63494719 101.59366733]
DEBUG:root:					anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:			_mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:				Param:
DEBUG:root:					ws: [ 64.          80.63494719 101.59366733]
DEBUG:root:					hs: [ 64.          80.63494719 101.59366733]
DEBUG:root:					x_ctr: 7.5
DEBUG:root:					y_ctr: 7.5
DEBUG:root:				ws = ws[:, np.newaxis]
DEBUG:root:					ws: [[ 64.        ]
 [ 80.63494719]
 [101.59366733]]
DEBUG:root:				hs = hs[:, np.newaxis]
DEBUG:root:					hs: [[ 64.        ]
 [ 80.63494719]
 [101.59366733]]
DEBUG:root:				anchors = np.hstack(
DEBUG:root:				    (
DEBUG:root:				        x_ctr - 0.5 * (ws - 1),
DEBUG:root:				        y_ctr - 0.5 * (hs - 1),
DEBUG:root:				        x_ctr + 0.5 * (ws - 1),
DEBUG:root:				        y_ctr + 0.5 * (hs - 1),
DEBUG:root:				    )
DEBUG:root:				)
DEBUG:root:				anchors: [[-24.         -24.          39.          39.        ]
 [-32.3174736  -32.3174736   47.3174736   47.3174736 ]
 [-42.79683366 -42.79683366  57.79683366  57.79683366]]
DEBUG:root:				return anchors
DEBUG:root:			} // END _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors: [[-24.         -24.          39.          39.        ]
 [-32.3174736  -32.3174736   47.3174736   47.3174736 ]
 [-42.79683366 -42.79683366  57.79683366  57.79683366]]
DEBUG:root:				} // END _scale_enum(anchor, scales)
DEBUG:root:					_scale_enum(anchor, scales) { //BEGIN
DEBUG:root:					/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [ 2.5 -3.  12.5 18. ]
DEBUG:root:						scales: [4.         5.0396842  6.34960421]
DEBUG:root:					w, h, x_ctr, y_ctr = _whctrs(anchor)
DEBUG:root:				_whctrs(anchors) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [ 2.5 -3.  12.5 18. ]
DEBUG:root:						w = anchor[2] - anchor[0] + 1
DEBUG:root:						w: 11.0
DEBUG:root:						h = anchor[3] - anchor[1] + 1
DEBUG:root:						h: 22.0
DEBUG:root:						x_ctr = anchor[0] + 0.5 * (w - 1)
DEBUG:root:						x_ctr: 7.5
DEBUG:root:						y_ctr = anchor[1] + 0.5 * (h - 1)
DEBUG:root:						y_ctr: 7.5
DEBUG:root:						return w, h, x_ctr, y_ctr
DEBUG:root:					} // END _whctrs(anchors)
DEBUG:root:					w: 11.0, h: 22.0, x_ctr: 7.5, y_ctr: 7.5
DEBUG:root:					ws = w * scale
DEBUG:root:					ws: [44.         55.4365262  69.84564629]
DEBUG:root:					hs = h * scale
DEBUG:root:					hs: [ 88.         110.87305239 139.69129257]
DEBUG:root:					anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:			_mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:				Param:
DEBUG:root:					ws: [44.         55.4365262  69.84564629]
DEBUG:root:					hs: [ 88.         110.87305239 139.69129257]
DEBUG:root:					x_ctr: 7.5
DEBUG:root:					y_ctr: 7.5
DEBUG:root:				ws = ws[:, np.newaxis]
DEBUG:root:					ws: [[44.        ]
 [55.4365262 ]
 [69.84564629]]
DEBUG:root:				hs = hs[:, np.newaxis]
DEBUG:root:					hs: [[ 88.        ]
 [110.87305239]
 [139.69129257]]
DEBUG:root:				anchors = np.hstack(
DEBUG:root:				    (
DEBUG:root:				        x_ctr - 0.5 * (ws - 1),
DEBUG:root:				        y_ctr - 0.5 * (hs - 1),
DEBUG:root:				        x_ctr + 0.5 * (ws - 1),
DEBUG:root:				        y_ctr + 0.5 * (hs - 1),
DEBUG:root:				    )
DEBUG:root:				)
DEBUG:root:				anchors: [[-14.         -36.          29.          51.        ]
 [-19.7182631  -47.4365262   34.7182631   62.4365262 ]
 [-26.92282314 -61.84564629  41.92282314  76.84564629]]
DEBUG:root:				return anchors
DEBUG:root:			} // END _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors: [[-14.         -36.          29.          51.        ]
 [-19.7182631  -47.4365262   34.7182631   62.4365262 ]
 [-26.92282314 -61.84564629  41.92282314  76.84564629]]
DEBUG:root:				} // END _scale_enum(anchor, scales)
DEBUG:root:anchors: [[-38.         -16.          53.          31.        ]
 [-49.9563683  -22.2381052   64.9563683   37.2381052 ]
 [-65.02044839 -30.09762525  80.02044839  45.09762525]
 [-24.         -24.          39.          39.        ]
 [-32.3174736  -32.3174736   47.3174736   47.3174736 ]
 [-42.79683366 -42.79683366  57.79683366  57.79683366]
 [-14.         -36.          29.          51.        ]
 [-19.7182631  -47.4365262   34.7182631   62.4365262 ]
 [-26.92282314 -61.84564629  41.92282314  76.84564629]]
DEBUG:root:							return torch.from_numpy(anchors)
DEBUG:root:						} // END _generate_anchors(base_size, scales, apect_ratios) END
DEBUG:root:					} // END generate_anchors(stride, sizes, aspect_ratios)
DEBUG:root:				generate_anchors(stride, sizes, aspect_ratios) { //BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Params:
DEBUG:root:						stride: 32
DEBUG:root:						sizes: (128.0, 161.26989438654377, 203.18733465192952)
DEBUG:root:						aspect_ratios: (0.5, 1.0, 2.0)
DEBUG:root:					return _generate_anchors(stride,
DEBUG:root:						     np.array(sizes, dtype=np.float) / stride,
DEBUG:root:						     np.array(aspect_ratios, dtype=np.float),)
DEBUG:root:						_generate_anchors(base_size, scales, aspect_ratios) { //BEGIN
DEBUG:root:						/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:							Params:
DEBUG:root:								base_size: 32
DEBUG:root:								scales: [4.         5.0396842  6.34960421]
DEBUG:root:								aspect_ratios: [0.5 1.  2. ]
DEBUG:root:anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
DEBUG:root:anchor: [ 0.  0. 31. 31.]
DEBUG:root:anchors = _ratio_enum(anchor, aspect_ratios)
DEBUG:root:				_ratio_enum(anchor, ratios) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [ 0.  0. 31. 31.]
DEBUG:root:						ratios: [0.5 1.  2. ]
DEBUG:root:				_whctrs(anchors) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [ 0.  0. 31. 31.]
DEBUG:root:						w = anchor[2] - anchor[0] + 1
DEBUG:root:						w: 32.0
DEBUG:root:						h = anchor[3] - anchor[1] + 1
DEBUG:root:						h: 32.0
DEBUG:root:						x_ctr = anchor[0] + 0.5 * (w - 1)
DEBUG:root:						x_ctr: 15.5
DEBUG:root:						y_ctr = anchor[1] + 0.5 * (h - 1)
DEBUG:root:						y_ctr: 15.5
DEBUG:root:						return w, h, x_ctr, y_ctr
DEBUG:root:					} // END _whctrs(anchors)
DEBUG:root:					w, h, x_ctr, y_ctr = _whctrs(anchor)
DEBUG:root:					w: 32.0, h: 32.0, x_ctr: 15.5, y_ctr: 15.5
DEBUG:root:					size = w * h
DEBUG:root:					size: 1024.0
DEBUG:root:					size_ratios = size / ratios
DEBUG:root:					size_ratios: [2048. 1024.  512.]
DEBUG:root:					ws = np.round(np.sqrt(size_ratios))
DEBUG:root:					ws: [45. 32. 23.]
DEBUG:root:					hs = np.round(ws * ratios)
DEBUG:root:					hs: [22. 32. 46.]
DEBUG:root:			_mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:				Param:
DEBUG:root:					ws: [45. 32. 23.]
DEBUG:root:					hs: [22. 32. 46.]
DEBUG:root:					x_ctr: 15.5
DEBUG:root:					y_ctr: 15.5
DEBUG:root:				ws = ws[:, np.newaxis]
DEBUG:root:					ws: [[45.]
 [32.]
 [23.]]
DEBUG:root:				hs = hs[:, np.newaxis]
DEBUG:root:					hs: [[22.]
 [32.]
 [46.]]
DEBUG:root:				anchors = np.hstack(
DEBUG:root:				    (
DEBUG:root:				        x_ctr - 0.5 * (ws - 1),
DEBUG:root:				        y_ctr - 0.5 * (hs - 1),
DEBUG:root:				        x_ctr + 0.5 * (ws - 1),
DEBUG:root:				        y_ctr + 0.5 * (hs - 1),
DEBUG:root:				    )
DEBUG:root:				)
DEBUG:root:				anchors: [[-6.5  5.  37.5 26. ]
 [ 0.   0.  31.  31. ]
 [ 4.5 -7.  26.5 38. ]]
DEBUG:root:				return anchors
DEBUG:root:			} // END _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors: [[-6.5  5.  37.5 26. ]
 [ 0.   0.  31.  31. ]
 [ 4.5 -7.  26.5 38. ]]
DEBUG:root:					return anchors
DEBUG:root:				} // END _ratio_enum(anchor, ratios)
DEBUG:root:anchors: [[-6.5  5.  37.5 26. ]
 [ 0.   0.  31.  31. ]
 [ 4.5 -7.  26.5 38. ]]
DEBUG:root:anchors = np.vstack(
DEBUG:root:[_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
DEBUG:root:)
DEBUG:root:					_scale_enum(anchor, scales) { //BEGIN
DEBUG:root:					/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [-6.5  5.  37.5 26. ]
DEBUG:root:						scales: [4.         5.0396842  6.34960421]
DEBUG:root:					w, h, x_ctr, y_ctr = _whctrs(anchor)
DEBUG:root:				_whctrs(anchors) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [-6.5  5.  37.5 26. ]
DEBUG:root:						w = anchor[2] - anchor[0] + 1
DEBUG:root:						w: 45.0
DEBUG:root:						h = anchor[3] - anchor[1] + 1
DEBUG:root:						h: 22.0
DEBUG:root:						x_ctr = anchor[0] + 0.5 * (w - 1)
DEBUG:root:						x_ctr: 15.5
DEBUG:root:						y_ctr = anchor[1] + 0.5 * (h - 1)
DEBUG:root:						y_ctr: 15.5
DEBUG:root:						return w, h, x_ctr, y_ctr
DEBUG:root:					} // END _whctrs(anchors)
DEBUG:root:					w: 45.0, h: 22.0, x_ctr: 15.5, y_ctr: 15.5
DEBUG:root:					ws = w * scale
DEBUG:root:					ws: [180.         226.78578898 285.73218935]
DEBUG:root:					hs = h * scale
DEBUG:root:					hs: [ 88.         110.87305239 139.69129257]
DEBUG:root:					anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:			_mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:				Param:
DEBUG:root:					ws: [180.         226.78578898 285.73218935]
DEBUG:root:					hs: [ 88.         110.87305239 139.69129257]
DEBUG:root:					x_ctr: 15.5
DEBUG:root:					y_ctr: 15.5
DEBUG:root:				ws = ws[:, np.newaxis]
DEBUG:root:					ws: [[180.        ]
 [226.78578898]
 [285.73218935]]
DEBUG:root:				hs = hs[:, np.newaxis]
DEBUG:root:					hs: [[ 88.        ]
 [110.87305239]
 [139.69129257]]
DEBUG:root:				anchors = np.hstack(
DEBUG:root:				    (
DEBUG:root:				        x_ctr - 0.5 * (ws - 1),
DEBUG:root:				        y_ctr - 0.5 * (hs - 1),
DEBUG:root:				        x_ctr + 0.5 * (ws - 1),
DEBUG:root:				        y_ctr + 0.5 * (hs - 1),
DEBUG:root:				    )
DEBUG:root:				)
DEBUG:root:				anchors: [[ -74.          -28.          105.           59.        ]
 [ -97.39289449  -39.4365262   128.39289449   70.4365262 ]
 [-126.86609468  -53.84564629  157.86609468   84.84564629]]
DEBUG:root:				return anchors
DEBUG:root:			} // END _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors: [[ -74.          -28.          105.           59.        ]
 [ -97.39289449  -39.4365262   128.39289449   70.4365262 ]
 [-126.86609468  -53.84564629  157.86609468   84.84564629]]
DEBUG:root:				} // END _scale_enum(anchor, scales)
DEBUG:root:					_scale_enum(anchor, scales) { //BEGIN
DEBUG:root:					/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [ 0.  0. 31. 31.]
DEBUG:root:						scales: [4.         5.0396842  6.34960421]
DEBUG:root:					w, h, x_ctr, y_ctr = _whctrs(anchor)
DEBUG:root:				_whctrs(anchors) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [ 0.  0. 31. 31.]
DEBUG:root:						w = anchor[2] - anchor[0] + 1
DEBUG:root:						w: 32.0
DEBUG:root:						h = anchor[3] - anchor[1] + 1
DEBUG:root:						h: 32.0
DEBUG:root:						x_ctr = anchor[0] + 0.5 * (w - 1)
DEBUG:root:						x_ctr: 15.5
DEBUG:root:						y_ctr = anchor[1] + 0.5 * (h - 1)
DEBUG:root:						y_ctr: 15.5
DEBUG:root:						return w, h, x_ctr, y_ctr
DEBUG:root:					} // END _whctrs(anchors)
DEBUG:root:					w: 32.0, h: 32.0, x_ctr: 15.5, y_ctr: 15.5
DEBUG:root:					ws = w * scale
DEBUG:root:					ws: [128.         161.26989439 203.18733465]
DEBUG:root:					hs = h * scale
DEBUG:root:					hs: [128.         161.26989439 203.18733465]
DEBUG:root:					anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:			_mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:				Param:
DEBUG:root:					ws: [128.         161.26989439 203.18733465]
DEBUG:root:					hs: [128.         161.26989439 203.18733465]
DEBUG:root:					x_ctr: 15.5
DEBUG:root:					y_ctr: 15.5
DEBUG:root:				ws = ws[:, np.newaxis]
DEBUG:root:					ws: [[128.        ]
 [161.26989439]
 [203.18733465]]
DEBUG:root:				hs = hs[:, np.newaxis]
DEBUG:root:					hs: [[128.        ]
 [161.26989439]
 [203.18733465]]
DEBUG:root:				anchors = np.hstack(
DEBUG:root:				    (
DEBUG:root:				        x_ctr - 0.5 * (ws - 1),
DEBUG:root:				        y_ctr - 0.5 * (hs - 1),
DEBUG:root:				        x_ctr + 0.5 * (ws - 1),
DEBUG:root:				        y_ctr + 0.5 * (hs - 1),
DEBUG:root:				    )
DEBUG:root:				)
DEBUG:root:				anchors: [[-48.         -48.          79.          79.        ]
 [-64.63494719 -64.63494719  95.63494719  95.63494719]
 [-85.59366733 -85.59366733 116.59366733 116.59366733]]
DEBUG:root:				return anchors
DEBUG:root:			} // END _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors: [[-48.         -48.          79.          79.        ]
 [-64.63494719 -64.63494719  95.63494719  95.63494719]
 [-85.59366733 -85.59366733 116.59366733 116.59366733]]
DEBUG:root:				} // END _scale_enum(anchor, scales)
DEBUG:root:					_scale_enum(anchor, scales) { //BEGIN
DEBUG:root:					/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [ 4.5 -7.  26.5 38. ]
DEBUG:root:						scales: [4.         5.0396842  6.34960421]
DEBUG:root:					w, h, x_ctr, y_ctr = _whctrs(anchor)
DEBUG:root:				_whctrs(anchors) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [ 4.5 -7.  26.5 38. ]
DEBUG:root:						w = anchor[2] - anchor[0] + 1
DEBUG:root:						w: 23.0
DEBUG:root:						h = anchor[3] - anchor[1] + 1
DEBUG:root:						h: 46.0
DEBUG:root:						x_ctr = anchor[0] + 0.5 * (w - 1)
DEBUG:root:						x_ctr: 15.5
DEBUG:root:						y_ctr = anchor[1] + 0.5 * (h - 1)
DEBUG:root:						y_ctr: 15.5
DEBUG:root:						return w, h, x_ctr, y_ctr
DEBUG:root:					} // END _whctrs(anchors)
DEBUG:root:					w: 23.0, h: 46.0, x_ctr: 15.5, y_ctr: 15.5
DEBUG:root:					ws = w * scale
DEBUG:root:					ws: [ 92.         115.91273659 146.04089678]
DEBUG:root:					hs = h * scale
DEBUG:root:					hs: [184.         231.82547318 292.08179356]
DEBUG:root:					anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:			_mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:				Param:
DEBUG:root:					ws: [ 92.         115.91273659 146.04089678]
DEBUG:root:					hs: [184.         231.82547318 292.08179356]
DEBUG:root:					x_ctr: 15.5
DEBUG:root:					y_ctr: 15.5
DEBUG:root:				ws = ws[:, np.newaxis]
DEBUG:root:					ws: [[ 92.        ]
 [115.91273659]
 [146.04089678]]
DEBUG:root:				hs = hs[:, np.newaxis]
DEBUG:root:					hs: [[184.        ]
 [231.82547318]
 [292.08179356]]
DEBUG:root:				anchors = np.hstack(
DEBUG:root:				    (
DEBUG:root:				        x_ctr - 0.5 * (ws - 1),
DEBUG:root:				        y_ctr - 0.5 * (hs - 1),
DEBUG:root:				        x_ctr + 0.5 * (ws - 1),
DEBUG:root:				        y_ctr + 0.5 * (hs - 1),
DEBUG:root:				    )
DEBUG:root:				)
DEBUG:root:				anchors: [[ -30.          -76.           61.          107.        ]
 [ -41.9563683   -99.91273659   72.9563683   130.91273659]
 [ -57.02044839 -130.04089678   88.02044839  161.04089678]]
DEBUG:root:				return anchors
DEBUG:root:			} // END _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors: [[ -30.          -76.           61.          107.        ]
 [ -41.9563683   -99.91273659   72.9563683   130.91273659]
 [ -57.02044839 -130.04089678   88.02044839  161.04089678]]
DEBUG:root:				} // END _scale_enum(anchor, scales)
DEBUG:root:anchors: [[ -74.          -28.          105.           59.        ]
 [ -97.39289449  -39.4365262   128.39289449   70.4365262 ]
 [-126.86609468  -53.84564629  157.86609468   84.84564629]
 [ -48.          -48.           79.           79.        ]
 [ -64.63494719  -64.63494719   95.63494719   95.63494719]
 [ -85.59366733  -85.59366733  116.59366733  116.59366733]
 [ -30.          -76.           61.          107.        ]
 [ -41.9563683   -99.91273659   72.9563683   130.91273659]
 [ -57.02044839 -130.04089678   88.02044839  161.04089678]]
DEBUG:root:							return torch.from_numpy(anchors)
DEBUG:root:						} // END _generate_anchors(base_size, scales, apect_ratios) END
DEBUG:root:					} // END generate_anchors(stride, sizes, aspect_ratios)
DEBUG:root:				generate_anchors(stride, sizes, aspect_ratios) { //BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Params:
DEBUG:root:						stride: 64
DEBUG:root:						sizes: (256.0, 322.53978877308754, 406.37466930385904)
DEBUG:root:						aspect_ratios: (0.5, 1.0, 2.0)
DEBUG:root:					return _generate_anchors(stride,
DEBUG:root:						     np.array(sizes, dtype=np.float) / stride,
DEBUG:root:						     np.array(aspect_ratios, dtype=np.float),)
DEBUG:root:						_generate_anchors(base_size, scales, aspect_ratios) { //BEGIN
DEBUG:root:						/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:							Params:
DEBUG:root:								base_size: 64
DEBUG:root:								scales: [4.         5.0396842  6.34960421]
DEBUG:root:								aspect_ratios: [0.5 1.  2. ]
DEBUG:root:anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
DEBUG:root:anchor: [ 0.  0. 63. 63.]
DEBUG:root:anchors = _ratio_enum(anchor, aspect_ratios)
DEBUG:root:				_ratio_enum(anchor, ratios) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [ 0.  0. 63. 63.]
DEBUG:root:						ratios: [0.5 1.  2. ]
DEBUG:root:				_whctrs(anchors) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [ 0.  0. 63. 63.]
DEBUG:root:						w = anchor[2] - anchor[0] + 1
DEBUG:root:						w: 64.0
DEBUG:root:						h = anchor[3] - anchor[1] + 1
DEBUG:root:						h: 64.0
DEBUG:root:						x_ctr = anchor[0] + 0.5 * (w - 1)
DEBUG:root:						x_ctr: 31.5
DEBUG:root:						y_ctr = anchor[1] + 0.5 * (h - 1)
DEBUG:root:						y_ctr: 31.5
DEBUG:root:						return w, h, x_ctr, y_ctr
DEBUG:root:					} // END _whctrs(anchors)
DEBUG:root:					w, h, x_ctr, y_ctr = _whctrs(anchor)
DEBUG:root:					w: 64.0, h: 64.0, x_ctr: 31.5, y_ctr: 31.5
DEBUG:root:					size = w * h
DEBUG:root:					size: 4096.0
DEBUG:root:					size_ratios = size / ratios
DEBUG:root:					size_ratios: [8192. 4096. 2048.]
DEBUG:root:					ws = np.round(np.sqrt(size_ratios))
DEBUG:root:					ws: [91. 64. 45.]
DEBUG:root:					hs = np.round(ws * ratios)
DEBUG:root:					hs: [46. 64. 90.]
DEBUG:root:			_mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:				Param:
DEBUG:root:					ws: [91. 64. 45.]
DEBUG:root:					hs: [46. 64. 90.]
DEBUG:root:					x_ctr: 31.5
DEBUG:root:					y_ctr: 31.5
DEBUG:root:				ws = ws[:, np.newaxis]
DEBUG:root:					ws: [[91.]
 [64.]
 [45.]]
DEBUG:root:				hs = hs[:, np.newaxis]
DEBUG:root:					hs: [[46.]
 [64.]
 [90.]]
DEBUG:root:				anchors = np.hstack(
DEBUG:root:				    (
DEBUG:root:				        x_ctr - 0.5 * (ws - 1),
DEBUG:root:				        y_ctr - 0.5 * (hs - 1),
DEBUG:root:				        x_ctr + 0.5 * (ws - 1),
DEBUG:root:				        y_ctr + 0.5 * (hs - 1),
DEBUG:root:				    )
DEBUG:root:				)
DEBUG:root:				anchors: [[-13.5   9.   76.5  54. ]
 [  0.    0.   63.   63. ]
 [  9.5 -13.   53.5  76. ]]
DEBUG:root:				return anchors
DEBUG:root:			} // END _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors: [[-13.5   9.   76.5  54. ]
 [  0.    0.   63.   63. ]
 [  9.5 -13.   53.5  76. ]]
DEBUG:root:					return anchors
DEBUG:root:				} // END _ratio_enum(anchor, ratios)
DEBUG:root:anchors: [[-13.5   9.   76.5  54. ]
 [  0.    0.   63.   63. ]
 [  9.5 -13.   53.5  76. ]]
DEBUG:root:anchors = np.vstack(
DEBUG:root:[_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
DEBUG:root:)
DEBUG:root:					_scale_enum(anchor, scales) { //BEGIN
DEBUG:root:					/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [-13.5   9.   76.5  54. ]
DEBUG:root:						scales: [4.         5.0396842  6.34960421]
DEBUG:root:					w, h, x_ctr, y_ctr = _whctrs(anchor)
DEBUG:root:				_whctrs(anchors) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [-13.5   9.   76.5  54. ]
DEBUG:root:						w = anchor[2] - anchor[0] + 1
DEBUG:root:						w: 91.0
DEBUG:root:						h = anchor[3] - anchor[1] + 1
DEBUG:root:						h: 46.0
DEBUG:root:						x_ctr = anchor[0] + 0.5 * (w - 1)
DEBUG:root:						x_ctr: 31.5
DEBUG:root:						y_ctr = anchor[1] + 0.5 * (h - 1)
DEBUG:root:						y_ctr: 31.5
DEBUG:root:						return w, h, x_ctr, y_ctr
DEBUG:root:					} // END _whctrs(anchors)
DEBUG:root:					w: 91.0, h: 46.0, x_ctr: 31.5, y_ctr: 31.5
DEBUG:root:					ws = w * scale
DEBUG:root:					ws: [364.         458.61126216 577.81398292]
DEBUG:root:					hs = h * scale
DEBUG:root:					hs: [184.         231.82547318 292.08179356]
DEBUG:root:					anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:			_mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:				Param:
DEBUG:root:					ws: [364.         458.61126216 577.81398292]
DEBUG:root:					hs: [184.         231.82547318 292.08179356]
DEBUG:root:					x_ctr: 31.5
DEBUG:root:					y_ctr: 31.5
DEBUG:root:				ws = ws[:, np.newaxis]
DEBUG:root:					ws: [[364.        ]
 [458.61126216]
 [577.81398292]]
DEBUG:root:				hs = hs[:, np.newaxis]
DEBUG:root:					hs: [[184.        ]
 [231.82547318]
 [292.08179356]]
DEBUG:root:				anchors = np.hstack(
DEBUG:root:				    (
DEBUG:root:				        x_ctr - 0.5 * (ws - 1),
DEBUG:root:				        y_ctr - 0.5 * (hs - 1),
DEBUG:root:				        x_ctr + 0.5 * (ws - 1),
DEBUG:root:				        y_ctr + 0.5 * (hs - 1),
DEBUG:root:				    )
DEBUG:root:				)
DEBUG:root:				anchors: [[-150.          -60.          213.          123.        ]
 [-197.30563108  -83.91273659  260.30563108  146.91273659]
 [-256.90699146 -114.04089678  319.90699146  177.04089678]]
DEBUG:root:				return anchors
DEBUG:root:			} // END _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors: [[-150.          -60.          213.          123.        ]
 [-197.30563108  -83.91273659  260.30563108  146.91273659]
 [-256.90699146 -114.04089678  319.90699146  177.04089678]]
DEBUG:root:				} // END _scale_enum(anchor, scales)
DEBUG:root:					_scale_enum(anchor, scales) { //BEGIN
DEBUG:root:					/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [ 0.  0. 63. 63.]
DEBUG:root:						scales: [4.         5.0396842  6.34960421]
DEBUG:root:					w, h, x_ctr, y_ctr = _whctrs(anchor)
DEBUG:root:				_whctrs(anchors) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [ 0.  0. 63. 63.]
DEBUG:root:						w = anchor[2] - anchor[0] + 1
DEBUG:root:						w: 64.0
DEBUG:root:						h = anchor[3] - anchor[1] + 1
DEBUG:root:						h: 64.0
DEBUG:root:						x_ctr = anchor[0] + 0.5 * (w - 1)
DEBUG:root:						x_ctr: 31.5
DEBUG:root:						y_ctr = anchor[1] + 0.5 * (h - 1)
DEBUG:root:						y_ctr: 31.5
DEBUG:root:						return w, h, x_ctr, y_ctr
DEBUG:root:					} // END _whctrs(anchors)
DEBUG:root:					w: 64.0, h: 64.0, x_ctr: 31.5, y_ctr: 31.5
DEBUG:root:					ws = w * scale
DEBUG:root:					ws: [256.         322.53978877 406.3746693 ]
DEBUG:root:					hs = h * scale
DEBUG:root:					hs: [256.         322.53978877 406.3746693 ]
DEBUG:root:					anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:			_mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:				Param:
DEBUG:root:					ws: [256.         322.53978877 406.3746693 ]
DEBUG:root:					hs: [256.         322.53978877 406.3746693 ]
DEBUG:root:					x_ctr: 31.5
DEBUG:root:					y_ctr: 31.5
DEBUG:root:				ws = ws[:, np.newaxis]
DEBUG:root:					ws: [[256.        ]
 [322.53978877]
 [406.3746693 ]]
DEBUG:root:				hs = hs[:, np.newaxis]
DEBUG:root:					hs: [[256.        ]
 [322.53978877]
 [406.3746693 ]]
DEBUG:root:				anchors = np.hstack(
DEBUG:root:				    (
DEBUG:root:				        x_ctr - 0.5 * (ws - 1),
DEBUG:root:				        y_ctr - 0.5 * (hs - 1),
DEBUG:root:				        x_ctr + 0.5 * (ws - 1),
DEBUG:root:				        y_ctr + 0.5 * (hs - 1),
DEBUG:root:				    )
DEBUG:root:				)
DEBUG:root:				anchors: [[ -96.          -96.          159.          159.        ]
 [-129.26989439 -129.26989439  192.26989439  192.26989439]
 [-171.18733465 -171.18733465  234.18733465  234.18733465]]
DEBUG:root:				return anchors
DEBUG:root:			} // END _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors: [[ -96.          -96.          159.          159.        ]
 [-129.26989439 -129.26989439  192.26989439  192.26989439]
 [-171.18733465 -171.18733465  234.18733465  234.18733465]]
DEBUG:root:				} // END _scale_enum(anchor, scales)
DEBUG:root:					_scale_enum(anchor, scales) { //BEGIN
DEBUG:root:					/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [  9.5 -13.   53.5  76. ]
DEBUG:root:						scales: [4.         5.0396842  6.34960421]
DEBUG:root:					w, h, x_ctr, y_ctr = _whctrs(anchor)
DEBUG:root:				_whctrs(anchors) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [  9.5 -13.   53.5  76. ]
DEBUG:root:						w = anchor[2] - anchor[0] + 1
DEBUG:root:						w: 45.0
DEBUG:root:						h = anchor[3] - anchor[1] + 1
DEBUG:root:						h: 90.0
DEBUG:root:						x_ctr = anchor[0] + 0.5 * (w - 1)
DEBUG:root:						x_ctr: 31.5
DEBUG:root:						y_ctr = anchor[1] + 0.5 * (h - 1)
DEBUG:root:						y_ctr: 31.5
DEBUG:root:						return w, h, x_ctr, y_ctr
DEBUG:root:					} // END _whctrs(anchors)
DEBUG:root:					w: 45.0, h: 90.0, x_ctr: 31.5, y_ctr: 31.5
DEBUG:root:					ws = w * scale
DEBUG:root:					ws: [180.         226.78578898 285.73218935]
DEBUG:root:					hs = h * scale
DEBUG:root:					hs: [360.         453.57157796 571.46437871]
DEBUG:root:					anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:			_mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:				Param:
DEBUG:root:					ws: [180.         226.78578898 285.73218935]
DEBUG:root:					hs: [360.         453.57157796 571.46437871]
DEBUG:root:					x_ctr: 31.5
DEBUG:root:					y_ctr: 31.5
DEBUG:root:				ws = ws[:, np.newaxis]
DEBUG:root:					ws: [[180.        ]
 [226.78578898]
 [285.73218935]]
DEBUG:root:				hs = hs[:, np.newaxis]
DEBUG:root:					hs: [[360.        ]
 [453.57157796]
 [571.46437871]]
DEBUG:root:				anchors = np.hstack(
DEBUG:root:				    (
DEBUG:root:				        x_ctr - 0.5 * (ws - 1),
DEBUG:root:				        y_ctr - 0.5 * (hs - 1),
DEBUG:root:				        x_ctr + 0.5 * (ws - 1),
DEBUG:root:				        y_ctr + 0.5 * (hs - 1),
DEBUG:root:				    )
DEBUG:root:				)
DEBUG:root:				anchors: [[ -58.         -148.          121.          211.        ]
 [ -81.39289449 -194.78578898  144.39289449  257.78578898]
 [-110.86609468 -253.73218935  173.86609468  316.73218935]]
DEBUG:root:				return anchors
DEBUG:root:			} // END _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors: [[ -58.         -148.          121.          211.        ]
 [ -81.39289449 -194.78578898  144.39289449  257.78578898]
 [-110.86609468 -253.73218935  173.86609468  316.73218935]]
DEBUG:root:				} // END _scale_enum(anchor, scales)
DEBUG:root:anchors: [[-150.          -60.          213.          123.        ]
 [-197.30563108  -83.91273659  260.30563108  146.91273659]
 [-256.90699146 -114.04089678  319.90699146  177.04089678]
 [ -96.          -96.          159.          159.        ]
 [-129.26989439 -129.26989439  192.26989439  192.26989439]
 [-171.18733465 -171.18733465  234.18733465  234.18733465]
 [ -58.         -148.          121.          211.        ]
 [ -81.39289449 -194.78578898  144.39289449  257.78578898]
 [-110.86609468 -253.73218935  173.86609468  316.73218935]]
DEBUG:root:							return torch.from_numpy(anchors)
DEBUG:root:						} // END _generate_anchors(base_size, scales, apect_ratios) END
DEBUG:root:					} // END generate_anchors(stride, sizes, aspect_ratios)
DEBUG:root:				generate_anchors(stride, sizes, aspect_ratios) { //BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Params:
DEBUG:root:						stride: 128
DEBUG:root:						sizes: (512.0, 645.0795775461751, 812.7493386077181)
DEBUG:root:						aspect_ratios: (0.5, 1.0, 2.0)
DEBUG:root:					return _generate_anchors(stride,
DEBUG:root:						     np.array(sizes, dtype=np.float) / stride,
DEBUG:root:						     np.array(aspect_ratios, dtype=np.float),)
DEBUG:root:						_generate_anchors(base_size, scales, aspect_ratios) { //BEGIN
DEBUG:root:						/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:							Params:
DEBUG:root:								base_size: 128
DEBUG:root:								scales: [4.         5.0396842  6.34960421]
DEBUG:root:								aspect_ratios: [0.5 1.  2. ]
DEBUG:root:anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
DEBUG:root:anchor: [  0.   0. 127. 127.]
DEBUG:root:anchors = _ratio_enum(anchor, aspect_ratios)
DEBUG:root:				_ratio_enum(anchor, ratios) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [  0.   0. 127. 127.]
DEBUG:root:						ratios: [0.5 1.  2. ]
DEBUG:root:				_whctrs(anchors) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [  0.   0. 127. 127.]
DEBUG:root:						w = anchor[2] - anchor[0] + 1
DEBUG:root:						w: 128.0
DEBUG:root:						h = anchor[3] - anchor[1] + 1
DEBUG:root:						h: 128.0
DEBUG:root:						x_ctr = anchor[0] + 0.5 * (w - 1)
DEBUG:root:						x_ctr: 63.5
DEBUG:root:						y_ctr = anchor[1] + 0.5 * (h - 1)
DEBUG:root:						y_ctr: 63.5
DEBUG:root:						return w, h, x_ctr, y_ctr
DEBUG:root:					} // END _whctrs(anchors)
DEBUG:root:					w, h, x_ctr, y_ctr = _whctrs(anchor)
DEBUG:root:					w: 128.0, h: 128.0, x_ctr: 63.5, y_ctr: 63.5
DEBUG:root:					size = w * h
DEBUG:root:					size: 16384.0
DEBUG:root:					size_ratios = size / ratios
DEBUG:root:					size_ratios: [32768. 16384.  8192.]
DEBUG:root:					ws = np.round(np.sqrt(size_ratios))
DEBUG:root:					ws: [181. 128.  91.]
DEBUG:root:					hs = np.round(ws * ratios)
DEBUG:root:					hs: [ 90. 128. 182.]
DEBUG:root:			_mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:				Param:
DEBUG:root:					ws: [181. 128.  91.]
DEBUG:root:					hs: [ 90. 128. 182.]
DEBUG:root:					x_ctr: 63.5
DEBUG:root:					y_ctr: 63.5
DEBUG:root:				ws = ws[:, np.newaxis]
DEBUG:root:					ws: [[181.]
 [128.]
 [ 91.]]
DEBUG:root:				hs = hs[:, np.newaxis]
DEBUG:root:					hs: [[ 90.]
 [128.]
 [182.]]
DEBUG:root:				anchors = np.hstack(
DEBUG:root:				    (
DEBUG:root:				        x_ctr - 0.5 * (ws - 1),
DEBUG:root:				        y_ctr - 0.5 * (hs - 1),
DEBUG:root:				        x_ctr + 0.5 * (ws - 1),
DEBUG:root:				        y_ctr + 0.5 * (hs - 1),
DEBUG:root:				    )
DEBUG:root:				)
DEBUG:root:				anchors: [[-26.5  19.  153.5 108. ]
 [  0.    0.  127.  127. ]
 [ 18.5 -27.  108.5 154. ]]
DEBUG:root:				return anchors
DEBUG:root:			} // END _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors: [[-26.5  19.  153.5 108. ]
 [  0.    0.  127.  127. ]
 [ 18.5 -27.  108.5 154. ]]
DEBUG:root:					return anchors
DEBUG:root:				} // END _ratio_enum(anchor, ratios)
DEBUG:root:anchors: [[-26.5  19.  153.5 108. ]
 [  0.    0.  127.  127. ]
 [ 18.5 -27.  108.5 154. ]]
DEBUG:root:anchors = np.vstack(
DEBUG:root:[_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
DEBUG:root:)
DEBUG:root:					_scale_enum(anchor, scales) { //BEGIN
DEBUG:root:					/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [-26.5  19.  153.5 108. ]
DEBUG:root:						scales: [4.         5.0396842  6.34960421]
DEBUG:root:					w, h, x_ctr, y_ctr = _whctrs(anchor)
DEBUG:root:				_whctrs(anchors) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [-26.5  19.  153.5 108. ]
DEBUG:root:						w = anchor[2] - anchor[0] + 1
DEBUG:root:						w: 181.0
DEBUG:root:						h = anchor[3] - anchor[1] + 1
DEBUG:root:						h: 90.0
DEBUG:root:						x_ctr = anchor[0] + 0.5 * (w - 1)
DEBUG:root:						x_ctr: 63.5
DEBUG:root:						y_ctr = anchor[1] + 0.5 * (h - 1)
DEBUG:root:						y_ctr: 63.5
DEBUG:root:						return w, h, x_ctr, y_ctr
DEBUG:root:					} // END _whctrs(anchors)
DEBUG:root:					w: 181.0, h: 90.0, x_ctr: 63.5, y_ctr: 63.5
DEBUG:root:					ws = w * scale
DEBUG:root:					ws: [ 724.          912.18284012 1149.27836162]
DEBUG:root:					hs = h * scale
DEBUG:root:					hs: [360.         453.57157796 571.46437871]
DEBUG:root:					anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:			_mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:				Param:
DEBUG:root:					ws: [ 724.          912.18284012 1149.27836162]
DEBUG:root:					hs: [360.         453.57157796 571.46437871]
DEBUG:root:					x_ctr: 63.5
DEBUG:root:					y_ctr: 63.5
DEBUG:root:				ws = ws[:, np.newaxis]
DEBUG:root:					ws: [[ 724.        ]
 [ 912.18284012]
 [1149.27836162]]
DEBUG:root:				hs = hs[:, np.newaxis]
DEBUG:root:					hs: [[360.        ]
 [453.57157796]
 [571.46437871]]
DEBUG:root:				anchors = np.hstack(
DEBUG:root:				    (
DEBUG:root:				        x_ctr - 0.5 * (ws - 1),
DEBUG:root:				        y_ctr - 0.5 * (hs - 1),
DEBUG:root:				        x_ctr + 0.5 * (ws - 1),
DEBUG:root:				        y_ctr + 0.5 * (hs - 1),
DEBUG:root:				    )
DEBUG:root:				)
DEBUG:root:				anchors: [[-298.         -116.          425.          243.        ]
 [-392.09142006 -162.78578898  519.09142006  289.78578898]
 [-510.63918081 -221.73218935  637.63918081  348.73218935]]
DEBUG:root:				return anchors
DEBUG:root:			} // END _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors: [[-298.         -116.          425.          243.        ]
 [-392.09142006 -162.78578898  519.09142006  289.78578898]
 [-510.63918081 -221.73218935  637.63918081  348.73218935]]
DEBUG:root:				} // END _scale_enum(anchor, scales)
DEBUG:root:					_scale_enum(anchor, scales) { //BEGIN
DEBUG:root:					/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [  0.   0. 127. 127.]
DEBUG:root:						scales: [4.         5.0396842  6.34960421]
DEBUG:root:					w, h, x_ctr, y_ctr = _whctrs(anchor)
DEBUG:root:				_whctrs(anchors) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [  0.   0. 127. 127.]
DEBUG:root:						w = anchor[2] - anchor[0] + 1
DEBUG:root:						w: 128.0
DEBUG:root:						h = anchor[3] - anchor[1] + 1
DEBUG:root:						h: 128.0
DEBUG:root:						x_ctr = anchor[0] + 0.5 * (w - 1)
DEBUG:root:						x_ctr: 63.5
DEBUG:root:						y_ctr = anchor[1] + 0.5 * (h - 1)
DEBUG:root:						y_ctr: 63.5
DEBUG:root:						return w, h, x_ctr, y_ctr
DEBUG:root:					} // END _whctrs(anchors)
DEBUG:root:					w: 128.0, h: 128.0, x_ctr: 63.5, y_ctr: 63.5
DEBUG:root:					ws = w * scale
DEBUG:root:					ws: [512.         645.07957755 812.74933861]
DEBUG:root:					hs = h * scale
DEBUG:root:					hs: [512.         645.07957755 812.74933861]
DEBUG:root:					anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:			_mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:				Param:
DEBUG:root:					ws: [512.         645.07957755 812.74933861]
DEBUG:root:					hs: [512.         645.07957755 812.74933861]
DEBUG:root:					x_ctr: 63.5
DEBUG:root:					y_ctr: 63.5
DEBUG:root:				ws = ws[:, np.newaxis]
DEBUG:root:					ws: [[512.        ]
 [645.07957755]
 [812.74933861]]
DEBUG:root:				hs = hs[:, np.newaxis]
DEBUG:root:					hs: [[512.        ]
 [645.07957755]
 [812.74933861]]
DEBUG:root:				anchors = np.hstack(
DEBUG:root:				    (
DEBUG:root:				        x_ctr - 0.5 * (ws - 1),
DEBUG:root:				        y_ctr - 0.5 * (hs - 1),
DEBUG:root:				        x_ctr + 0.5 * (ws - 1),
DEBUG:root:				        y_ctr + 0.5 * (hs - 1),
DEBUG:root:				    )
DEBUG:root:				)
DEBUG:root:				anchors: [[-192.         -192.          319.          319.        ]
 [-258.53978877 -258.53978877  385.53978877  385.53978877]
 [-342.3746693  -342.3746693   469.3746693   469.3746693 ]]
DEBUG:root:				return anchors
DEBUG:root:			} // END _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors: [[-192.         -192.          319.          319.        ]
 [-258.53978877 -258.53978877  385.53978877  385.53978877]
 [-342.3746693  -342.3746693   469.3746693   469.3746693 ]]
DEBUG:root:				} // END _scale_enum(anchor, scales)
DEBUG:root:					_scale_enum(anchor, scales) { //BEGIN
DEBUG:root:					/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [ 18.5 -27.  108.5 154. ]
DEBUG:root:						scales: [4.         5.0396842  6.34960421]
DEBUG:root:					w, h, x_ctr, y_ctr = _whctrs(anchor)
DEBUG:root:				_whctrs(anchors) { //BEGIN
DEBUG:root:				/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:					Param:
DEBUG:root:						anchor: [ 18.5 -27.  108.5 154. ]
DEBUG:root:						w = anchor[2] - anchor[0] + 1
DEBUG:root:						w: 91.0
DEBUG:root:						h = anchor[3] - anchor[1] + 1
DEBUG:root:						h: 182.0
DEBUG:root:						x_ctr = anchor[0] + 0.5 * (w - 1)
DEBUG:root:						x_ctr: 63.5
DEBUG:root:						y_ctr = anchor[1] + 0.5 * (h - 1)
DEBUG:root:						y_ctr: 63.5
DEBUG:root:						return w, h, x_ctr, y_ctr
DEBUG:root:					} // END _whctrs(anchors)
DEBUG:root:					w: 91.0, h: 182.0, x_ctr: 63.5, y_ctr: 63.5
DEBUG:root:					ws = w * scale
DEBUG:root:					ws: [364.         458.61126216 577.81398292]
DEBUG:root:					hs = h * scale
DEBUG:root:					hs: [ 728.          917.22252432 1155.62796583]
DEBUG:root:					anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:			_mkanchors(ws, hs, x_ctr, y_ctr) { // BEGIN
DEBUG:root:			/home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:				Param:
DEBUG:root:					ws: [364.         458.61126216 577.81398292]
DEBUG:root:					hs: [ 728.          917.22252432 1155.62796583]
DEBUG:root:					x_ctr: 63.5
DEBUG:root:					y_ctr: 63.5
DEBUG:root:				ws = ws[:, np.newaxis]
DEBUG:root:					ws: [[364.        ]
 [458.61126216]
 [577.81398292]]
DEBUG:root:				hs = hs[:, np.newaxis]
DEBUG:root:					hs: [[ 728.        ]
 [ 917.22252432]
 [1155.62796583]]
DEBUG:root:				anchors = np.hstack(
DEBUG:root:				    (
DEBUG:root:				        x_ctr - 0.5 * (ws - 1),
DEBUG:root:				        y_ctr - 0.5 * (hs - 1),
DEBUG:root:				        x_ctr + 0.5 * (ws - 1),
DEBUG:root:				        y_ctr + 0.5 * (hs - 1),
DEBUG:root:				    )
DEBUG:root:				)
DEBUG:root:				anchors: [[-118.         -300.          245.          427.        ]
 [-165.30563108 -394.61126216  292.30563108  521.61126216]
 [-224.90699146 -513.81398292  351.90699146  640.81398292]]
DEBUG:root:				return anchors
DEBUG:root:			} // END _mkanchors(ws, hs, x_ctr, y_ctr)
DEBUG:root:					anchors: [[-118.         -300.          245.          427.        ]
 [-165.30563108 -394.61126216  292.30563108  521.61126216]
 [-224.90699146 -513.81398292  351.90699146  640.81398292]]
DEBUG:root:				} // END _scale_enum(anchor, scales)
DEBUG:root:anchors: [[-298.         -116.          425.          243.        ]
 [-392.09142006 -162.78578898  519.09142006  289.78578898]
 [-510.63918081 -221.73218935  637.63918081  348.73218935]
 [-192.         -192.          319.          319.        ]
 [-258.53978877 -258.53978877  385.53978877  385.53978877]
 [-342.3746693  -342.3746693   469.3746693   469.3746693 ]
 [-118.         -300.          245.          427.        ]
 [-165.30563108 -394.61126216  292.30563108  521.61126216]
 [-224.90699146 -513.81398292  351.90699146  640.81398292]]
DEBUG:root:							return torch.from_numpy(anchors)
DEBUG:root:						} // END _generate_anchors(base_size, scales, apect_ratios) END
DEBUG:root:					} // END generate_anchors(stride, sizes, aspect_ratios)
DEBUG:root:cell_anchors: [tensor([[-18.0000,  -8.0000,  25.0000,  15.0000],
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
DEBUG:root:BufferList.__init__(slef, buffers=None) { // BEGIN
DEBUG:root:// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:Params
DEBUG:root:	len(buffers): 5
DEBUG:root:super(BufferList, self).__init__()
DEBUG:root:if buffers is not None:
DEBUG:root:BufferList.extend(self, buffers) { // BEGIN
DEBUG:root:// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:Params
DEBUG:root:	len(buffers): 5
DEBUG:root:	offset = len(self)
DEBUG:root:	len(buffers): 5
DEBUG:root:for i, buffer in enumerate(buffers) { // BEGIN
DEBUG:root:i: 0
DEBUG:root:buffer.shape: torch.Size([9, 4])
DEBUG:root:self.register_buffer(str(offset + i), buffer)
DEBUG:root:i: 1
DEBUG:root:buffer.shape: torch.Size([9, 4])
DEBUG:root:self.register_buffer(str(offset + i), buffer)
DEBUG:root:i: 2
DEBUG:root:buffer.shape: torch.Size([9, 4])
DEBUG:root:self.register_buffer(str(offset + i), buffer)
DEBUG:root:i: 3
DEBUG:root:buffer.shape: torch.Size([9, 4])
DEBUG:root:self.register_buffer(str(offset + i), buffer)
DEBUG:root:i: 4
DEBUG:root:buffer.shape: torch.Size([9, 4])
DEBUG:root:self.register_buffer(str(offset + i), buffer)
DEBUG:root:} // END for i, buffer in enumerate(buffers)
DEBUG:root:self: BufferList()
DEBUG:root:return self
DEBUG:root:} // END BufferList.extend(self, buffers)
DEBUG:root:self.extend(buffers)
DEBUG:root:	len(buffers): 5
DEBUG:root:} // END BufferList.__init__(slef, buffers=None)
DEBUG:root:	self.strides: (8, 16, 32, 64, 128)
DEBUG:root:	self.cell_anchors: BufferList()
DEBUG:root:	self.straddle_thresh: -1
DEBUG:root:		} // END AnchorGenerator.__init__(sizes, aspect_ratios, anchor_strides, straddle_thresh)
DEBUG:root:		anchor_generator = AnchorGenerator(
  (cell_anchors): BufferList()
)
DEBUG:root:
		return anchor_generator
DEBUG:root:		} // make_anchor_generator_retinanet(config) END

DEBUG:root:	anchor_generator: AnchorGenerator(
  (cell_anchors): BufferList()
)
DEBUG:root:	head = RetinaNetHead(cfg, in_channels=1024)
DEBUG:root:

			RetinaNetHead.__init__(cfg, in_channels) { //BEGIN
DEBUG:root:				// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/retinanet/retinanet.py
DEBUG:root:				Params:
DEBUG:root:					cfg:
DEBUG:root:					in_channles: 1024:
DEBUG:root:				num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
DEBUG:root:					num_classes: 1
DEBUG:root:				num_anchors = len(cfg.MODEL.RETINANET.ASPECT_RATIOS) \
DEBUG:root:				                * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE
DEBUG:root:					cfg.MODEL.RETINANET.ASPECT_RATIOS: (0.5, 1.0, 2.0)
DEBUG:root:					cfg.MODEL.RETINANET.SCALES_PER_OCTAVE: 3
DEBUG:root:					num_anchors: 9
DEBUG:root:

} // END RetinaNetHead._init__(cfg, in_channels)
DEBUG:root:	head: RetinaNetHead(
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
DEBUG:root:	box_coder = BoxCoder(weights=(10., 10., 5., 5.))
DEBUG:root:	box_coder: <maskrcnn_benchmark.modeling.box_coder.BoxCoder object at 0x7f492f589978>
DEBUG:root:	box_selector_test = make_retinanet_postprocessor(cfg, box_coder)
DEBUG:root:RPNPostProcessing.__init__() { //BEGIN
DEBUG:root:	// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/inference.py
DEBUG:root:} // END RPNPostProcessing.__init__()
DEBUG:root:	box_selector_test: RetinaNetPostProcessor()
DEBUG:root:	self.anchor_generator = anchor_generator
DEBUG:root:	self.head = head
DEBUG:root:	self.box_selector_test = box_selector_test
DEBUG:root:

} // RetinaNetModule.__init__(self, cfg, in_channels) END
DEBUG:root:	} // END build_retinanet(cfg, in_channels)
DEBUG:root:	self.rpn = build_rpn(cfg, self.backbone.out_channels) // RETURNED
DEBUG:root:	self.rpn: RetinaNetModule(
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
DEBUG:root:} END GeneralizedRCNN.__init__(self, cfg)
DEBUG:root:	self.model.eval()
DEBUG:root:	checkpointer = DetectronCheckpointer(cfg, self.model, save_dir='/dev/null')
DEBUG:root:	_ = checkpointer.load(weight)
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
DEBUG:root:	self.transforms = build_transforms(self.cfg, self.is_recognition)
DEBUG:root:} // END DetectionDemo.__init__
DEBUG:root:compute_prediction(self, image) { // BEGIN
DEBUG:root:	// defined in detection_model_debug.py

DEBUG:root:	Params:
DEBUG:root:		image: H, W=(438,512)
DEBUG:root:
		transforms.py Compose class __call__  ====== BEGIN
DEBUG:root:
		for t in self.transforms:
DEBUG:root:			image = <maskrcnn_benchmark.data.transforms.transforms.Resize object at 0x7f4932d0c080>(image)
DEBUG:root:			image = <maskrcnn_benchmark.data.transforms.transforms.ToTensor object at 0x7f4932d0c128>(image)
DEBUG:root:			image = <maskrcnn_benchmark.data.transforms.transforms.Normalize object at 0x7f4932d0c0b8>(image)
DEBUG:root:
		return image
DEBUG:root:		transforms.py Compose class __call__  ====== END
DEBUG:root:	image_tensor = self.transforms(image)
DEBUG:root:
	image_tensor.shape: torch.Size([3, 480, 561])
DEBUG:root:
	padding images for 32 divisible size on width and height
DEBUG:root:	image_list = to_image_list(image_tensor, 32).to(self.device)
DEBUG:root:
	to_image_list(tensors, size_divisible=32) ====== BEGIN
DEBUG:root:		type(batched_imgs): <class 'torch.Tensor'>
DEBUG:root:		batched_imgs.shape: torch.Size([1, 3, 480, 576])
DEBUG:root:		image_sizes: [torch.Size([480, 561])]
DEBUG:root:		return ImageList(batched_imgs, image_sizes)
DEBUG:root:	to_image_list(tensors, size_divisible=32) ====== END

DEBUG:root:	image_list.image_sizes: [torch.Size([480, 561])]
DEBUG:root:	image_list.tensors.shape: torch.Size([1, 3, 480, 576])
DEBUG:root:	pred = self.model(image_list)
DEBUG:root:

	GeneralizedRCNN.forward(self, images, targets=None) { //BEGIN
DEBUG:root:	// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/detector/generalized_rcnn.py
DEBUG:root:		Params:
DEBUG:root:			images:
DEBUG:root:				type(images): <class 'maskrcnn_benchmark.structures.image_list.ImageList'>
DEBUG:root:			targets: None
DEBUG:root:	if self.training: False: 
DEBUG:root:	images = to_image_list(images)
DEBUG:root:
	to_image_list(tensors, size_divisible=0) ====== BEGIN
DEBUG:root:		if isinstance(tensors, ImageList):
DEBUG:root:		return tensors
DEBUG:root:	to_image_list(tensors, size_divisible=0) ====== END

DEBUG:root:	images.image_sizes: [torch.Size([480, 561])]
DEBUG:root:	images.tensors.shape: torch.Size([1, 3, 480, 576])
DEBUG:root:	model.backbone.forward(images.tensors) BEFORE
DEBUG:root:
	Resnet.forward(self, x) { //BEGIN
DEBUG:root:		Param
DEBUG:root:			x.shape=torch.Size([1, 3, 480, 576])

DEBUG:root:		x = self.stem(x)
DEBUG:root:		x.shape: torch.Size([1, 64, 120, 144])
DEBUG:root:		stem output of shape (1, 64, 120, 144) saved into ./npy_save/stem_output.npy


DEBUG:root:		for stage_name in self.stages:
DEBUG:root:			stage_name: layer1
DEBUG:root:				output shape of layer1: torch.Size([1, 256, 120, 144])
DEBUG:root:			outputs.append(x) stage_name: layer1
DEBUG:root:			x.shape: torch.Size([1, 256, 120, 144])
DEBUG:root:		layer1 output of shape (1, 256, 120, 144) saved into ./npy_save/layer1_output.npy


DEBUG:root:			stage_name: layer2
DEBUG:root:				output shape of layer2: torch.Size([1, 512, 60, 72])
DEBUG:root:			outputs.append(x) stage_name: layer2
DEBUG:root:			x.shape: torch.Size([1, 512, 60, 72])
DEBUG:root:		layer2 output of shape (1, 512, 60, 72) saved into ./npy_save/layer2_output.npy


DEBUG:root:			stage_name: layer3
DEBUG:root:				output shape of layer3: torch.Size([1, 1024, 30, 36])
DEBUG:root:			outputs.append(x) stage_name: layer3
DEBUG:root:			x.shape: torch.Size([1, 1024, 30, 36])
DEBUG:root:		layer3 output of shape (1, 1024, 30, 36) saved into ./npy_save/layer3_output.npy


DEBUG:root:			stage_name: layer4
DEBUG:root:				output shape of layer4: torch.Size([1, 2048, 15, 18])
DEBUG:root:			outputs.append(x) stage_name: layer4
DEBUG:root:			x.shape: torch.Size([1, 2048, 15, 18])
DEBUG:root:		layer4 output of shape (1, 2048, 15, 18) saved into ./npy_save/layer4_output.npy


DEBUG:root:
		ResNet::forward return value
DEBUG:root:			outputs[0]: torch.Size([1, 256, 120, 144])
DEBUG:root:			outputs[1]: torch.Size([1, 512, 60, 72])
DEBUG:root:			outputs[2]: torch.Size([1, 1024, 30, 36])
DEBUG:root:			outputs[3]: torch.Size([1, 2048, 15, 18])
DEBUG:root:
		return outputs
DEBUG:root:
	} // END Resnet.forward()
DEBUG:root:

	FPN.forward(self,x) { // BEGIN
DEBUG:root:		Param: x  = [C2, C3, C4, C5] 
DEBUG:root:			len(x) = 4
DEBUG:root:			C[1].shape : torch.Size([1, 256, 120, 144])
DEBUG:root:			C[2].shape : torch.Size([1, 512, 60, 72])
DEBUG:root:			C[3].shape : torch.Size([1, 1024, 30, 36])
DEBUG:root:			C[4].shape : torch.Size([1, 2048, 15, 18])
DEBUG:root:
			x[-1].shape = torch.Size([1, 2048, 15, 18])
DEBUG:root:
			===========================================================================
DEBUG:root:			FPN block info
DEBUG:root:			self.inner_blocks: ['fpn_inner2', 'fpn_inner3', 'fpn_inner4'])
DEBUG:root:			self.layer_blocks: ['fpn_layer2', 'fpn_layer3', 'fpn_layer4'])
DEBUG:root:			===========================================================================
DEBUG:root:
	last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
DEBUG:root:		self.innerblocks[-1] = Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))
DEBUG:root:		x[-1].shape = torch.Size([1, 2048, 15, 18])
DEBUG:root:		last_inner.shape = torch.Size([1, 1024, 15, 18])

DEBUG:root:	fpn_inner4 output of shape (1, 1024, 15, 18) saved into ./npy_save/fpn_inner4_output.npy


DEBUG:root:
	results.append(fpn_layer4(last_inner))
DEBUG:root:		self.layer_blocks[-1]: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
DEBUG:root:		results[0].shape: torch.Size([1, 1024, 15, 18])

DEBUG:root:		fpn_layer4 output of shape (1, 1024, 15, 18) saved into ./npy_save/fpn_layer4_output.npy


DEBUG:root:	for feature, inner_block, layer_block
			in zip[(x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]):

DEBUG:root:		====================================
DEBUG:root:		iteration 0 summary
DEBUG:root:		====================================
DEBUG:root:		feature.shape: torch.Size([1, 1024, 30, 36])
DEBUG:root:		inner_block: fpn_inner3 ==> Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
DEBUG:root:		layer_block: fpn_layer3 ==> Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
DEBUG:root:		last_inner.shape: torch.Size([1, 1024, 15, 18])
DEBUG:root:		====================================

DEBUG:root:		--------------------------------------------------
DEBUG:root:		0.1 Upsample : replace with Decovolution in caffe
DEBUG:root:		layer name in caffe: fpn_inner3_upsample = Deconvolution(last_inner)
DEBUG:root:		--------------------------------------------------
DEBUG:root:		inner_top_down = F.interpolate(last_inner, scale_factor=2, mode='nearest'
DEBUG:root:		last_inner.shape: torch.Size([1, 1024, 15, 18])
DEBUG:root:		inner_top_down.shape : torch.Size([1, 1024, 30, 36])
DEBUG:root:		--------------------------------------------------

DEBUG:root:		inner_top_down of shape (1, 1024, 30, 36) saved into ./npy_save/inner_top_down_forfpn_inner3.npy


DEBUG:root:		--------------------------------------------------
DEBUG:root:		0.2 inner_lateral = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))(feature)
DEBUG:root:		layer name in caffe: fpn_inner3_lateral=Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))(feature)
DEBUG:root:		--------------------------------------------------
DEBUG:root:			inner_block: fpn_inner3 ==> Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
DEBUG:root:			input: feature.shape: torch.Size([1, 1024, 30, 36])
DEBUG:root:			output: inner_lateral.shape: torch.Size([1, 1024, 30, 36])

DEBUG:root:		--------------------------------------------------

DEBUG:root:		fpn_inner3 output of shape (1, 1024, 30, 36) saved into ./npy_save/fpn_inner3_output.npy


DEBUG:root:		--------------------------------------------------
DEBUG:root:		0.3 Elementwise Addition: replaced with eltwise in caffe
DEBUG:root:		layer in caffe: eltwise_3 = eltwise(fpn_inner3_lateral, fpn_inner3_upsample )
DEBUG:root:		--------------------------------------------------
DEBUG:root:		last_inner = inner_lateral + inner_top_down
DEBUG:root:			inner_lateral.shape: torch.Size([1, 1024, 30, 36])
DEBUG:root:			inner_top_down.shape: torch.Size([1, 1024, 30, 36])
DEBUG:root:			last_inner.shape : torch.Size([1, 1024, 30, 36])
DEBUG:root:		--------------------------------------------------

DEBUG:root:		superimposing result of fpn_inner3 output plus inner topdown of shape (1, 1024, 30, 36) saved into ./npy_save/fpn_inner3_ouptut_plus_inner_topdown.npy


DEBUG:root:		--------------------------------------------------
DEBUG:root:		0.4 results.insert(0, Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(last_inner)
DEBUG:root:		layer in caffe: fpn_layer3 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(eltwise_3)
DEBUG:root:		--------------------------------------------------
DEBUG:root:			layer_block: fpn_layer3 ==> Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
DEBUG:root:			input: last_inner.shape = torch.Size([1, 1024, 30, 36])
DEBUG:root:		--------------------------------------------------

DEBUG:root:		fpn_layer3 output of shape (1, 1024, 30, 36) saved into ./npy_save/fpn_layer3_ouptut.npy


DEBUG:root:		--------------------------------------------------
DEBUG:root:		results after iteration 0
DEBUG:root:		--------------------------------------------------
DEBUG:root:			results[0].shape: torch.Size([1, 1024, 30, 36])
DEBUG:root:			results[1].shape: torch.Size([1, 1024, 15, 18])
DEBUG:root:		--------------------------------------------------

DEBUG:root:		====================================
DEBUG:root:		iteration 1 summary
DEBUG:root:		====================================
DEBUG:root:		feature.shape: torch.Size([1, 512, 60, 72])
DEBUG:root:		inner_block: fpn_inner2 ==> Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
DEBUG:root:		layer_block: fpn_layer2 ==> Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
DEBUG:root:		last_inner.shape: torch.Size([1, 1024, 30, 36])
DEBUG:root:		====================================

DEBUG:root:		--------------------------------------------------
DEBUG:root:		1.1 Upsample : replace with Decovolution in caffe
DEBUG:root:		layer name in caffe: fpn_inner2_upsample = Deconvolution(last_inner)
DEBUG:root:		--------------------------------------------------
DEBUG:root:		inner_top_down = F.interpolate(last_inner, scale_factor=2, mode='nearest'
DEBUG:root:		last_inner.shape: torch.Size([1, 1024, 30, 36])
DEBUG:root:		inner_top_down.shape : torch.Size([1, 1024, 60, 72])
DEBUG:root:		--------------------------------------------------

DEBUG:root:		inner_top_down of shape (1, 1024, 60, 72) saved into ./npy_save/inner_top_down_forfpn_inner2.npy


DEBUG:root:		--------------------------------------------------
DEBUG:root:		1.2 inner_lateral = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))(feature)
DEBUG:root:		layer name in caffe: fpn_inner2_lateral=Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))(feature)
DEBUG:root:		--------------------------------------------------
DEBUG:root:			inner_block: fpn_inner2 ==> Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
DEBUG:root:			input: feature.shape: torch.Size([1, 512, 60, 72])
DEBUG:root:			output: inner_lateral.shape: torch.Size([1, 1024, 60, 72])

DEBUG:root:		--------------------------------------------------

DEBUG:root:		fpn_inner2 output of shape (1, 1024, 60, 72) saved into ./npy_save/fpn_inner2_output.npy


DEBUG:root:		--------------------------------------------------
DEBUG:root:		1.3 Elementwise Addition: replaced with eltwise in caffe
DEBUG:root:		layer in caffe: eltwise_2 = eltwise(fpn_inner2_lateral, fpn_inner2_upsample )
DEBUG:root:		--------------------------------------------------
DEBUG:root:		last_inner = inner_lateral + inner_top_down
DEBUG:root:			inner_lateral.shape: torch.Size([1, 1024, 60, 72])
DEBUG:root:			inner_top_down.shape: torch.Size([1, 1024, 60, 72])
DEBUG:root:			last_inner.shape : torch.Size([1, 1024, 60, 72])
DEBUG:root:		--------------------------------------------------

DEBUG:root:		superimposing result of fpn_inner2 output plus inner topdown of shape (1, 1024, 60, 72) saved into ./npy_save/fpn_inner2_ouptut_plus_inner_topdown.npy


DEBUG:root:		--------------------------------------------------
DEBUG:root:		1.4 results.insert(0, Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(last_inner)
DEBUG:root:		layer in caffe: fpn_layer2 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(eltwise_2)
DEBUG:root:		--------------------------------------------------
DEBUG:root:			layer_block: fpn_layer2 ==> Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
DEBUG:root:			input: last_inner.shape = torch.Size([1, 1024, 60, 72])
DEBUG:root:		--------------------------------------------------

DEBUG:root:		fpn_layer2 output of shape (1, 1024, 60, 72) saved into ./npy_save/fpn_layer2_ouptut.npy


DEBUG:root:		--------------------------------------------------
DEBUG:root:		results after iteration 1
DEBUG:root:		--------------------------------------------------
DEBUG:root:			results[0].shape: torch.Size([1, 1024, 60, 72])
DEBUG:root:			results[1].shape: torch.Size([1, 1024, 30, 36])
DEBUG:root:			results[2].shape: torch.Size([1, 1024, 15, 18])
DEBUG:root:		--------------------------------------------------

DEBUG:root:	for loop END

DEBUG:root:
	if isinstance(self.top_blocks, LastLevelP6P7):
DEBUG:root:			self.top_blocks: LastLevelP6P7(
  (p6): Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (p7): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
)
DEBUG:root:			len(x): 4
DEBUG:root:			x[0].shape : torch.Size([1, 256, 120, 144])
DEBUG:root:			x[1].shape : torch.Size([1, 512, 60, 72])
DEBUG:root:			x[2].shape : torch.Size([1, 1024, 30, 36])
DEBUG:root:			x[3].shape : torch.Size([1, 2048, 15, 18])
DEBUG:root:			x[-1].shape: torch.Size([1, 2048, 15, 18])


DEBUG:root:			len(results): 3
DEBUG:root:			results[0].shape : torch.Size([1, 1024, 60, 72])
DEBUG:root:			results[1].shape : torch.Size([1, 1024, 30, 36])
DEBUG:root:			results[2].shape : torch.Size([1, 1024, 15, 18])
DEBUG:root:			results[-1].shape: torch.Size([1, 1024, 15, 18])


DEBUG:root:
			LastLevelP6P7.forward(self, c5, p5) { // BEGIN 
DEBUG:root:
				Param:
DEBUG:root:					c5.shape: torch.Size([1, 2048, 15, 18])
DEBUG:root:					p5.shape: torch.Size([1, 1024, 15, 18])

DEBUG:root:				if (self.use_P5 == False)
DEBUG:root:					x=c5
DEBUG:root:				x.shape = torch.Size([1, 2048, 15, 18])
DEBUG:root:				p6 = self.p6(x)
DEBUG:root:					self.p6: Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
DEBUG:root:					p6.shape: torch.Size([1, 1024, 8, 9])

DEBUG:root:				LastLevelP6P7::forward() self.p6 ==> Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) output of shape (1, 1024, 8, 9) saved into ./npy_save/P6.npy


DEBUG:root:				p7 = self.p7(F.relu(p6))
DEBUG:root:					self.p7: Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
DEBUG:root:					p7.shape: torch.Size([1, 1024, 4, 5])

DEBUG:root:			LastLevelP6P7::forward() self.p7 Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))(F.relu(p6)) output of shape (1, 1024, 4, 5) saved into ./npy_save/P7.npy


DEBUG:root:				returns [p6, p7]
DEBUG:root:			} // END LastLevelP6P7.forward(self, c5, p5)


DEBUG:root:		last_result = self.top_blocks(x[-1], results[-1])
DEBUG:root:			x[-1] => c5, results[-1])=> p5
DEBUG:root:			len(last_results):2
DEBUG:root:			last_results[0].shape : torch.Size([1, 1024, 8, 9])
DEBUG:root:			last_results[1].shape : torch.Size([1, 1024, 4, 5])
DEBUG:root:		results.extend(last_results)
DEBUG:root:			len(results): 5
DEBUG:root:			results[0].shape : torch.Size([1, 1024, 60, 72])
DEBUG:root:			results[1].shape : torch.Size([1, 1024, 30, 36])
DEBUG:root:			results[2].shape : torch.Size([1, 1024, 15, 18])
DEBUG:root:			results[3].shape : torch.Size([1, 1024, 8, 9])
DEBUG:root:			results[4].shape : torch.Size([1, 1024, 4, 5])
DEBUG:root:


DEBUG:root:
		results
DEBUG:root:		result[0].shape: torch.Size([1, 1024, 60, 72])
DEBUG:root:		result[1].shape: torch.Size([1, 1024, 30, 36])
DEBUG:root:		result[2].shape: torch.Size([1, 1024, 15, 18])
DEBUG:root:		result[3].shape: torch.Size([1, 1024, 8, 9])
DEBUG:root:		result[4].shape: torch.Size([1, 1024, 4, 5])
DEBUG:root:
	return tuple(results)
DEBUG:root:

	} // END FPN.forward(self,x)
DEBUG:root:	model.backbone.forward(images.tensors) DONE
DEBUG:root:proposals, proposal_losses = self.rpn(images, features, targets) BEFORE
DEBUG:root:

RetinaNetModule.forward(self, images, features, targets=None) { // BEGIN
DEBUG:root:// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/retinanet/retinanet.py
DEBUG:root:	Params:
DEBUG:root:		type(images.image_size): <class 'list'>
DEBUG:root:		type(images.tensors): <class 'torch.Tensor'>
DEBUG:root:		len(features)): 5
DEBUG:root:			feature[0].shape: torch.Size([1, 1024, 60, 72])
DEBUG:root:			feature[1].shape: torch.Size([1, 1024, 30, 36])
DEBUG:root:			feature[2].shape: torch.Size([1, 1024, 15, 18])
DEBUG:root:			feature[3].shape: torch.Size([1, 1024, 8, 9])
DEBUG:root:			feature[4].shape: torch.Size([1, 1024, 4, 5])
DEBUG:root:

	RetinaNetHead.forward(self, x) { // BEGIN
DEBUG:root:	// // defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/retinanet/retinanet.py
DEBUG:root:		Param:
DEBUG:root:			len(x)): 5
DEBUG:root:				x[0].shape: torch.Size([1, 1024, 60, 72])
DEBUG:root:				x[1].shape: torch.Size([1, 1024, 30, 36])
DEBUG:root:				x[2].shape: torch.Size([1, 1024, 15, 18])
DEBUG:root:				x[3].shape: torch.Size([1, 1024, 8, 9])
DEBUG:root:				x[4].shape: torch.Size([1, 1024, 4, 5])
DEBUG:root:		logits = []
DEBUG:root:		bbox_reg = []

DEBUG:root:		self.cls_tower:
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

DEBUG:root:		self.cls_logits:
Conv2d(1024, 9, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

DEBUG:root:		self.bbox_tower:
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

DEBUG:root:		self.bbox_pred:
Conv2d(1024, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

DEBUG:root:

		for idx, feature in enumerate(x) {
DEBUG:root:			===== iteration: 0 ====
DEBUG:root:			feature[0].shape: torch.Size([1, 1024, 60, 72])

DEBUG:root:			logits.append(self.cls_logits(self.cls_tower(feature)))
DEBUG:root:				len(logits): 1

DEBUG:root:			bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
DEBUG:root:				len(bbox_reg): 1

DEBUG:root:			===== iteration: 1 ====
DEBUG:root:			feature[1].shape: torch.Size([1, 1024, 30, 36])

DEBUG:root:			logits.append(self.cls_logits(self.cls_tower(feature)))
DEBUG:root:				len(logits): 2

DEBUG:root:			bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
DEBUG:root:				len(bbox_reg): 2

DEBUG:root:			===== iteration: 2 ====
DEBUG:root:			feature[2].shape: torch.Size([1, 1024, 15, 18])

DEBUG:root:			logits.append(self.cls_logits(self.cls_tower(feature)))
DEBUG:root:				len(logits): 3

DEBUG:root:			bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
DEBUG:root:				len(bbox_reg): 3

DEBUG:root:			===== iteration: 3 ====
DEBUG:root:			feature[3].shape: torch.Size([1, 1024, 8, 9])

DEBUG:root:			logits.append(self.cls_logits(self.cls_tower(feature)))
DEBUG:root:				len(logits): 4

DEBUG:root:			bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
DEBUG:root:				len(bbox_reg): 4

DEBUG:root:			===== iteration: 4 ====
DEBUG:root:			feature[4].shape: torch.Size([1, 1024, 4, 5])

DEBUG:root:			logits.append(self.cls_logits(self.cls_tower(feature)))
DEBUG:root:				len(logits): 5

DEBUG:root:			bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
DEBUG:root:				len(bbox_reg): 5

DEBUG:root:

		}// END for idx, feature n enumerate(x)
DEBUG:root: ==== logits ====
DEBUG:root:logits[0].shape: torch.Size([1, 9, 60, 72])
DEBUG:root:logits[1].shape: torch.Size([1, 9, 30, 36])
DEBUG:root:logits[2].shape: torch.Size([1, 9, 15, 18])
DEBUG:root:logits[3].shape: torch.Size([1, 9, 8, 9])
DEBUG:root:logits[4].shape: torch.Size([1, 9, 4, 5])
DEBUG:root:
 ==== bbox_reg ====
DEBUG:root:bbox_reg[0].shape: torch.Size([1, 36, 60, 72])
DEBUG:root:bbox_reg[1].shape: torch.Size([1, 36, 30, 36])
DEBUG:root:bbox_reg[2].shape: torch.Size([1, 36, 15, 18])
DEBUG:root:bbox_reg[3].shape: torch.Size([1, 36, 8, 9])
DEBUG:root:bbox_reg[4].shape: torch.Size([1, 36, 4, 5])
DEBUG:root:
return logits, bbox_reg
DEBUG:root:	} // END RetinaNetHead.forward(self, x)
DEBUG:root:self.head: RetinaNetHead(
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
DEBUG:root:box_cls, box_regression = self.head(features)
DEBUG:root:	len(box_cls): 5
DEBUG:root:	box_cls[0].shape: torch.Size([1, 9, 60, 72])
DEBUG:root:	box_cls[1].shape: torch.Size([1, 9, 30, 36])
DEBUG:root:	box_cls[2].shape: torch.Size([1, 9, 15, 18])
DEBUG:root:	box_cls[3].shape: torch.Size([1, 9, 8, 9])
DEBUG:root:	box_cls[4].shape: torch.Size([1, 9, 4, 5])
DEBUG:root:	len(box_regression): 5
DEBUG:root:	box_regression[0].shape: torch.Size([1, 36, 60, 72])
DEBUG:root:	box_regression[1].shape: torch.Size([1, 36, 30, 36])
DEBUG:root:	box_regression[2].shape: torch.Size([1, 36, 15, 18])
DEBUG:root:	box_regression[3].shape: torch.Size([1, 36, 8, 9])
DEBUG:root:	box_regression[4].shape: torch.Size([1, 36, 4, 5])
DEBUG:root:
	AnchorGenerator.forward(image_list, feature_maps) { //BEGIN
DEBUG:root:	// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:		Params:
DEBUG:root:			image_list:
DEBUG:root:				len(image_list.image_sizes): 1
DEBUG:root:				image_list.image_sizes[0]: torch.Size([480, 561])
DEBUG:root:				len(image_list.tensors): 1
DEBUG:root:				image_list.tensors[0].shape: torch.Size([3, 480, 576])
DEBUG:root:			feature_maps:
DEBUG:root:				feature_maps[0].shape: torch.Size([1, 1024, 60, 72])
DEBUG:root:				feature_maps[1].shape: torch.Size([1, 1024, 30, 36])
DEBUG:root:				feature_maps[2].shape: torch.Size([1, 1024, 15, 18])
DEBUG:root:				feature_maps[3].shape: torch.Size([1, 1024, 8, 9])
DEBUG:root:				feature_maps[4].shape: torch.Size([1, 1024, 4, 5])
DEBUG:root:
grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
DEBUG:root:anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
DEBUG:root:		AnchorGenerator.grid_anchors(grid_sizes) { // BEGIN
DEBUG:root:			Param:
DEBUG:root:			grid_sizes: [torch.Size([60, 72]), torch.Size([30, 36]), torch.Size([15, 18]), torch.Size([8, 9]), torch.Size([4, 5])]
DEBUG:root:return anchors
DEBUG:root:		} // END AnchorGenerator.grid_anchors(grid_sizes)

DEBUG:root:anchors = []
DEBUG:root:for i, (image_height, image_width) in enumerate(image_list.image_sizes) {

DEBUG:root:		anchors_in_image = []

DEBUG:root:		for anchors_per_feature_map in anchors_over_all_feature_maps {

DEBUG:root:		========================
DEBUG:root:		anchors_per_feature_map.shape: torch.Size([38880, 4])
DEBUG:root:		========================
DEBUG:root:		boxlist = BoxList( anchors_per_feature_map, (image_width, image_height), mode="xyxy" )
DEBUG:root:		boxlist:
			BoxList(num_boxes=38880, image_width=561, image_height=480, mode=xyxy)

DEBUG:root:		self.add_visibility_to(boxlist)

DEBUG:root:AnchorGenerator.add_visibitity_to(boxlist) { // BEGIN
DEBUG:root:// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:		} // END AnchorGenerator.add_visibitity_to(boxlist)

DEBUG:root:		boxlist:
			BoxList(num_boxes=38880, image_width=561, image_height=480, mode=xyxy)

DEBUG:root:		anchors_in_image.append(boxlist)

DEBUG:root:		========================
DEBUG:root:		anchors_per_feature_map.shape: torch.Size([9720, 4])
DEBUG:root:		========================
DEBUG:root:		boxlist = BoxList( anchors_per_feature_map, (image_width, image_height), mode="xyxy" )
DEBUG:root:		boxlist:
			BoxList(num_boxes=9720, image_width=561, image_height=480, mode=xyxy)

DEBUG:root:		self.add_visibility_to(boxlist)

DEBUG:root:AnchorGenerator.add_visibitity_to(boxlist) { // BEGIN
DEBUG:root:// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:		} // END AnchorGenerator.add_visibitity_to(boxlist)

DEBUG:root:		boxlist:
			BoxList(num_boxes=9720, image_width=561, image_height=480, mode=xyxy)

DEBUG:root:		anchors_in_image.append(boxlist)

DEBUG:root:		========================
DEBUG:root:		anchors_per_feature_map.shape: torch.Size([2430, 4])
DEBUG:root:		========================
DEBUG:root:		boxlist = BoxList( anchors_per_feature_map, (image_width, image_height), mode="xyxy" )
DEBUG:root:		boxlist:
			BoxList(num_boxes=2430, image_width=561, image_height=480, mode=xyxy)

DEBUG:root:		self.add_visibility_to(boxlist)

DEBUG:root:AnchorGenerator.add_visibitity_to(boxlist) { // BEGIN
DEBUG:root:// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:		} // END AnchorGenerator.add_visibitity_to(boxlist)

DEBUG:root:		boxlist:
			BoxList(num_boxes=2430, image_width=561, image_height=480, mode=xyxy)

DEBUG:root:		anchors_in_image.append(boxlist)

DEBUG:root:		========================
DEBUG:root:		anchors_per_feature_map.shape: torch.Size([648, 4])
DEBUG:root:		========================
DEBUG:root:		boxlist = BoxList( anchors_per_feature_map, (image_width, image_height), mode="xyxy" )
DEBUG:root:		boxlist:
			BoxList(num_boxes=648, image_width=561, image_height=480, mode=xyxy)

DEBUG:root:		self.add_visibility_to(boxlist)

DEBUG:root:AnchorGenerator.add_visibitity_to(boxlist) { // BEGIN
DEBUG:root:// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:		} // END AnchorGenerator.add_visibitity_to(boxlist)

DEBUG:root:		boxlist:
			BoxList(num_boxes=648, image_width=561, image_height=480, mode=xyxy)

DEBUG:root:		anchors_in_image.append(boxlist)

DEBUG:root:		========================
DEBUG:root:		anchors_per_feature_map.shape: torch.Size([180, 4])
DEBUG:root:		========================
DEBUG:root:		boxlist = BoxList( anchors_per_feature_map, (image_width, image_height), mode="xyxy" )
DEBUG:root:		boxlist:
			BoxList(num_boxes=180, image_width=561, image_height=480, mode=xyxy)

DEBUG:root:		self.add_visibility_to(boxlist)

DEBUG:root:AnchorGenerator.add_visibitity_to(boxlist) { // BEGIN
DEBUG:root:// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
DEBUG:root:		} // END AnchorGenerator.add_visibitity_to(boxlist)

DEBUG:root:		boxlist:
			BoxList(num_boxes=180, image_width=561, image_height=480, mode=xyxy)

DEBUG:root:		anchors_in_image.append(boxlist)

DEBUG:root:		} // END for anchors_per_feature_map in anchors_over_all_feature_maps

DEBUG:root:		anchors_in_image:
			[BoxList(num_boxes=38880, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=9720, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=2430, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=648, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=180, image_width=561, image_height=480, mode=xyxy)]

DEBUG:root:		anchors.append(anchors_in_image)

DEBUG:root:} // END for i, (image_height, image_width) in enumerate(image_list.image_sizes)

DEBUG:root:		anchors:
			[[BoxList(num_boxes=38880, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=9720, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=2430, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=648, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=180, image_width=561, image_height=480, mode=xyxy)]]

DEBUG:root:return anchors
DEBUG:root:	} // END AnchorGenerator.forward(image_list, feature_maps)
DEBUG:root:anchors = self.anchor_generator(images, features)
DEBUG:root:self.anchor_generator: AnchorGenerator(
  (cell_anchors): BufferList()
)
DEBUG:root:anchors: [[BoxList(num_boxes=38880, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=9720, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=2430, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=648, image_width=561, image_height=480, mode=xyxy), BoxList(num_boxes=180, image_width=561, image_height=480, mode=xyxy)]]
DEBUG:root:if self.training: False
DEBUG:root:	return self._forward_test(anchors, box_cls, box_regression)
DEBUG:root:

RetinaNetModule._forward_test(self, anchors, box_cls, box_regression) { // BEGIN
DEBUG:root:// defined in /home/kimkk/work/lomin/maskrcnn_benchmark/modeling/rpn/retinanet/retinanet.py
DEBUG:root:	params:
DEBUG:root:	len(anchors): 1
DEBUG:root:	len(box_cls): 5
DEBUG:root:	len(box_regression): 5
DEBUG:root:	self.box_selector_test: RetinaNetPostProcessor()
DEBUG:root:	boxes = self.box_selector_test(anchors, box_cls, box_regression)
DEBUG:root:		RPNPostProcessing.forward(self. anchors, objectness, box_regression, targets=None) { // BEGIN
DEBUG:root:		Params:
DEBUG:root:		tanchors: len(anchors) : 1
DEBUG:root:		tobjectness: len(objectness) : 5
DEBUG:root:		tbox_regression: type(box_regression) : <class 'list'>
DEBUG:root:		ttarget: None
DEBUG:root:return boxlists
DEBUG:root:		} // END RPNPostProcessing.forward(self. anchors, objectness, box_regression, targets=None)
DEBUG:root:len(boxes): 1
DEBUG:root:(boxes): [BoxList(num_boxes=72, image_width=561, image_height=480, mode=xyxy)]
DEBUG:root:return boxes, {} # {} is just empty dictionayr
DEBUG:root:

} // RetinaNetModule._forward_test(self, anchors, box_cls, box_regression): END
DEBUG:root:} // END RetinaNetModule.forward(self, images, features, targets=None)
DEBUG:root:proposals, proposal_losses = self.rpn(images, features, targets) DONE
DEBUG:root:x = features
DEBUG:root:result = proposals
DEBUG:root:return result
DEBUG:root:} // END GeneralizedRCNN.forward(self, images, targets=None)
DEBUG:root:return pred
DEBUG:root:	pred: BoxList(num_boxes=72, image_width=561, image_height=480, mode=xyxy)
DEBUG:root:} // END compute_prediction(self, image)




