??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8??

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer-1
layer-2
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
 
_
	inshape

trainable_variables
	variables
regularization_losses
	keras_api
 
 
 
?
layer_metrics
trainable_variables
	variables
layer_regularization_losses

layers
regularization_losses
metrics
non_trainable_variables
 
 
 
 
 
?
layer_metrics

trainable_variables
layer_regularization_losses
	variables

layers
regularization_losses
metrics
non_trainable_variables
 
 

0
1
2
 
 
 
 
 
 
 
?
serving_default_input_3Placeholder*6
_output_shapes$
": ????????????*
dtype0*+
shape": ????????????
?
serving_default_input_4Placeholder*6
_output_shapes$
": ????????????*
dtype0*+
shape": ????????????
?
PartitionedCallPartitionedCallserving_default_input_3serving_default_input_4*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference_signature_wrapper_64387
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCallStatefulPartitionedCallsaver_filenameConst*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*'
f"R 
__inference__traced_save_65410
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__traced_restore_65420??
?
l
B__inference_model_2_layer_call_and_return_conditional_losses_64376

inputs
inputs_1
identity?
transformer/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_transformer_layer_call_and_return_conditional_losses_643382
transformer/PartitionedCall?
IdentityIdentity$transformer/PartitionedCall:output:0*
T0*6
_output_shapes$
": ????????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D: ????????????: ????????????:^ Z
6
_output_shapes$
": ????????????
 
_user_specified_nameinputs:^Z
6
_output_shapes$
": ????????????
 
_user_specified_nameinputs
?.
n
B__inference_model_2_layer_call_and_return_conditional_losses_65041
inputs_0
inputs_1
identity?
transformer/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"?????   ?   ?      2
transformer/Reshape/shape?
transformer/ReshapeReshapeinputs_0"transformer/Reshape/shape:output:0*
T0*6
_output_shapes$
": ????????????2
transformer/Reshape?
transformer/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"?????   ?   ?      2
transformer/Reshape_1/shape?
transformer/Reshape_1Reshapeinputs_1$transformer/Reshape_1/shape:output:0*
T0*6
_output_shapes$
": ????????????2
transformer/Reshape_1z
transformer/map/ShapeShapetransformer/Reshape:output:0*
T0*
_output_shapes
:2
transformer/map/Shape?
#transformer/map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#transformer/map/strided_slice/stack?
%transformer/map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%transformer/map/strided_slice/stack_1?
%transformer/map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%transformer/map/strided_slice/stack_2?
transformer/map/strided_sliceStridedSlicetransformer/map/Shape:output:0,transformer/map/strided_slice/stack:output:0.transformer/map/strided_slice/stack_1:output:0.transformer/map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
transformer/map/strided_slice?
+transformer/map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+transformer/map/TensorArrayV2/element_shape?
transformer/map/TensorArrayV2TensorListReserve4transformer/map/TensorArrayV2/element_shape:output:0&transformer/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
transformer/map/TensorArrayV2?
-transformer/map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-transformer/map/TensorArrayV2_1/element_shape?
transformer/map/TensorArrayV2_1TensorListReserve6transformer/map/TensorArrayV2_1/element_shape:output:0&transformer/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
transformer/map/TensorArrayV2_1?
Etransformer/map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      2G
Etransformer/map/TensorArrayUnstack/TensorListFromTensor/element_shape?
7transformer/map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensortransformer/Reshape:output:0Ntransformer/map/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7transformer/map/TensorArrayUnstack/TensorListFromTensor?
Gtransformer/map/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      2I
Gtransformer/map/TensorArrayUnstack_1/TensorListFromTensor/element_shape?
9transformer/map/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortransformer/Reshape_1:output:0Ptransformer/map/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9transformer/map/TensorArrayUnstack_1/TensorListFromTensorp
transformer/map/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
transformer/map/Const?
-transformer/map/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-transformer/map/TensorArrayV2_2/element_shape?
transformer/map/TensorArrayV2_2TensorListReserve6transformer/map/TensorArrayV2_2/element_shape:output:0&transformer/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
transformer/map/TensorArrayV2_2?
"transformer/map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"transformer/map/while/loop_counter?
transformer/map/whileStatelessWhile+transformer/map/while/loop_counter:output:0&transformer/map/strided_slice:output:0transformer/map/Const:output:0(transformer/map/TensorArrayV2_2:handle:0&transformer/map/strided_slice:output:0Gtransformer/map/TensorArrayUnstack/TensorListFromTensor:output_handle:0Itransformer/map/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : * 
_read_only_resource_inputs
 *,
body$R"
 transformer_map_while_body_64739*,
cond$R"
 transformer_map_while_cond_64738*!
output_shapes
: : : : : : : 2
transformer/map/while?
@transformer/map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      2B
@transformer/map/TensorArrayV2Stack/TensorListStack/element_shape?
2transformer/map/TensorArrayV2Stack/TensorListStackTensorListStacktransformer/map/while:output:3Itransformer/map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*6
_output_shapes$
": ????????????*
element_dtype024
2transformer/map/TensorArrayV2Stack/TensorListStack?
IdentityIdentity;transformer/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*6
_output_shapes$
": ????????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D: ????????????: ????????????:` \
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/0:`\
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/1
?
W
+__inference_transformer_layer_call_fn_65386
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_transformer_layer_call_and_return_conditional_losses_643382
PartitionedCall{
IdentityIdentityPartitionedCall:output:0*
T0*6
_output_shapes$
": ????????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D: ????????????: ????????????:` \
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/0:`\
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/1
??
?
 transformer_map_while_body_64739&
"transformer_map_while_loop_counter!
transformer_map_strided_slice
placeholder
placeholder_1%
!transformer_map_strided_slice_1_0a
]tensorarrayv2read_tensorlistgetitem_transformer_map_tensorarrayunstack_tensorlistfromtensor_0e
atensorarrayv2read_1_tensorlistgetitem_transformer_map_tensorarrayunstack_1_tensorlistfromtensor_0
identity

identity_1

identity_2

identity_3#
transformer_map_strided_slice_1_
[tensorarrayv2read_tensorlistgetitem_transformer_map_tensorarrayunstack_tensorlistfromtensorc
_tensorarrayv2read_1_tensorlistgetitem_transformer_map_tensorarrayunstack_1_tensorlistfromtensor?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItem]tensorarrayv2read_tensorlistgetitem_transformer_map_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*)
_output_shapes
:???*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
3TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      25
3TensorArrayV2Read_1/TensorListGetItem/element_shape?
%TensorArrayV2Read_1/TensorListGetItemTensorListGetItematensorarrayv2read_1_tensorlistgetitem_transformer_map_tensorarrayunstack_1_tensorlistfromtensor_0placeholder<TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*)
_output_shapes
:???*
element_dtype02'
%TensorArrayV2Read_1/TensorListGetItem\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start]
range/limitConst*
_output_shapes
: *
dtype0*
value
B :?2
range/limit\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltav
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes	
:?2
range`
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range_1/starta
range_1/limitConst*
_output_shapes
: *
dtype0*
value
B :?2
range_1/limit`
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_1/delta?
range_1Rangerange_1/start:output:0range_1/limit:output:0range_1/delta:output:0*
_output_shapes	
:?2	
range_1`
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range_2/starta
range_2/limitConst*
_output_shapes
: *
dtype0*
value
B :?2
range_2/limit`
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_2/delta?
range_2Rangerange_2/start:output:0range_2/limit:output:0range_2/delta:output:0*
_output_shapes	
:?2	
range_2s
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape/shapes
ReshapeReshaperange:output:0Reshape/shape:output:0*
T0*#
_output_shapes
:?2	
Reshapew
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ????   2
Reshape_1/shape{
	Reshape_1Reshaperange_1:output:0Reshape_1/shape:output:0*
T0*#
_output_shapes
:?2
	Reshape_1w
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
Reshape_2/shape{
	Reshape_2Reshaperange_2:output:0Reshape_2/shape:output:0*
T0*#
_output_shapes
:?2
	Reshape_2O
SizeConst*
_output_shapes
: *
dtype0*
value
B :?2
SizeS
Size_1Const*
_output_shapes
: *
dtype0*
value
B :?2
Size_1S
Size_2Const*
_output_shapes
: *
dtype0*
value
B :?2
Size_2c
stackConst*
_output_shapes
:*
dtype0*!
valueB"   ?   ?   2
stackf
TileTileReshape:output:0stack:output:0*
T0*%
_output_shapes
:???2
Tileg
stack_1Const*
_output_shapes
:*
dtype0*!
valueB"?      ?   2	
stack_1n
Tile_1TileReshape_1:output:0stack_1:output:0*
T0*%
_output_shapes
:???2
Tile_1g
stack_2Const*
_output_shapes
:*
dtype0*!
valueB"?   ?      2	
stack_2n
Tile_2TileReshape_2:output:0stack_2:output:0*
T0*%
_output_shapes
:???2
Tile_2b
CastCastTile:output:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSlice,TensorArrayV2Read_1/TensorListGetItem:item:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slicee
addAddV2Cast:y:0strided_slice:output:0*
T0*%
_output_shapes
:???2
addh
Cast_1CastTile_1:output:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlice,TensorArrayV2Read_1/TensorListGetItem:item:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_1m
add_1AddV2
Cast_1:y:0strided_slice_1:output:0*
T0*%
_output_shapes
:???2
add_1h
Cast_2CastTile_2:output:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSlice,TensorArrayV2Read_1/TensorListGetItem:item:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_2m
add_2AddV2
Cast_2:y:0strided_slice_2:output:0*
T0*%
_output_shapes
:???2
add_2?
stack_3Packadd:z:0	add_1:z:0	add_2:z:0*
N*
T0*)
_output_shapes
:???*
axis?????????2	
stack_3]
FloorFloorstack_3:output:0*
T0*)
_output_shapes
:???2
Floor
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_3w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumstrided_slice_3:output:0 clip_by_value/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSlicestack_3:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_4{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimumstrided_slice_4:output:0"clip_by_value_1/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_1
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSlicestack_3:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_5{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimumstrided_slice_5:output:0"clip_by_value_2/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_2
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSlice	Floor:y:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_6{
clip_by_value_3/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_3/Minimum/y?
clip_by_value_3/MinimumMinimumstrided_slice_6:output:0"clip_by_value_3/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_3/Minimumk
clip_by_value_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_3/y?
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_3
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSlice	Floor:y:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_7{
clip_by_value_4/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_4/Minimum/y?
clip_by_value_4/MinimumMinimumstrided_slice_7:output:0"clip_by_value_4/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_4/Minimumk
clip_by_value_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_4/y?
clip_by_value_4Maximumclip_by_value_4/Minimum:z:0clip_by_value_4/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_4
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSlice	Floor:y:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_8{
clip_by_value_5/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_5/Minimum/y?
clip_by_value_5/MinimumMinimumstrided_slice_8:output:0"clip_by_value_5/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_5/Minimumk
clip_by_value_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_5/y?
clip_by_value_5Maximumclip_by_value_5/Minimum:z:0clip_by_value_5/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_5W
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_3/yn
add_3AddV2clip_by_value_3:z:0add_3/y:output:0*
T0*%
_output_shapes
:???2
add_3{
clip_by_value_6/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_6/Minimum/y?
clip_by_value_6/MinimumMinimum	add_3:z:0"clip_by_value_6/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_6/Minimumk
clip_by_value_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_6/y?
clip_by_value_6Maximumclip_by_value_6/Minimum:z:0clip_by_value_6/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_6W
add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_4/yn
add_4AddV2clip_by_value_4:z:0add_4/y:output:0*
T0*%
_output_shapes
:???2
add_4{
clip_by_value_7/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_7/Minimum/y?
clip_by_value_7/MinimumMinimum	add_4:z:0"clip_by_value_7/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_7/Minimumk
clip_by_value_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_7/y?
clip_by_value_7Maximumclip_by_value_7/Minimum:z:0clip_by_value_7/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_7W
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_5/yn
add_5AddV2clip_by_value_5:z:0add_5/y:output:0*
T0*%
_output_shapes
:???2
add_5{
clip_by_value_8/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_8/Minimum/y?
clip_by_value_8/MinimumMinimum	add_5:z:0"clip_by_value_8/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_8/Minimumk
clip_by_value_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_8/y?
clip_by_value_8Maximumclip_by_value_8/Minimum:z:0clip_by_value_8/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_8l
Cast_3Castclip_by_value_3:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_3l
Cast_4Castclip_by_value_4:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_4l
Cast_5Castclip_by_value_5:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_5l
Cast_6Castclip_by_value_6:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_6l
Cast_7Castclip_by_value_7:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_7l
Cast_8Castclip_by_value_8:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_8i
subSubclip_by_value_6:z:0clip_by_value:z:0*
T0*%
_output_shapes
:???2
subo
sub_1Subclip_by_value_7:z:0clip_by_value_1:z:0*
T0*%
_output_shapes
:???2
sub_1o
sub_2Subclip_by_value_8:z:0clip_by_value_2:z:0*
T0*%
_output_shapes
:???2
sub_2W
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_3/x`
sub_3Subsub_3/x:output:0sub:z:0*
T0*%
_output_shapes
:???2
sub_3W
sub_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_4/xb
sub_4Subsub_4/x:output:0	sub_1:z:0*
T0*%
_output_shapes
:???2
sub_4W
sub_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_5/xb
sub_5Subsub_5/x:output:0	sub_2:z:0*
T0*%
_output_shapes
:???2
sub_5Q
mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
mul/y]
mulMul
Cast_4:y:0mul/y:output:0*
T0*%
_output_shapes
:???2
mul\
add_6AddV2
Cast_5:y:0mul:z:0*
T0*%
_output_shapes
:???2
add_6V
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2	
mul_1/yc
mul_1Mul
Cast_3:y:0mul_1/y:output:0*
T0*%
_output_shapes
:???2
mul_1]
add_7AddV2	add_6:z:0	mul_1:z:0*
T0*%
_output_shapes
:???2
add_7s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_3/shape?
	Reshape_3Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_3/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_3`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Reshape_3:output:0	add_7:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2Y
mul_2Mulsub:z:0	sub_1:z:0*
T0*%
_output_shapes
:???2
mul_2[
mul_3Mul	mul_2:z:0	sub_2:z:0*
T0*%
_output_shapes
:???2
mul_3k
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim~

ExpandDims
ExpandDims	mul_3:z:0ExpandDims/dim:output:0*
T0*)
_output_shapes
:???2

ExpandDimsq
mul_4MulExpandDims:output:0GatherV2:output:0*
T0*)
_output_shapes
:???2
mul_4W
add_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2	
add_8/xh
add_8AddV2add_8/x:output:0	mul_4:z:0*
T0*)
_output_shapes
:???2
add_8U
mul_5/yConst*
_output_shapes
: *
dtype0*
value
B :?2	
mul_5/yc
mul_5Mul
Cast_4:y:0mul_5/y:output:0*
T0*%
_output_shapes
:???2
mul_5^
add_9AddV2
Cast_8:y:0	mul_5:z:0*
T0*%
_output_shapes
:???2
add_9V
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2	
mul_6/yc
mul_6Mul
Cast_3:y:0mul_6/y:output:0*
T0*%
_output_shapes
:???2
mul_6_
add_10AddV2	add_9:z:0	mul_6:z:0*
T0*%
_output_shapes
:???2
add_10s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_4/shape?
	Reshape_4Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_4/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_4d
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axis?

GatherV2_1GatherV2Reshape_4:output:0
add_10:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_1Y
mul_7Mulsub:z:0	sub_1:z:0*
T0*%
_output_shapes
:???2
mul_7[
mul_8Mul	mul_7:z:0	sub_5:z:0*
T0*%
_output_shapes
:???2
mul_8o
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_1/dim?
ExpandDims_1
ExpandDims	mul_8:z:0ExpandDims_1/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_1u
mul_9MulExpandDims_1:output:0GatherV2_1:output:0*
T0*)
_output_shapes
:???2
mul_9c
add_11AddV2	add_8:z:0	mul_9:z:0*
T0*)
_output_shapes
:???2
add_11W
mul_10/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_10/yf
mul_10Mul
Cast_7:y:0mul_10/y:output:0*
T0*%
_output_shapes
:???2
mul_10a
add_12AddV2
Cast_5:y:0
mul_10:z:0*
T0*%
_output_shapes
:???2
add_12X
mul_11/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_11/yf
mul_11Mul
Cast_3:y:0mul_11/y:output:0*
T0*%
_output_shapes
:???2
mul_11a
add_13AddV2
add_12:z:0
mul_11:z:0*
T0*%
_output_shapes
:???2
add_13s
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_5/shape?
	Reshape_5Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_5/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_5d
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_2/axis?

GatherV2_2GatherV2Reshape_5:output:0
add_13:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_2[
mul_12Mulsub:z:0	sub_4:z:0*
T0*%
_output_shapes
:???2
mul_12^
mul_13Mul
mul_12:z:0	sub_2:z:0*
T0*%
_output_shapes
:???2
mul_13o
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_2/dim?
ExpandDims_2
ExpandDims
mul_13:z:0ExpandDims_2/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_2w
mul_14MulExpandDims_2:output:0GatherV2_2:output:0*
T0*)
_output_shapes
:???2
mul_14e
add_14AddV2
add_11:z:0
mul_14:z:0*
T0*)
_output_shapes
:???2
add_14W
mul_15/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_15/yf
mul_15Mul
Cast_7:y:0mul_15/y:output:0*
T0*%
_output_shapes
:???2
mul_15a
add_15AddV2
Cast_8:y:0
mul_15:z:0*
T0*%
_output_shapes
:???2
add_15X
mul_16/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_16/yf
mul_16Mul
Cast_3:y:0mul_16/y:output:0*
T0*%
_output_shapes
:???2
mul_16a
add_16AddV2
add_15:z:0
mul_16:z:0*
T0*%
_output_shapes
:???2
add_16s
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_6/shape?
	Reshape_6Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_6/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_6d
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_3/axis?

GatherV2_3GatherV2Reshape_6:output:0
add_16:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_3[
mul_17Mulsub:z:0	sub_4:z:0*
T0*%
_output_shapes
:???2
mul_17^
mul_18Mul
mul_17:z:0	sub_5:z:0*
T0*%
_output_shapes
:???2
mul_18o
ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_3/dim?
ExpandDims_3
ExpandDims
mul_18:z:0ExpandDims_3/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_3w
mul_19MulExpandDims_3:output:0GatherV2_3:output:0*
T0*)
_output_shapes
:???2
mul_19e
add_17AddV2
add_14:z:0
mul_19:z:0*
T0*)
_output_shapes
:???2
add_17W
mul_20/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_20/yf
mul_20Mul
Cast_4:y:0mul_20/y:output:0*
T0*%
_output_shapes
:???2
mul_20a
add_18AddV2
Cast_5:y:0
mul_20:z:0*
T0*%
_output_shapes
:???2
add_18X
mul_21/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_21/yf
mul_21Mul
Cast_6:y:0mul_21/y:output:0*
T0*%
_output_shapes
:???2
mul_21a
add_19AddV2
add_18:z:0
mul_21:z:0*
T0*%
_output_shapes
:???2
add_19s
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_7/shape?
	Reshape_7Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_7/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_7d
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_4/axis?

GatherV2_4GatherV2Reshape_7:output:0
add_19:z:0GatherV2_4/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_4]
mul_22Mul	sub_3:z:0	sub_1:z:0*
T0*%
_output_shapes
:???2
mul_22^
mul_23Mul
mul_22:z:0	sub_2:z:0*
T0*%
_output_shapes
:???2
mul_23o
ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_4/dim?
ExpandDims_4
ExpandDims
mul_23:z:0ExpandDims_4/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_4w
mul_24MulExpandDims_4:output:0GatherV2_4:output:0*
T0*)
_output_shapes
:???2
mul_24e
add_20AddV2
add_17:z:0
mul_24:z:0*
T0*)
_output_shapes
:???2
add_20W
mul_25/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_25/yf
mul_25Mul
Cast_4:y:0mul_25/y:output:0*
T0*%
_output_shapes
:???2
mul_25a
add_21AddV2
Cast_8:y:0
mul_25:z:0*
T0*%
_output_shapes
:???2
add_21X
mul_26/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_26/yf
mul_26Mul
Cast_6:y:0mul_26/y:output:0*
T0*%
_output_shapes
:???2
mul_26a
add_22AddV2
add_21:z:0
mul_26:z:0*
T0*%
_output_shapes
:???2
add_22s
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_8/shape?
	Reshape_8Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_8/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_8d
GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_5/axis?

GatherV2_5GatherV2Reshape_8:output:0
add_22:z:0GatherV2_5/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_5]
mul_27Mul	sub_3:z:0	sub_1:z:0*
T0*%
_output_shapes
:???2
mul_27^
mul_28Mul
mul_27:z:0	sub_5:z:0*
T0*%
_output_shapes
:???2
mul_28o
ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_5/dim?
ExpandDims_5
ExpandDims
mul_28:z:0ExpandDims_5/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_5w
mul_29MulExpandDims_5:output:0GatherV2_5:output:0*
T0*)
_output_shapes
:???2
mul_29e
add_23AddV2
add_20:z:0
mul_29:z:0*
T0*)
_output_shapes
:???2
add_23W
mul_30/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_30/yf
mul_30Mul
Cast_7:y:0mul_30/y:output:0*
T0*%
_output_shapes
:???2
mul_30a
add_24AddV2
Cast_5:y:0
mul_30:z:0*
T0*%
_output_shapes
:???2
add_24X
mul_31/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_31/yf
mul_31Mul
Cast_6:y:0mul_31/y:output:0*
T0*%
_output_shapes
:???2
mul_31a
add_25AddV2
add_24:z:0
mul_31:z:0*
T0*%
_output_shapes
:???2
add_25s
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_9/shape?
	Reshape_9Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_9/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_9d
GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_6/axis?

GatherV2_6GatherV2Reshape_9:output:0
add_25:z:0GatherV2_6/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_6]
mul_32Mul	sub_3:z:0	sub_4:z:0*
T0*%
_output_shapes
:???2
mul_32^
mul_33Mul
mul_32:z:0	sub_2:z:0*
T0*%
_output_shapes
:???2
mul_33o
ExpandDims_6/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_6/dim?
ExpandDims_6
ExpandDims
mul_33:z:0ExpandDims_6/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_6w
mul_34MulExpandDims_6:output:0GatherV2_6:output:0*
T0*)
_output_shapes
:???2
mul_34e
add_26AddV2
add_23:z:0
mul_34:z:0*
T0*)
_output_shapes
:???2
add_26W
mul_35/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_35/yf
mul_35Mul
Cast_7:y:0mul_35/y:output:0*
T0*%
_output_shapes
:???2
mul_35a
add_27AddV2
Cast_8:y:0
mul_35:z:0*
T0*%
_output_shapes
:???2
add_27X
mul_36/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_36/yf
mul_36Mul
Cast_6:y:0mul_36/y:output:0*
T0*%
_output_shapes
:???2
mul_36a
add_28AddV2
add_27:z:0
mul_36:z:0*
T0*%
_output_shapes
:???2
add_28u
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_10/shape?

Reshape_10Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_10/shape:output:0*
T0*!
_output_shapes
:???2

Reshape_10d
GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_7/axis?

GatherV2_7GatherV2Reshape_10:output:0
add_28:z:0GatherV2_7/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_7]
mul_37Mul	sub_3:z:0	sub_4:z:0*
T0*%
_output_shapes
:???2
mul_37^
mul_38Mul
mul_37:z:0	sub_5:z:0*
T0*%
_output_shapes
:???2
mul_38o
ExpandDims_7/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_7/dim?
ExpandDims_7
ExpandDims
mul_38:z:0ExpandDims_7/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_7w
mul_39MulExpandDims_7:output:0GatherV2_7:output:0*
T0*)
_output_shapes
:???2
mul_39e
add_29AddV2
add_26:z:0
mul_39:z:0*
T0*)
_output_shapes
:???2
add_29?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder
add_29:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemV
add_30/yConst*
_output_shapes
: *
dtype0*
value	B :2

add_30/yZ
add_30AddV2placeholderadd_30/y:output:0*
T0*
_output_shapes
: 2
add_30V
add_31/yConst*
_output_shapes
: *
dtype0*
value	B :2

add_31/yq
add_31AddV2"transformer_map_while_loop_counteradd_31/y:output:0*
T0*
_output_shapes
: 2
add_31M
IdentityIdentity
add_31:z:0*
T0*
_output_shapes
: 2

Identityd

Identity_1Identitytransformer_map_strided_slice*
T0*
_output_shapes
: 2

Identity_1Q

Identity_2Identity
add_30:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"?
_tensorarrayv2read_1_tensorlistgetitem_transformer_map_tensorarrayunstack_1_tensorlistfromtensoratensorarrayv2read_1_tensorlistgetitem_transformer_map_tensorarrayunstack_1_tensorlistfromtensor_0"?
[tensorarrayv2read_tensorlistgetitem_transformer_map_tensorarrayunstack_tensorlistfromtensor]tensorarrayv2read_tensorlistgetitem_transformer_map_tensorarrayunstack_tensorlistfromtensor_0"D
transformer_map_strided_slice_1!transformer_map_strided_slice_1_0*!
_input_shapes
: : : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
map_while_body_65078
map_while_loop_counter
map_strided_slice
placeholder
placeholder_1
map_strided_slice_1_0U
Qtensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0Y
Utensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_0
identity

identity_1

identity_2

identity_3
map_strided_slice_1S
Otensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensorW
Stensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemQtensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*)
_output_shapes
:???*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
3TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      25
3TensorArrayV2Read_1/TensorListGetItem/element_shape?
%TensorArrayV2Read_1/TensorListGetItemTensorListGetItemUtensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_0placeholder<TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*)
_output_shapes
:???*
element_dtype02'
%TensorArrayV2Read_1/TensorListGetItem\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start]
range/limitConst*
_output_shapes
: *
dtype0*
value
B :?2
range/limit\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltav
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes	
:?2
range`
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range_1/starta
range_1/limitConst*
_output_shapes
: *
dtype0*
value
B :?2
range_1/limit`
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_1/delta?
range_1Rangerange_1/start:output:0range_1/limit:output:0range_1/delta:output:0*
_output_shapes	
:?2	
range_1`
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range_2/starta
range_2/limitConst*
_output_shapes
: *
dtype0*
value
B :?2
range_2/limit`
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_2/delta?
range_2Rangerange_2/start:output:0range_2/limit:output:0range_2/delta:output:0*
_output_shapes	
:?2	
range_2s
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape/shapes
ReshapeReshaperange:output:0Reshape/shape:output:0*
T0*#
_output_shapes
:?2	
Reshapew
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ????   2
Reshape_1/shape{
	Reshape_1Reshaperange_1:output:0Reshape_1/shape:output:0*
T0*#
_output_shapes
:?2
	Reshape_1w
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
Reshape_2/shape{
	Reshape_2Reshaperange_2:output:0Reshape_2/shape:output:0*
T0*#
_output_shapes
:?2
	Reshape_2O
SizeConst*
_output_shapes
: *
dtype0*
value
B :?2
SizeS
Size_1Const*
_output_shapes
: *
dtype0*
value
B :?2
Size_1S
Size_2Const*
_output_shapes
: *
dtype0*
value
B :?2
Size_2c
stackConst*
_output_shapes
:*
dtype0*!
valueB"   ?   ?   2
stackf
TileTileReshape:output:0stack:output:0*
T0*%
_output_shapes
:???2
Tileg
stack_1Const*
_output_shapes
:*
dtype0*!
valueB"?      ?   2	
stack_1n
Tile_1TileReshape_1:output:0stack_1:output:0*
T0*%
_output_shapes
:???2
Tile_1g
stack_2Const*
_output_shapes
:*
dtype0*!
valueB"?   ?      2	
stack_2n
Tile_2TileReshape_2:output:0stack_2:output:0*
T0*%
_output_shapes
:???2
Tile_2b
CastCastTile:output:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSlice,TensorArrayV2Read_1/TensorListGetItem:item:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slicee
addAddV2Cast:y:0strided_slice:output:0*
T0*%
_output_shapes
:???2
addh
Cast_1CastTile_1:output:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlice,TensorArrayV2Read_1/TensorListGetItem:item:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_1m
add_1AddV2
Cast_1:y:0strided_slice_1:output:0*
T0*%
_output_shapes
:???2
add_1h
Cast_2CastTile_2:output:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSlice,TensorArrayV2Read_1/TensorListGetItem:item:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_2m
add_2AddV2
Cast_2:y:0strided_slice_2:output:0*
T0*%
_output_shapes
:???2
add_2?
stack_3Packadd:z:0	add_1:z:0	add_2:z:0*
N*
T0*)
_output_shapes
:???*
axis?????????2	
stack_3]
FloorFloorstack_3:output:0*
T0*)
_output_shapes
:???2
Floor
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_3w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumstrided_slice_3:output:0 clip_by_value/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSlicestack_3:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_4{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimumstrided_slice_4:output:0"clip_by_value_1/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_1
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSlicestack_3:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_5{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimumstrided_slice_5:output:0"clip_by_value_2/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_2
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSlice	Floor:y:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_6{
clip_by_value_3/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_3/Minimum/y?
clip_by_value_3/MinimumMinimumstrided_slice_6:output:0"clip_by_value_3/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_3/Minimumk
clip_by_value_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_3/y?
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_3
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSlice	Floor:y:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_7{
clip_by_value_4/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_4/Minimum/y?
clip_by_value_4/MinimumMinimumstrided_slice_7:output:0"clip_by_value_4/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_4/Minimumk
clip_by_value_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_4/y?
clip_by_value_4Maximumclip_by_value_4/Minimum:z:0clip_by_value_4/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_4
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSlice	Floor:y:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_8{
clip_by_value_5/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_5/Minimum/y?
clip_by_value_5/MinimumMinimumstrided_slice_8:output:0"clip_by_value_5/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_5/Minimumk
clip_by_value_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_5/y?
clip_by_value_5Maximumclip_by_value_5/Minimum:z:0clip_by_value_5/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_5W
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_3/yn
add_3AddV2clip_by_value_3:z:0add_3/y:output:0*
T0*%
_output_shapes
:???2
add_3{
clip_by_value_6/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_6/Minimum/y?
clip_by_value_6/MinimumMinimum	add_3:z:0"clip_by_value_6/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_6/Minimumk
clip_by_value_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_6/y?
clip_by_value_6Maximumclip_by_value_6/Minimum:z:0clip_by_value_6/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_6W
add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_4/yn
add_4AddV2clip_by_value_4:z:0add_4/y:output:0*
T0*%
_output_shapes
:???2
add_4{
clip_by_value_7/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_7/Minimum/y?
clip_by_value_7/MinimumMinimum	add_4:z:0"clip_by_value_7/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_7/Minimumk
clip_by_value_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_7/y?
clip_by_value_7Maximumclip_by_value_7/Minimum:z:0clip_by_value_7/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_7W
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_5/yn
add_5AddV2clip_by_value_5:z:0add_5/y:output:0*
T0*%
_output_shapes
:???2
add_5{
clip_by_value_8/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_8/Minimum/y?
clip_by_value_8/MinimumMinimum	add_5:z:0"clip_by_value_8/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_8/Minimumk
clip_by_value_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_8/y?
clip_by_value_8Maximumclip_by_value_8/Minimum:z:0clip_by_value_8/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_8l
Cast_3Castclip_by_value_3:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_3l
Cast_4Castclip_by_value_4:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_4l
Cast_5Castclip_by_value_5:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_5l
Cast_6Castclip_by_value_6:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_6l
Cast_7Castclip_by_value_7:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_7l
Cast_8Castclip_by_value_8:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_8i
subSubclip_by_value_6:z:0clip_by_value:z:0*
T0*%
_output_shapes
:???2
subo
sub_1Subclip_by_value_7:z:0clip_by_value_1:z:0*
T0*%
_output_shapes
:???2
sub_1o
sub_2Subclip_by_value_8:z:0clip_by_value_2:z:0*
T0*%
_output_shapes
:???2
sub_2W
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_3/x`
sub_3Subsub_3/x:output:0sub:z:0*
T0*%
_output_shapes
:???2
sub_3W
sub_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_4/xb
sub_4Subsub_4/x:output:0	sub_1:z:0*
T0*%
_output_shapes
:???2
sub_4W
sub_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_5/xb
sub_5Subsub_5/x:output:0	sub_2:z:0*
T0*%
_output_shapes
:???2
sub_5Q
mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
mul/y]
mulMul
Cast_4:y:0mul/y:output:0*
T0*%
_output_shapes
:???2
mul\
add_6AddV2
Cast_5:y:0mul:z:0*
T0*%
_output_shapes
:???2
add_6V
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2	
mul_1/yc
mul_1Mul
Cast_3:y:0mul_1/y:output:0*
T0*%
_output_shapes
:???2
mul_1]
add_7AddV2	add_6:z:0	mul_1:z:0*
T0*%
_output_shapes
:???2
add_7s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_3/shape?
	Reshape_3Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_3/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_3`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Reshape_3:output:0	add_7:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2Y
mul_2Mulsub:z:0	sub_1:z:0*
T0*%
_output_shapes
:???2
mul_2[
mul_3Mul	mul_2:z:0	sub_2:z:0*
T0*%
_output_shapes
:???2
mul_3k
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim~

ExpandDims
ExpandDims	mul_3:z:0ExpandDims/dim:output:0*
T0*)
_output_shapes
:???2

ExpandDimsq
mul_4MulExpandDims:output:0GatherV2:output:0*
T0*)
_output_shapes
:???2
mul_4W
add_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2	
add_8/xh
add_8AddV2add_8/x:output:0	mul_4:z:0*
T0*)
_output_shapes
:???2
add_8U
mul_5/yConst*
_output_shapes
: *
dtype0*
value
B :?2	
mul_5/yc
mul_5Mul
Cast_4:y:0mul_5/y:output:0*
T0*%
_output_shapes
:???2
mul_5^
add_9AddV2
Cast_8:y:0	mul_5:z:0*
T0*%
_output_shapes
:???2
add_9V
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2	
mul_6/yc
mul_6Mul
Cast_3:y:0mul_6/y:output:0*
T0*%
_output_shapes
:???2
mul_6_
add_10AddV2	add_9:z:0	mul_6:z:0*
T0*%
_output_shapes
:???2
add_10s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_4/shape?
	Reshape_4Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_4/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_4d
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axis?

GatherV2_1GatherV2Reshape_4:output:0
add_10:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_1Y
mul_7Mulsub:z:0	sub_1:z:0*
T0*%
_output_shapes
:???2
mul_7[
mul_8Mul	mul_7:z:0	sub_5:z:0*
T0*%
_output_shapes
:???2
mul_8o
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_1/dim?
ExpandDims_1
ExpandDims	mul_8:z:0ExpandDims_1/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_1u
mul_9MulExpandDims_1:output:0GatherV2_1:output:0*
T0*)
_output_shapes
:???2
mul_9c
add_11AddV2	add_8:z:0	mul_9:z:0*
T0*)
_output_shapes
:???2
add_11W
mul_10/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_10/yf
mul_10Mul
Cast_7:y:0mul_10/y:output:0*
T0*%
_output_shapes
:???2
mul_10a
add_12AddV2
Cast_5:y:0
mul_10:z:0*
T0*%
_output_shapes
:???2
add_12X
mul_11/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_11/yf
mul_11Mul
Cast_3:y:0mul_11/y:output:0*
T0*%
_output_shapes
:???2
mul_11a
add_13AddV2
add_12:z:0
mul_11:z:0*
T0*%
_output_shapes
:???2
add_13s
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_5/shape?
	Reshape_5Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_5/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_5d
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_2/axis?

GatherV2_2GatherV2Reshape_5:output:0
add_13:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_2[
mul_12Mulsub:z:0	sub_4:z:0*
T0*%
_output_shapes
:???2
mul_12^
mul_13Mul
mul_12:z:0	sub_2:z:0*
T0*%
_output_shapes
:???2
mul_13o
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_2/dim?
ExpandDims_2
ExpandDims
mul_13:z:0ExpandDims_2/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_2w
mul_14MulExpandDims_2:output:0GatherV2_2:output:0*
T0*)
_output_shapes
:???2
mul_14e
add_14AddV2
add_11:z:0
mul_14:z:0*
T0*)
_output_shapes
:???2
add_14W
mul_15/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_15/yf
mul_15Mul
Cast_7:y:0mul_15/y:output:0*
T0*%
_output_shapes
:???2
mul_15a
add_15AddV2
Cast_8:y:0
mul_15:z:0*
T0*%
_output_shapes
:???2
add_15X
mul_16/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_16/yf
mul_16Mul
Cast_3:y:0mul_16/y:output:0*
T0*%
_output_shapes
:???2
mul_16a
add_16AddV2
add_15:z:0
mul_16:z:0*
T0*%
_output_shapes
:???2
add_16s
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_6/shape?
	Reshape_6Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_6/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_6d
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_3/axis?

GatherV2_3GatherV2Reshape_6:output:0
add_16:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_3[
mul_17Mulsub:z:0	sub_4:z:0*
T0*%
_output_shapes
:???2
mul_17^
mul_18Mul
mul_17:z:0	sub_5:z:0*
T0*%
_output_shapes
:???2
mul_18o
ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_3/dim?
ExpandDims_3
ExpandDims
mul_18:z:0ExpandDims_3/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_3w
mul_19MulExpandDims_3:output:0GatherV2_3:output:0*
T0*)
_output_shapes
:???2
mul_19e
add_17AddV2
add_14:z:0
mul_19:z:0*
T0*)
_output_shapes
:???2
add_17W
mul_20/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_20/yf
mul_20Mul
Cast_4:y:0mul_20/y:output:0*
T0*%
_output_shapes
:???2
mul_20a
add_18AddV2
Cast_5:y:0
mul_20:z:0*
T0*%
_output_shapes
:???2
add_18X
mul_21/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_21/yf
mul_21Mul
Cast_6:y:0mul_21/y:output:0*
T0*%
_output_shapes
:???2
mul_21a
add_19AddV2
add_18:z:0
mul_21:z:0*
T0*%
_output_shapes
:???2
add_19s
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_7/shape?
	Reshape_7Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_7/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_7d
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_4/axis?

GatherV2_4GatherV2Reshape_7:output:0
add_19:z:0GatherV2_4/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_4]
mul_22Mul	sub_3:z:0	sub_1:z:0*
T0*%
_output_shapes
:???2
mul_22^
mul_23Mul
mul_22:z:0	sub_2:z:0*
T0*%
_output_shapes
:???2
mul_23o
ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_4/dim?
ExpandDims_4
ExpandDims
mul_23:z:0ExpandDims_4/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_4w
mul_24MulExpandDims_4:output:0GatherV2_4:output:0*
T0*)
_output_shapes
:???2
mul_24e
add_20AddV2
add_17:z:0
mul_24:z:0*
T0*)
_output_shapes
:???2
add_20W
mul_25/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_25/yf
mul_25Mul
Cast_4:y:0mul_25/y:output:0*
T0*%
_output_shapes
:???2
mul_25a
add_21AddV2
Cast_8:y:0
mul_25:z:0*
T0*%
_output_shapes
:???2
add_21X
mul_26/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_26/yf
mul_26Mul
Cast_6:y:0mul_26/y:output:0*
T0*%
_output_shapes
:???2
mul_26a
add_22AddV2
add_21:z:0
mul_26:z:0*
T0*%
_output_shapes
:???2
add_22s
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_8/shape?
	Reshape_8Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_8/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_8d
GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_5/axis?

GatherV2_5GatherV2Reshape_8:output:0
add_22:z:0GatherV2_5/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_5]
mul_27Mul	sub_3:z:0	sub_1:z:0*
T0*%
_output_shapes
:???2
mul_27^
mul_28Mul
mul_27:z:0	sub_5:z:0*
T0*%
_output_shapes
:???2
mul_28o
ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_5/dim?
ExpandDims_5
ExpandDims
mul_28:z:0ExpandDims_5/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_5w
mul_29MulExpandDims_5:output:0GatherV2_5:output:0*
T0*)
_output_shapes
:???2
mul_29e
add_23AddV2
add_20:z:0
mul_29:z:0*
T0*)
_output_shapes
:???2
add_23W
mul_30/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_30/yf
mul_30Mul
Cast_7:y:0mul_30/y:output:0*
T0*%
_output_shapes
:???2
mul_30a
add_24AddV2
Cast_5:y:0
mul_30:z:0*
T0*%
_output_shapes
:???2
add_24X
mul_31/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_31/yf
mul_31Mul
Cast_6:y:0mul_31/y:output:0*
T0*%
_output_shapes
:???2
mul_31a
add_25AddV2
add_24:z:0
mul_31:z:0*
T0*%
_output_shapes
:???2
add_25s
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_9/shape?
	Reshape_9Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_9/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_9d
GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_6/axis?

GatherV2_6GatherV2Reshape_9:output:0
add_25:z:0GatherV2_6/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_6]
mul_32Mul	sub_3:z:0	sub_4:z:0*
T0*%
_output_shapes
:???2
mul_32^
mul_33Mul
mul_32:z:0	sub_2:z:0*
T0*%
_output_shapes
:???2
mul_33o
ExpandDims_6/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_6/dim?
ExpandDims_6
ExpandDims
mul_33:z:0ExpandDims_6/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_6w
mul_34MulExpandDims_6:output:0GatherV2_6:output:0*
T0*)
_output_shapes
:???2
mul_34e
add_26AddV2
add_23:z:0
mul_34:z:0*
T0*)
_output_shapes
:???2
add_26W
mul_35/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_35/yf
mul_35Mul
Cast_7:y:0mul_35/y:output:0*
T0*%
_output_shapes
:???2
mul_35a
add_27AddV2
Cast_8:y:0
mul_35:z:0*
T0*%
_output_shapes
:???2
add_27X
mul_36/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_36/yf
mul_36Mul
Cast_6:y:0mul_36/y:output:0*
T0*%
_output_shapes
:???2
mul_36a
add_28AddV2
add_27:z:0
mul_36:z:0*
T0*%
_output_shapes
:???2
add_28u
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_10/shape?

Reshape_10Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_10/shape:output:0*
T0*!
_output_shapes
:???2

Reshape_10d
GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_7/axis?

GatherV2_7GatherV2Reshape_10:output:0
add_28:z:0GatherV2_7/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_7]
mul_37Mul	sub_3:z:0	sub_4:z:0*
T0*%
_output_shapes
:???2
mul_37^
mul_38Mul
mul_37:z:0	sub_5:z:0*
T0*%
_output_shapes
:???2
mul_38o
ExpandDims_7/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_7/dim?
ExpandDims_7
ExpandDims
mul_38:z:0ExpandDims_7/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_7w
mul_39MulExpandDims_7:output:0GatherV2_7:output:0*
T0*)
_output_shapes
:???2
mul_39e
add_29AddV2
add_26:z:0
mul_39:z:0*
T0*)
_output_shapes
:???2
add_29?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder
add_29:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemV
add_30/yConst*
_output_shapes
: *
dtype0*
value	B :2

add_30/yZ
add_30AddV2placeholderadd_30/y:output:0*
T0*
_output_shapes
: 2
add_30V
add_31/yConst*
_output_shapes
: *
dtype0*
value	B :2

add_31/ye
add_31AddV2map_while_loop_counteradd_31/y:output:0*
T0*
_output_shapes
: 2
add_31M
IdentityIdentity
add_31:z:0*
T0*
_output_shapes
: 2

IdentityX

Identity_1Identitymap_strided_slice*
T0*
_output_shapes
: 2

Identity_1Q

Identity_2Identity
add_30:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0",
map_strided_slice_1map_strided_slice_1_0"?
Stensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensorUtensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_0"?
Otensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensorQtensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*!
_input_shapes
: : : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
S
!__inference__traced_restore_65420
file_prefix

identity_1??	RestoreV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:*
dtypes
22
	RestoreV29
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpd
IdentityIdentityfile_prefix^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identityd

Identity_1IdentityIdentity:output:0
^RestoreV2*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: 2
	RestoreV2	RestoreV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
l
B__inference_model_2_layer_call_and_return_conditional_losses_64354
input_3
input_4
identity?
transformer/PartitionedCallPartitionedCallinput_3input_4*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_transformer_layer_call_and_return_conditional_losses_643382
transformer/PartitionedCall?
IdentityIdentity$transformer/PartitionedCall:output:0*
T0*6
_output_shapes$
": ????????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D: ????????????: ????????????:_ [
6
_output_shapes$
": ????????????
!
_user_specified_name	input_3:_[
6
_output_shapes$
": ????????????
!
_user_specified_name	input_4
?
?
 transformer_map_while_cond_64411&
"transformer_map_while_loop_counter!
transformer_map_strided_slice
placeholder
placeholder_1&
"less_transformer_map_strided_slice=
9transformer_map_while_cond_64411___redundant_placeholder0=
9transformer_map_while_cond_64411___redundant_placeholder1
identity
f
LessLessplaceholder"less_transformer_map_strided_slice*
T0*
_output_shapes
: 2
Less|
Less_1Less"transformer_map_while_loop_countertransformer_map_strided_slice*
T0*
_output_shapes
: 2
Less_1T

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: 2

LogicalAndQ
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
?'
r
F__inference_transformer_layer_call_and_return_conditional_losses_65380
inputs_0
inputs_1
identity{
Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"?????   ?   ?      2
Reshape/shape?
ReshapeReshapeinputs_0Reshape/shape:output:0*
T0*6
_output_shapes$
": ????????????2	
Reshape
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"?????   ?   ?      2
Reshape_1/shape?
	Reshape_1Reshapeinputs_1Reshape_1/shape:output:0*
T0*6
_output_shapes$
": ????????????2
	Reshape_1V
	map/ShapeShapeReshape:output:0*
T0*
_output_shapes
:2
	map/Shape|
map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
map/strided_slice/stack?
map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
map/strided_slice/stack_1?
map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
map/strided_slice/stack_2?
map/strided_sliceStridedSlicemap/Shape:output:0 map/strided_slice/stack:output:0"map/strided_slice/stack_1:output:0"map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
map/strided_slice?
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
map/TensorArrayV2/element_shape?
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2?
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!map/TensorArrayV2_1/element_shape?
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2_1?
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      2;
9map/TensorArrayUnstack/TensorListFromTensor/element_shape?
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReshape:output:0Bmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+map/TensorArrayUnstack/TensorListFromTensor?
;map/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      2=
;map/TensorArrayUnstack_1/TensorListFromTensor/element_shape?
-map/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorReshape_1:output:0Dmap/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-map/TensorArrayUnstack_1/TensorListFromTensorX
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
	map/Const?
!map/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!map/TensorArrayV2_2/element_shape?
map/TensorArrayV2_2TensorListReserve*map/TensorArrayV2_2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2_2r
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
map/while/loop_counter?
	map/whileStatelessWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_2:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0=map/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : * 
_read_only_resource_inputs
 * 
bodyR
map_while_body_65078* 
condR
map_while_cond_65077*!
output_shapes
: : : : : : : 2
	map/while?
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      26
4map/TensorArrayV2Stack/TensorListStack/element_shape?
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*6
_output_shapes$
": ????????????*
element_dtype02(
&map/TensorArrayV2Stack/TensorListStack?
IdentityIdentity/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*6
_output_shapes$
": ????????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D: ????????????: ????????????:` \
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/0:`\
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/1
?3
J
 __inference__wrapped_model_64006
input_3
input_4
identity?
!model_2/transformer/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"?????   ?   ?      2#
!model_2/transformer/Reshape/shape?
model_2/transformer/ReshapeReshapeinput_3*model_2/transformer/Reshape/shape:output:0*
T0*6
_output_shapes$
": ????????????2
model_2/transformer/Reshape?
#model_2/transformer/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"?????   ?   ?      2%
#model_2/transformer/Reshape_1/shape?
model_2/transformer/Reshape_1Reshapeinput_4,model_2/transformer/Reshape_1/shape:output:0*
T0*6
_output_shapes$
": ????????????2
model_2/transformer/Reshape_1?
model_2/transformer/map/ShapeShape$model_2/transformer/Reshape:output:0*
T0*
_output_shapes
:2
model_2/transformer/map/Shape?
+model_2/transformer/map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+model_2/transformer/map/strided_slice/stack?
-model_2/transformer/map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_2/transformer/map/strided_slice/stack_1?
-model_2/transformer/map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_2/transformer/map/strided_slice/stack_2?
%model_2/transformer/map/strided_sliceStridedSlice&model_2/transformer/map/Shape:output:04model_2/transformer/map/strided_slice/stack:output:06model_2/transformer/map/strided_slice/stack_1:output:06model_2/transformer/map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%model_2/transformer/map/strided_slice?
3model_2/transformer/map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????25
3model_2/transformer/map/TensorArrayV2/element_shape?
%model_2/transformer/map/TensorArrayV2TensorListReserve<model_2/transformer/map/TensorArrayV2/element_shape:output:0.model_2/transformer/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%model_2/transformer/map/TensorArrayV2?
5model_2/transformer/map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????27
5model_2/transformer/map/TensorArrayV2_1/element_shape?
'model_2/transformer/map/TensorArrayV2_1TensorListReserve>model_2/transformer/map/TensorArrayV2_1/element_shape:output:0.model_2/transformer/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'model_2/transformer/map/TensorArrayV2_1?
Mmodel_2/transformer/map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      2O
Mmodel_2/transformer/map/TensorArrayUnstack/TensorListFromTensor/element_shape?
?model_2/transformer/map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$model_2/transformer/Reshape:output:0Vmodel_2/transformer/map/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?model_2/transformer/map/TensorArrayUnstack/TensorListFromTensor?
Omodel_2/transformer/map/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      2Q
Omodel_2/transformer/map/TensorArrayUnstack_1/TensorListFromTensor/element_shape?
Amodel_2/transformer/map/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor&model_2/transformer/Reshape_1:output:0Xmodel_2/transformer/map/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02C
Amodel_2/transformer/map/TensorArrayUnstack_1/TensorListFromTensor?
model_2/transformer/map/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
model_2/transformer/map/Const?
5model_2/transformer/map/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????27
5model_2/transformer/map/TensorArrayV2_2/element_shape?
'model_2/transformer/map/TensorArrayV2_2TensorListReserve>model_2/transformer/map/TensorArrayV2_2/element_shape:output:0.model_2/transformer/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'model_2/transformer/map/TensorArrayV2_2?
*model_2/transformer/map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model_2/transformer/map/while/loop_counter?
model_2/transformer/map/whileStatelessWhile3model_2/transformer/map/while/loop_counter:output:0.model_2/transformer/map/strided_slice:output:0&model_2/transformer/map/Const:output:00model_2/transformer/map/TensorArrayV2_2:handle:0.model_2/transformer/map/strided_slice:output:0Omodel_2/transformer/map/TensorArrayUnstack/TensorListFromTensor:output_handle:0Qmodel_2/transformer/map/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : * 
_read_only_resource_inputs
 *4
body,R*
(model_2_transformer_map_while_body_63704*4
cond,R*
(model_2_transformer_map_while_cond_63703*!
output_shapes
: : : : : : : 2
model_2/transformer/map/while?
Hmodel_2/transformer/map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      2J
Hmodel_2/transformer/map/TensorArrayV2Stack/TensorListStack/element_shape?
:model_2/transformer/map/TensorArrayV2Stack/TensorListStackTensorListStack&model_2/transformer/map/while:output:3Qmodel_2/transformer/map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*6
_output_shapes$
": ????????????*
element_dtype02<
:model_2/transformer/map/TensorArrayV2Stack/TensorListStack?
IdentityIdentityCmodel_2/transformer/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*6
_output_shapes$
": ????????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D: ????????????: ????????????:_ [
6
_output_shapes$
": ????????????
!
_user_specified_name	input_3:_[
6
_output_shapes$
": ????????????
!
_user_specified_name	input_4
?
Q
'__inference_model_2_layer_call_fn_64379
input_3
input_4
identity?
PartitionedCallPartitionedCallinput_3input_4*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_643762
PartitionedCall{
IdentityIdentityPartitionedCall:output:0*
T0*6
_output_shapes$
": ????????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D: ????????????: ????????????:_ [
6
_output_shapes$
": ????????????
!
_user_specified_name	input_3:_[
6
_output_shapes$
": ????????????
!
_user_specified_name	input_4
?
M
#__inference_signature_wrapper_64387
input_3
input_4
identity?
PartitionedCallPartitionedCallinput_3input_4*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__wrapped_model_640062
PartitionedCall{
IdentityIdentityPartitionedCall:output:0*
T0*6
_output_shapes$
": ????????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D: ????????????: ????????????:_ [
6
_output_shapes$
": ????????????
!
_user_specified_name	input_3:_[
6
_output_shapes$
": ????????????
!
_user_specified_name	input_4
?'
p
F__inference_transformer_layer_call_and_return_conditional_losses_64338

inputs
inputs_1
identity{
Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"?????   ?   ?      2
Reshape/shape~
ReshapeReshapeinputsReshape/shape:output:0*
T0*6
_output_shapes$
": ????????????2	
Reshape
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"?????   ?   ?      2
Reshape_1/shape?
	Reshape_1Reshapeinputs_1Reshape_1/shape:output:0*
T0*6
_output_shapes$
": ????????????2
	Reshape_1V
	map/ShapeShapeReshape:output:0*
T0*
_output_shapes
:2
	map/Shape|
map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
map/strided_slice/stack?
map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
map/strided_slice/stack_1?
map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
map/strided_slice/stack_2?
map/strided_sliceStridedSlicemap/Shape:output:0 map/strided_slice/stack:output:0"map/strided_slice/stack_1:output:0"map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
map/strided_slice?
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
map/TensorArrayV2/element_shape?
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2?
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!map/TensorArrayV2_1/element_shape?
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2_1?
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      2;
9map/TensorArrayUnstack/TensorListFromTensor/element_shape?
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReshape:output:0Bmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+map/TensorArrayUnstack/TensorListFromTensor?
;map/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      2=
;map/TensorArrayUnstack_1/TensorListFromTensor/element_shape?
-map/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorReshape_1:output:0Dmap/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-map/TensorArrayUnstack_1/TensorListFromTensorX
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
	map/Const?
!map/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!map/TensorArrayV2_2/element_shape?
map/TensorArrayV2_2TensorListReserve*map/TensorArrayV2_2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2_2r
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
map/while/loop_counter?
	map/whileStatelessWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_2:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0=map/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : * 
_read_only_resource_inputs
 * 
bodyR
map_while_body_64036* 
condR
map_while_cond_64035*!
output_shapes
: : : : : : : 2
	map/while?
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      26
4map/TensorArrayV2Stack/TensorListStack/element_shape?
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*6
_output_shapes$
": ????????????*
element_dtype02(
&map/TensorArrayV2Stack/TensorListStack?
IdentityIdentity/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*6
_output_shapes$
": ????????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D: ????????????: ????????????:^ Z
6
_output_shapes$
": ????????????
 
_user_specified_nameinputs:^Z
6
_output_shapes$
": ????????????
 
_user_specified_nameinputs
?
Q
'__inference_model_2_layer_call_fn_64367
input_3
input_4
identity?
PartitionedCallPartitionedCallinput_3input_4*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_643642
PartitionedCall{
IdentityIdentityPartitionedCall:output:0*
T0*6
_output_shapes$
": ????????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D: ????????????: ????????????:_ [
6
_output_shapes$
": ????????????
!
_user_specified_name	input_3:_[
6
_output_shapes$
": ????????????
!
_user_specified_name	input_4
?
l
B__inference_model_2_layer_call_and_return_conditional_losses_64348
input_3
input_4
identity?
transformer/PartitionedCallPartitionedCallinput_3input_4*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_transformer_layer_call_and_return_conditional_losses_643382
transformer/PartitionedCall?
IdentityIdentity$transformer/PartitionedCall:output:0*
T0*6
_output_shapes$
": ????????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D: ????????????: ????????????:_ [
6
_output_shapes$
": ????????????
!
_user_specified_name	input_3:_[
6
_output_shapes$
": ????????????
!
_user_specified_name	input_4
?
?
(model_2_transformer_map_while_cond_63703.
*model_2_transformer_map_while_loop_counter)
%model_2_transformer_map_strided_slice
placeholder
placeholder_1.
*less_model_2_transformer_map_strided_sliceE
Amodel_2_transformer_map_while_cond_63703___redundant_placeholder0E
Amodel_2_transformer_map_while_cond_63703___redundant_placeholder1
identity
n
LessLessplaceholder*less_model_2_transformer_map_strided_slice*
T0*
_output_shapes
: 2
Less?
Less_1Less*model_2_transformer_map_while_loop_counter%model_2_transformer_map_strided_slice*
T0*
_output_shapes
: 2
Less_1T

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: 2

LogicalAndQ
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
?
l
B__inference_model_2_layer_call_and_return_conditional_losses_64364

inputs
inputs_1
identity?
transformer/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_transformer_layer_call_and_return_conditional_losses_643382
transformer/PartitionedCall?
IdentityIdentity$transformer/PartitionedCall:output:0*
T0*6
_output_shapes$
": ????????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D: ????????????: ????????????:^ Z
6
_output_shapes$
": ????????????
 
_user_specified_nameinputs:^Z
6
_output_shapes$
": ????????????
 
_user_specified_nameinputs
?
?
map_while_cond_64035
map_while_loop_counter
map_strided_slice
placeholder
placeholder_1
less_map_strided_slice1
-map_while_cond_64035___redundant_placeholder01
-map_while_cond_64035___redundant_placeholder1
identity
Z
LessLessplaceholderless_map_strided_slice*
T0*
_output_shapes
: 2
Lessd
Less_1Lessmap_while_loop_countermap_strided_slice*
T0*
_output_shapes
: 2
Less_1T

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: 2

LogicalAndQ
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
?
?
 transformer_map_while_cond_64738&
"transformer_map_while_loop_counter!
transformer_map_strided_slice
placeholder
placeholder_1&
"less_transformer_map_strided_slice=
9transformer_map_while_cond_64738___redundant_placeholder0=
9transformer_map_while_cond_64738___redundant_placeholder1
identity
f
LessLessplaceholder"less_transformer_map_strided_slice*
T0*
_output_shapes
: 2
Less|
Less_1Less"transformer_map_while_loop_countertransformer_map_strided_slice*
T0*
_output_shapes
: 2
Less_1T

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: 2

LogicalAndQ
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
?
?
map_while_cond_65077
map_while_loop_counter
map_strided_slice
placeholder
placeholder_1
less_map_strided_slice1
-map_while_cond_65077___redundant_placeholder01
-map_while_cond_65077___redundant_placeholder1
identity
Z
LessLessplaceholderless_map_strided_slice*
T0*
_output_shapes
: 2
Lessd
Less_1Lessmap_while_loop_countermap_strided_slice*
T0*
_output_shapes
: 2
Less_1T

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: 2

LogicalAndQ
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
?
S
'__inference_model_2_layer_call_fn_65053
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_643762
PartitionedCall{
IdentityIdentityPartitionedCall:output:0*
T0*6
_output_shapes$
": ????????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D: ????????????: ????????????:` \
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/0:`\
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/1
?
t
__inference__traced_save_65410
file_prefix
savev2_const

identity_1??MergeV2Checkpoints?SaveV2?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_0cb64e57b78c46c8a9207afb46050c1a/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix^SaveV2"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identityv

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 
??
?
 transformer_map_while_body_64412&
"transformer_map_while_loop_counter!
transformer_map_strided_slice
placeholder
placeholder_1%
!transformer_map_strided_slice_1_0a
]tensorarrayv2read_tensorlistgetitem_transformer_map_tensorarrayunstack_tensorlistfromtensor_0e
atensorarrayv2read_1_tensorlistgetitem_transformer_map_tensorarrayunstack_1_tensorlistfromtensor_0
identity

identity_1

identity_2

identity_3#
transformer_map_strided_slice_1_
[tensorarrayv2read_tensorlistgetitem_transformer_map_tensorarrayunstack_tensorlistfromtensorc
_tensorarrayv2read_1_tensorlistgetitem_transformer_map_tensorarrayunstack_1_tensorlistfromtensor?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItem]tensorarrayv2read_tensorlistgetitem_transformer_map_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*)
_output_shapes
:???*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
3TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      25
3TensorArrayV2Read_1/TensorListGetItem/element_shape?
%TensorArrayV2Read_1/TensorListGetItemTensorListGetItematensorarrayv2read_1_tensorlistgetitem_transformer_map_tensorarrayunstack_1_tensorlistfromtensor_0placeholder<TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*)
_output_shapes
:???*
element_dtype02'
%TensorArrayV2Read_1/TensorListGetItem\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start]
range/limitConst*
_output_shapes
: *
dtype0*
value
B :?2
range/limit\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltav
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes	
:?2
range`
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range_1/starta
range_1/limitConst*
_output_shapes
: *
dtype0*
value
B :?2
range_1/limit`
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_1/delta?
range_1Rangerange_1/start:output:0range_1/limit:output:0range_1/delta:output:0*
_output_shapes	
:?2	
range_1`
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range_2/starta
range_2/limitConst*
_output_shapes
: *
dtype0*
value
B :?2
range_2/limit`
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_2/delta?
range_2Rangerange_2/start:output:0range_2/limit:output:0range_2/delta:output:0*
_output_shapes	
:?2	
range_2s
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape/shapes
ReshapeReshaperange:output:0Reshape/shape:output:0*
T0*#
_output_shapes
:?2	
Reshapew
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ????   2
Reshape_1/shape{
	Reshape_1Reshaperange_1:output:0Reshape_1/shape:output:0*
T0*#
_output_shapes
:?2
	Reshape_1w
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
Reshape_2/shape{
	Reshape_2Reshaperange_2:output:0Reshape_2/shape:output:0*
T0*#
_output_shapes
:?2
	Reshape_2O
SizeConst*
_output_shapes
: *
dtype0*
value
B :?2
SizeS
Size_1Const*
_output_shapes
: *
dtype0*
value
B :?2
Size_1S
Size_2Const*
_output_shapes
: *
dtype0*
value
B :?2
Size_2c
stackConst*
_output_shapes
:*
dtype0*!
valueB"   ?   ?   2
stackf
TileTileReshape:output:0stack:output:0*
T0*%
_output_shapes
:???2
Tileg
stack_1Const*
_output_shapes
:*
dtype0*!
valueB"?      ?   2	
stack_1n
Tile_1TileReshape_1:output:0stack_1:output:0*
T0*%
_output_shapes
:???2
Tile_1g
stack_2Const*
_output_shapes
:*
dtype0*!
valueB"?   ?      2	
stack_2n
Tile_2TileReshape_2:output:0stack_2:output:0*
T0*%
_output_shapes
:???2
Tile_2b
CastCastTile:output:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSlice,TensorArrayV2Read_1/TensorListGetItem:item:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slicee
addAddV2Cast:y:0strided_slice:output:0*
T0*%
_output_shapes
:???2
addh
Cast_1CastTile_1:output:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlice,TensorArrayV2Read_1/TensorListGetItem:item:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_1m
add_1AddV2
Cast_1:y:0strided_slice_1:output:0*
T0*%
_output_shapes
:???2
add_1h
Cast_2CastTile_2:output:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSlice,TensorArrayV2Read_1/TensorListGetItem:item:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_2m
add_2AddV2
Cast_2:y:0strided_slice_2:output:0*
T0*%
_output_shapes
:???2
add_2?
stack_3Packadd:z:0	add_1:z:0	add_2:z:0*
N*
T0*)
_output_shapes
:???*
axis?????????2	
stack_3]
FloorFloorstack_3:output:0*
T0*)
_output_shapes
:???2
Floor
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_3w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumstrided_slice_3:output:0 clip_by_value/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSlicestack_3:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_4{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimumstrided_slice_4:output:0"clip_by_value_1/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_1
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSlicestack_3:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_5{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimumstrided_slice_5:output:0"clip_by_value_2/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_2
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSlice	Floor:y:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_6{
clip_by_value_3/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_3/Minimum/y?
clip_by_value_3/MinimumMinimumstrided_slice_6:output:0"clip_by_value_3/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_3/Minimumk
clip_by_value_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_3/y?
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_3
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSlice	Floor:y:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_7{
clip_by_value_4/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_4/Minimum/y?
clip_by_value_4/MinimumMinimumstrided_slice_7:output:0"clip_by_value_4/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_4/Minimumk
clip_by_value_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_4/y?
clip_by_value_4Maximumclip_by_value_4/Minimum:z:0clip_by_value_4/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_4
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSlice	Floor:y:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_8{
clip_by_value_5/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_5/Minimum/y?
clip_by_value_5/MinimumMinimumstrided_slice_8:output:0"clip_by_value_5/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_5/Minimumk
clip_by_value_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_5/y?
clip_by_value_5Maximumclip_by_value_5/Minimum:z:0clip_by_value_5/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_5W
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_3/yn
add_3AddV2clip_by_value_3:z:0add_3/y:output:0*
T0*%
_output_shapes
:???2
add_3{
clip_by_value_6/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_6/Minimum/y?
clip_by_value_6/MinimumMinimum	add_3:z:0"clip_by_value_6/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_6/Minimumk
clip_by_value_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_6/y?
clip_by_value_6Maximumclip_by_value_6/Minimum:z:0clip_by_value_6/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_6W
add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_4/yn
add_4AddV2clip_by_value_4:z:0add_4/y:output:0*
T0*%
_output_shapes
:???2
add_4{
clip_by_value_7/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_7/Minimum/y?
clip_by_value_7/MinimumMinimum	add_4:z:0"clip_by_value_7/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_7/Minimumk
clip_by_value_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_7/y?
clip_by_value_7Maximumclip_by_value_7/Minimum:z:0clip_by_value_7/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_7W
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_5/yn
add_5AddV2clip_by_value_5:z:0add_5/y:output:0*
T0*%
_output_shapes
:???2
add_5{
clip_by_value_8/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_8/Minimum/y?
clip_by_value_8/MinimumMinimum	add_5:z:0"clip_by_value_8/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_8/Minimumk
clip_by_value_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_8/y?
clip_by_value_8Maximumclip_by_value_8/Minimum:z:0clip_by_value_8/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_8l
Cast_3Castclip_by_value_3:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_3l
Cast_4Castclip_by_value_4:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_4l
Cast_5Castclip_by_value_5:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_5l
Cast_6Castclip_by_value_6:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_6l
Cast_7Castclip_by_value_7:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_7l
Cast_8Castclip_by_value_8:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_8i
subSubclip_by_value_6:z:0clip_by_value:z:0*
T0*%
_output_shapes
:???2
subo
sub_1Subclip_by_value_7:z:0clip_by_value_1:z:0*
T0*%
_output_shapes
:???2
sub_1o
sub_2Subclip_by_value_8:z:0clip_by_value_2:z:0*
T0*%
_output_shapes
:???2
sub_2W
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_3/x`
sub_3Subsub_3/x:output:0sub:z:0*
T0*%
_output_shapes
:???2
sub_3W
sub_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_4/xb
sub_4Subsub_4/x:output:0	sub_1:z:0*
T0*%
_output_shapes
:???2
sub_4W
sub_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_5/xb
sub_5Subsub_5/x:output:0	sub_2:z:0*
T0*%
_output_shapes
:???2
sub_5Q
mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
mul/y]
mulMul
Cast_4:y:0mul/y:output:0*
T0*%
_output_shapes
:???2
mul\
add_6AddV2
Cast_5:y:0mul:z:0*
T0*%
_output_shapes
:???2
add_6V
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2	
mul_1/yc
mul_1Mul
Cast_3:y:0mul_1/y:output:0*
T0*%
_output_shapes
:???2
mul_1]
add_7AddV2	add_6:z:0	mul_1:z:0*
T0*%
_output_shapes
:???2
add_7s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_3/shape?
	Reshape_3Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_3/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_3`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Reshape_3:output:0	add_7:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2Y
mul_2Mulsub:z:0	sub_1:z:0*
T0*%
_output_shapes
:???2
mul_2[
mul_3Mul	mul_2:z:0	sub_2:z:0*
T0*%
_output_shapes
:???2
mul_3k
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim~

ExpandDims
ExpandDims	mul_3:z:0ExpandDims/dim:output:0*
T0*)
_output_shapes
:???2

ExpandDimsq
mul_4MulExpandDims:output:0GatherV2:output:0*
T0*)
_output_shapes
:???2
mul_4W
add_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2	
add_8/xh
add_8AddV2add_8/x:output:0	mul_4:z:0*
T0*)
_output_shapes
:???2
add_8U
mul_5/yConst*
_output_shapes
: *
dtype0*
value
B :?2	
mul_5/yc
mul_5Mul
Cast_4:y:0mul_5/y:output:0*
T0*%
_output_shapes
:???2
mul_5^
add_9AddV2
Cast_8:y:0	mul_5:z:0*
T0*%
_output_shapes
:???2
add_9V
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2	
mul_6/yc
mul_6Mul
Cast_3:y:0mul_6/y:output:0*
T0*%
_output_shapes
:???2
mul_6_
add_10AddV2	add_9:z:0	mul_6:z:0*
T0*%
_output_shapes
:???2
add_10s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_4/shape?
	Reshape_4Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_4/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_4d
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axis?

GatherV2_1GatherV2Reshape_4:output:0
add_10:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_1Y
mul_7Mulsub:z:0	sub_1:z:0*
T0*%
_output_shapes
:???2
mul_7[
mul_8Mul	mul_7:z:0	sub_5:z:0*
T0*%
_output_shapes
:???2
mul_8o
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_1/dim?
ExpandDims_1
ExpandDims	mul_8:z:0ExpandDims_1/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_1u
mul_9MulExpandDims_1:output:0GatherV2_1:output:0*
T0*)
_output_shapes
:???2
mul_9c
add_11AddV2	add_8:z:0	mul_9:z:0*
T0*)
_output_shapes
:???2
add_11W
mul_10/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_10/yf
mul_10Mul
Cast_7:y:0mul_10/y:output:0*
T0*%
_output_shapes
:???2
mul_10a
add_12AddV2
Cast_5:y:0
mul_10:z:0*
T0*%
_output_shapes
:???2
add_12X
mul_11/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_11/yf
mul_11Mul
Cast_3:y:0mul_11/y:output:0*
T0*%
_output_shapes
:???2
mul_11a
add_13AddV2
add_12:z:0
mul_11:z:0*
T0*%
_output_shapes
:???2
add_13s
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_5/shape?
	Reshape_5Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_5/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_5d
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_2/axis?

GatherV2_2GatherV2Reshape_5:output:0
add_13:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_2[
mul_12Mulsub:z:0	sub_4:z:0*
T0*%
_output_shapes
:???2
mul_12^
mul_13Mul
mul_12:z:0	sub_2:z:0*
T0*%
_output_shapes
:???2
mul_13o
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_2/dim?
ExpandDims_2
ExpandDims
mul_13:z:0ExpandDims_2/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_2w
mul_14MulExpandDims_2:output:0GatherV2_2:output:0*
T0*)
_output_shapes
:???2
mul_14e
add_14AddV2
add_11:z:0
mul_14:z:0*
T0*)
_output_shapes
:???2
add_14W
mul_15/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_15/yf
mul_15Mul
Cast_7:y:0mul_15/y:output:0*
T0*%
_output_shapes
:???2
mul_15a
add_15AddV2
Cast_8:y:0
mul_15:z:0*
T0*%
_output_shapes
:???2
add_15X
mul_16/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_16/yf
mul_16Mul
Cast_3:y:0mul_16/y:output:0*
T0*%
_output_shapes
:???2
mul_16a
add_16AddV2
add_15:z:0
mul_16:z:0*
T0*%
_output_shapes
:???2
add_16s
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_6/shape?
	Reshape_6Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_6/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_6d
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_3/axis?

GatherV2_3GatherV2Reshape_6:output:0
add_16:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_3[
mul_17Mulsub:z:0	sub_4:z:0*
T0*%
_output_shapes
:???2
mul_17^
mul_18Mul
mul_17:z:0	sub_5:z:0*
T0*%
_output_shapes
:???2
mul_18o
ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_3/dim?
ExpandDims_3
ExpandDims
mul_18:z:0ExpandDims_3/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_3w
mul_19MulExpandDims_3:output:0GatherV2_3:output:0*
T0*)
_output_shapes
:???2
mul_19e
add_17AddV2
add_14:z:0
mul_19:z:0*
T0*)
_output_shapes
:???2
add_17W
mul_20/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_20/yf
mul_20Mul
Cast_4:y:0mul_20/y:output:0*
T0*%
_output_shapes
:???2
mul_20a
add_18AddV2
Cast_5:y:0
mul_20:z:0*
T0*%
_output_shapes
:???2
add_18X
mul_21/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_21/yf
mul_21Mul
Cast_6:y:0mul_21/y:output:0*
T0*%
_output_shapes
:???2
mul_21a
add_19AddV2
add_18:z:0
mul_21:z:0*
T0*%
_output_shapes
:???2
add_19s
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_7/shape?
	Reshape_7Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_7/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_7d
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_4/axis?

GatherV2_4GatherV2Reshape_7:output:0
add_19:z:0GatherV2_4/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_4]
mul_22Mul	sub_3:z:0	sub_1:z:0*
T0*%
_output_shapes
:???2
mul_22^
mul_23Mul
mul_22:z:0	sub_2:z:0*
T0*%
_output_shapes
:???2
mul_23o
ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_4/dim?
ExpandDims_4
ExpandDims
mul_23:z:0ExpandDims_4/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_4w
mul_24MulExpandDims_4:output:0GatherV2_4:output:0*
T0*)
_output_shapes
:???2
mul_24e
add_20AddV2
add_17:z:0
mul_24:z:0*
T0*)
_output_shapes
:???2
add_20W
mul_25/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_25/yf
mul_25Mul
Cast_4:y:0mul_25/y:output:0*
T0*%
_output_shapes
:???2
mul_25a
add_21AddV2
Cast_8:y:0
mul_25:z:0*
T0*%
_output_shapes
:???2
add_21X
mul_26/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_26/yf
mul_26Mul
Cast_6:y:0mul_26/y:output:0*
T0*%
_output_shapes
:???2
mul_26a
add_22AddV2
add_21:z:0
mul_26:z:0*
T0*%
_output_shapes
:???2
add_22s
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_8/shape?
	Reshape_8Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_8/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_8d
GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_5/axis?

GatherV2_5GatherV2Reshape_8:output:0
add_22:z:0GatherV2_5/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_5]
mul_27Mul	sub_3:z:0	sub_1:z:0*
T0*%
_output_shapes
:???2
mul_27^
mul_28Mul
mul_27:z:0	sub_5:z:0*
T0*%
_output_shapes
:???2
mul_28o
ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_5/dim?
ExpandDims_5
ExpandDims
mul_28:z:0ExpandDims_5/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_5w
mul_29MulExpandDims_5:output:0GatherV2_5:output:0*
T0*)
_output_shapes
:???2
mul_29e
add_23AddV2
add_20:z:0
mul_29:z:0*
T0*)
_output_shapes
:???2
add_23W
mul_30/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_30/yf
mul_30Mul
Cast_7:y:0mul_30/y:output:0*
T0*%
_output_shapes
:???2
mul_30a
add_24AddV2
Cast_5:y:0
mul_30:z:0*
T0*%
_output_shapes
:???2
add_24X
mul_31/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_31/yf
mul_31Mul
Cast_6:y:0mul_31/y:output:0*
T0*%
_output_shapes
:???2
mul_31a
add_25AddV2
add_24:z:0
mul_31:z:0*
T0*%
_output_shapes
:???2
add_25s
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_9/shape?
	Reshape_9Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_9/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_9d
GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_6/axis?

GatherV2_6GatherV2Reshape_9:output:0
add_25:z:0GatherV2_6/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_6]
mul_32Mul	sub_3:z:0	sub_4:z:0*
T0*%
_output_shapes
:???2
mul_32^
mul_33Mul
mul_32:z:0	sub_2:z:0*
T0*%
_output_shapes
:???2
mul_33o
ExpandDims_6/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_6/dim?
ExpandDims_6
ExpandDims
mul_33:z:0ExpandDims_6/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_6w
mul_34MulExpandDims_6:output:0GatherV2_6:output:0*
T0*)
_output_shapes
:???2
mul_34e
add_26AddV2
add_23:z:0
mul_34:z:0*
T0*)
_output_shapes
:???2
add_26W
mul_35/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_35/yf
mul_35Mul
Cast_7:y:0mul_35/y:output:0*
T0*%
_output_shapes
:???2
mul_35a
add_27AddV2
Cast_8:y:0
mul_35:z:0*
T0*%
_output_shapes
:???2
add_27X
mul_36/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_36/yf
mul_36Mul
Cast_6:y:0mul_36/y:output:0*
T0*%
_output_shapes
:???2
mul_36a
add_28AddV2
add_27:z:0
mul_36:z:0*
T0*%
_output_shapes
:???2
add_28u
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_10/shape?

Reshape_10Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_10/shape:output:0*
T0*!
_output_shapes
:???2

Reshape_10d
GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_7/axis?

GatherV2_7GatherV2Reshape_10:output:0
add_28:z:0GatherV2_7/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_7]
mul_37Mul	sub_3:z:0	sub_4:z:0*
T0*%
_output_shapes
:???2
mul_37^
mul_38Mul
mul_37:z:0	sub_5:z:0*
T0*%
_output_shapes
:???2
mul_38o
ExpandDims_7/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_7/dim?
ExpandDims_7
ExpandDims
mul_38:z:0ExpandDims_7/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_7w
mul_39MulExpandDims_7:output:0GatherV2_7:output:0*
T0*)
_output_shapes
:???2
mul_39e
add_29AddV2
add_26:z:0
mul_39:z:0*
T0*)
_output_shapes
:???2
add_29?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder
add_29:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemV
add_30/yConst*
_output_shapes
: *
dtype0*
value	B :2

add_30/yZ
add_30AddV2placeholderadd_30/y:output:0*
T0*
_output_shapes
: 2
add_30V
add_31/yConst*
_output_shapes
: *
dtype0*
value	B :2

add_31/yq
add_31AddV2"transformer_map_while_loop_counteradd_31/y:output:0*
T0*
_output_shapes
: 2
add_31M
IdentityIdentity
add_31:z:0*
T0*
_output_shapes
: 2

Identityd

Identity_1Identitytransformer_map_strided_slice*
T0*
_output_shapes
: 2

Identity_1Q

Identity_2Identity
add_30:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"?
_tensorarrayv2read_1_tensorlistgetitem_transformer_map_tensorarrayunstack_1_tensorlistfromtensoratensorarrayv2read_1_tensorlistgetitem_transformer_map_tensorarrayunstack_1_tensorlistfromtensor_0"?
[tensorarrayv2read_tensorlistgetitem_transformer_map_tensorarrayunstack_tensorlistfromtensor]tensorarrayv2read_tensorlistgetitem_transformer_map_tensorarrayunstack_tensorlistfromtensor_0"D
transformer_map_strided_slice_1!transformer_map_strided_slice_1_0*!
_input_shapes
: : : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
S
'__inference_model_2_layer_call_fn_65047
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_643642
PartitionedCall{
IdentityIdentityPartitionedCall:output:0*
T0*6
_output_shapes$
": ????????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D: ????????????: ????????????:` \
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/0:`\
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/1
?.
n
B__inference_model_2_layer_call_and_return_conditional_losses_64714
inputs_0
inputs_1
identity?
transformer/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"?????   ?   ?      2
transformer/Reshape/shape?
transformer/ReshapeReshapeinputs_0"transformer/Reshape/shape:output:0*
T0*6
_output_shapes$
": ????????????2
transformer/Reshape?
transformer/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"?????   ?   ?      2
transformer/Reshape_1/shape?
transformer/Reshape_1Reshapeinputs_1$transformer/Reshape_1/shape:output:0*
T0*6
_output_shapes$
": ????????????2
transformer/Reshape_1z
transformer/map/ShapeShapetransformer/Reshape:output:0*
T0*
_output_shapes
:2
transformer/map/Shape?
#transformer/map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#transformer/map/strided_slice/stack?
%transformer/map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%transformer/map/strided_slice/stack_1?
%transformer/map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%transformer/map/strided_slice/stack_2?
transformer/map/strided_sliceStridedSlicetransformer/map/Shape:output:0,transformer/map/strided_slice/stack:output:0.transformer/map/strided_slice/stack_1:output:0.transformer/map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
transformer/map/strided_slice?
+transformer/map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+transformer/map/TensorArrayV2/element_shape?
transformer/map/TensorArrayV2TensorListReserve4transformer/map/TensorArrayV2/element_shape:output:0&transformer/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
transformer/map/TensorArrayV2?
-transformer/map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-transformer/map/TensorArrayV2_1/element_shape?
transformer/map/TensorArrayV2_1TensorListReserve6transformer/map/TensorArrayV2_1/element_shape:output:0&transformer/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
transformer/map/TensorArrayV2_1?
Etransformer/map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      2G
Etransformer/map/TensorArrayUnstack/TensorListFromTensor/element_shape?
7transformer/map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensortransformer/Reshape:output:0Ntransformer/map/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7transformer/map/TensorArrayUnstack/TensorListFromTensor?
Gtransformer/map/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      2I
Gtransformer/map/TensorArrayUnstack_1/TensorListFromTensor/element_shape?
9transformer/map/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortransformer/Reshape_1:output:0Ptransformer/map/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9transformer/map/TensorArrayUnstack_1/TensorListFromTensorp
transformer/map/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
transformer/map/Const?
-transformer/map/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-transformer/map/TensorArrayV2_2/element_shape?
transformer/map/TensorArrayV2_2TensorListReserve6transformer/map/TensorArrayV2_2/element_shape:output:0&transformer/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
transformer/map/TensorArrayV2_2?
"transformer/map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"transformer/map/while/loop_counter?
transformer/map/whileStatelessWhile+transformer/map/while/loop_counter:output:0&transformer/map/strided_slice:output:0transformer/map/Const:output:0(transformer/map/TensorArrayV2_2:handle:0&transformer/map/strided_slice:output:0Gtransformer/map/TensorArrayUnstack/TensorListFromTensor:output_handle:0Itransformer/map/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : * 
_read_only_resource_inputs
 *,
body$R"
 transformer_map_while_body_64412*,
cond$R"
 transformer_map_while_cond_64411*!
output_shapes
: : : : : : : 2
transformer/map/while?
@transformer/map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      2B
@transformer/map/TensorArrayV2Stack/TensorListStack/element_shape?
2transformer/map/TensorArrayV2Stack/TensorListStackTensorListStacktransformer/map/while:output:3Itransformer/map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*6
_output_shapes$
": ????????????*
element_dtype024
2transformer/map/TensorArrayV2Stack/TensorListStack?
IdentityIdentity;transformer/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*6
_output_shapes$
": ????????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D: ????????????: ????????????:` \
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/0:`\
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/1
??
?
map_while_body_64036
map_while_loop_counter
map_strided_slice
placeholder
placeholder_1
map_strided_slice_1_0U
Qtensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0Y
Utensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_0
identity

identity_1

identity_2

identity_3
map_strided_slice_1S
Otensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensorW
Stensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemQtensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*)
_output_shapes
:???*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
3TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      25
3TensorArrayV2Read_1/TensorListGetItem/element_shape?
%TensorArrayV2Read_1/TensorListGetItemTensorListGetItemUtensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_0placeholder<TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*)
_output_shapes
:???*
element_dtype02'
%TensorArrayV2Read_1/TensorListGetItem\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start]
range/limitConst*
_output_shapes
: *
dtype0*
value
B :?2
range/limit\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltav
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes	
:?2
range`
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range_1/starta
range_1/limitConst*
_output_shapes
: *
dtype0*
value
B :?2
range_1/limit`
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_1/delta?
range_1Rangerange_1/start:output:0range_1/limit:output:0range_1/delta:output:0*
_output_shapes	
:?2	
range_1`
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range_2/starta
range_2/limitConst*
_output_shapes
: *
dtype0*
value
B :?2
range_2/limit`
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_2/delta?
range_2Rangerange_2/start:output:0range_2/limit:output:0range_2/delta:output:0*
_output_shapes	
:?2	
range_2s
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape/shapes
ReshapeReshaperange:output:0Reshape/shape:output:0*
T0*#
_output_shapes
:?2	
Reshapew
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ????   2
Reshape_1/shape{
	Reshape_1Reshaperange_1:output:0Reshape_1/shape:output:0*
T0*#
_output_shapes
:?2
	Reshape_1w
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
Reshape_2/shape{
	Reshape_2Reshaperange_2:output:0Reshape_2/shape:output:0*
T0*#
_output_shapes
:?2
	Reshape_2O
SizeConst*
_output_shapes
: *
dtype0*
value
B :?2
SizeS
Size_1Const*
_output_shapes
: *
dtype0*
value
B :?2
Size_1S
Size_2Const*
_output_shapes
: *
dtype0*
value
B :?2
Size_2c
stackConst*
_output_shapes
:*
dtype0*!
valueB"   ?   ?   2
stackf
TileTileReshape:output:0stack:output:0*
T0*%
_output_shapes
:???2
Tileg
stack_1Const*
_output_shapes
:*
dtype0*!
valueB"?      ?   2	
stack_1n
Tile_1TileReshape_1:output:0stack_1:output:0*
T0*%
_output_shapes
:???2
Tile_1g
stack_2Const*
_output_shapes
:*
dtype0*!
valueB"?   ?      2	
stack_2n
Tile_2TileReshape_2:output:0stack_2:output:0*
T0*%
_output_shapes
:???2
Tile_2b
CastCastTile:output:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSlice,TensorArrayV2Read_1/TensorListGetItem:item:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slicee
addAddV2Cast:y:0strided_slice:output:0*
T0*%
_output_shapes
:???2
addh
Cast_1CastTile_1:output:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlice,TensorArrayV2Read_1/TensorListGetItem:item:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_1m
add_1AddV2
Cast_1:y:0strided_slice_1:output:0*
T0*%
_output_shapes
:???2
add_1h
Cast_2CastTile_2:output:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSlice,TensorArrayV2Read_1/TensorListGetItem:item:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_2m
add_2AddV2
Cast_2:y:0strided_slice_2:output:0*
T0*%
_output_shapes
:???2
add_2?
stack_3Packadd:z:0	add_1:z:0	add_2:z:0*
N*
T0*)
_output_shapes
:???*
axis?????????2	
stack_3]
FloorFloorstack_3:output:0*
T0*)
_output_shapes
:???2
Floor
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_3w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumstrided_slice_3:output:0 clip_by_value/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSlicestack_3:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_4{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimumstrided_slice_4:output:0"clip_by_value_1/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_1
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSlicestack_3:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_5{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimumstrided_slice_5:output:0"clip_by_value_2/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_2
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSlice	Floor:y:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_6{
clip_by_value_3/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_3/Minimum/y?
clip_by_value_3/MinimumMinimumstrided_slice_6:output:0"clip_by_value_3/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_3/Minimumk
clip_by_value_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_3/y?
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_3
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSlice	Floor:y:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_7{
clip_by_value_4/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_4/Minimum/y?
clip_by_value_4/MinimumMinimumstrided_slice_7:output:0"clip_by_value_4/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_4/Minimumk
clip_by_value_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_4/y?
clip_by_value_4Maximumclip_by_value_4/Minimum:z:0clip_by_value_4/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_4
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSlice	Floor:y:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_8{
clip_by_value_5/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_5/Minimum/y?
clip_by_value_5/MinimumMinimumstrided_slice_8:output:0"clip_by_value_5/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_5/Minimumk
clip_by_value_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_5/y?
clip_by_value_5Maximumclip_by_value_5/Minimum:z:0clip_by_value_5/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_5W
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_3/yn
add_3AddV2clip_by_value_3:z:0add_3/y:output:0*
T0*%
_output_shapes
:???2
add_3{
clip_by_value_6/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_6/Minimum/y?
clip_by_value_6/MinimumMinimum	add_3:z:0"clip_by_value_6/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_6/Minimumk
clip_by_value_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_6/y?
clip_by_value_6Maximumclip_by_value_6/Minimum:z:0clip_by_value_6/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_6W
add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_4/yn
add_4AddV2clip_by_value_4:z:0add_4/y:output:0*
T0*%
_output_shapes
:???2
add_4{
clip_by_value_7/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_7/Minimum/y?
clip_by_value_7/MinimumMinimum	add_4:z:0"clip_by_value_7/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_7/Minimumk
clip_by_value_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_7/y?
clip_by_value_7Maximumclip_by_value_7/Minimum:z:0clip_by_value_7/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_7W
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_5/yn
add_5AddV2clip_by_value_5:z:0add_5/y:output:0*
T0*%
_output_shapes
:???2
add_5{
clip_by_value_8/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_8/Minimum/y?
clip_by_value_8/MinimumMinimum	add_5:z:0"clip_by_value_8/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_8/Minimumk
clip_by_value_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_8/y?
clip_by_value_8Maximumclip_by_value_8/Minimum:z:0clip_by_value_8/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_8l
Cast_3Castclip_by_value_3:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_3l
Cast_4Castclip_by_value_4:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_4l
Cast_5Castclip_by_value_5:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_5l
Cast_6Castclip_by_value_6:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_6l
Cast_7Castclip_by_value_7:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_7l
Cast_8Castclip_by_value_8:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_8i
subSubclip_by_value_6:z:0clip_by_value:z:0*
T0*%
_output_shapes
:???2
subo
sub_1Subclip_by_value_7:z:0clip_by_value_1:z:0*
T0*%
_output_shapes
:???2
sub_1o
sub_2Subclip_by_value_8:z:0clip_by_value_2:z:0*
T0*%
_output_shapes
:???2
sub_2W
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_3/x`
sub_3Subsub_3/x:output:0sub:z:0*
T0*%
_output_shapes
:???2
sub_3W
sub_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_4/xb
sub_4Subsub_4/x:output:0	sub_1:z:0*
T0*%
_output_shapes
:???2
sub_4W
sub_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_5/xb
sub_5Subsub_5/x:output:0	sub_2:z:0*
T0*%
_output_shapes
:???2
sub_5Q
mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
mul/y]
mulMul
Cast_4:y:0mul/y:output:0*
T0*%
_output_shapes
:???2
mul\
add_6AddV2
Cast_5:y:0mul:z:0*
T0*%
_output_shapes
:???2
add_6V
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2	
mul_1/yc
mul_1Mul
Cast_3:y:0mul_1/y:output:0*
T0*%
_output_shapes
:???2
mul_1]
add_7AddV2	add_6:z:0	mul_1:z:0*
T0*%
_output_shapes
:???2
add_7s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_3/shape?
	Reshape_3Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_3/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_3`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Reshape_3:output:0	add_7:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2Y
mul_2Mulsub:z:0	sub_1:z:0*
T0*%
_output_shapes
:???2
mul_2[
mul_3Mul	mul_2:z:0	sub_2:z:0*
T0*%
_output_shapes
:???2
mul_3k
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim~

ExpandDims
ExpandDims	mul_3:z:0ExpandDims/dim:output:0*
T0*)
_output_shapes
:???2

ExpandDimsq
mul_4MulExpandDims:output:0GatherV2:output:0*
T0*)
_output_shapes
:???2
mul_4W
add_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2	
add_8/xh
add_8AddV2add_8/x:output:0	mul_4:z:0*
T0*)
_output_shapes
:???2
add_8U
mul_5/yConst*
_output_shapes
: *
dtype0*
value
B :?2	
mul_5/yc
mul_5Mul
Cast_4:y:0mul_5/y:output:0*
T0*%
_output_shapes
:???2
mul_5^
add_9AddV2
Cast_8:y:0	mul_5:z:0*
T0*%
_output_shapes
:???2
add_9V
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2	
mul_6/yc
mul_6Mul
Cast_3:y:0mul_6/y:output:0*
T0*%
_output_shapes
:???2
mul_6_
add_10AddV2	add_9:z:0	mul_6:z:0*
T0*%
_output_shapes
:???2
add_10s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_4/shape?
	Reshape_4Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_4/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_4d
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axis?

GatherV2_1GatherV2Reshape_4:output:0
add_10:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_1Y
mul_7Mulsub:z:0	sub_1:z:0*
T0*%
_output_shapes
:???2
mul_7[
mul_8Mul	mul_7:z:0	sub_5:z:0*
T0*%
_output_shapes
:???2
mul_8o
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_1/dim?
ExpandDims_1
ExpandDims	mul_8:z:0ExpandDims_1/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_1u
mul_9MulExpandDims_1:output:0GatherV2_1:output:0*
T0*)
_output_shapes
:???2
mul_9c
add_11AddV2	add_8:z:0	mul_9:z:0*
T0*)
_output_shapes
:???2
add_11W
mul_10/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_10/yf
mul_10Mul
Cast_7:y:0mul_10/y:output:0*
T0*%
_output_shapes
:???2
mul_10a
add_12AddV2
Cast_5:y:0
mul_10:z:0*
T0*%
_output_shapes
:???2
add_12X
mul_11/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_11/yf
mul_11Mul
Cast_3:y:0mul_11/y:output:0*
T0*%
_output_shapes
:???2
mul_11a
add_13AddV2
add_12:z:0
mul_11:z:0*
T0*%
_output_shapes
:???2
add_13s
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_5/shape?
	Reshape_5Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_5/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_5d
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_2/axis?

GatherV2_2GatherV2Reshape_5:output:0
add_13:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_2[
mul_12Mulsub:z:0	sub_4:z:0*
T0*%
_output_shapes
:???2
mul_12^
mul_13Mul
mul_12:z:0	sub_2:z:0*
T0*%
_output_shapes
:???2
mul_13o
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_2/dim?
ExpandDims_2
ExpandDims
mul_13:z:0ExpandDims_2/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_2w
mul_14MulExpandDims_2:output:0GatherV2_2:output:0*
T0*)
_output_shapes
:???2
mul_14e
add_14AddV2
add_11:z:0
mul_14:z:0*
T0*)
_output_shapes
:???2
add_14W
mul_15/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_15/yf
mul_15Mul
Cast_7:y:0mul_15/y:output:0*
T0*%
_output_shapes
:???2
mul_15a
add_15AddV2
Cast_8:y:0
mul_15:z:0*
T0*%
_output_shapes
:???2
add_15X
mul_16/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_16/yf
mul_16Mul
Cast_3:y:0mul_16/y:output:0*
T0*%
_output_shapes
:???2
mul_16a
add_16AddV2
add_15:z:0
mul_16:z:0*
T0*%
_output_shapes
:???2
add_16s
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_6/shape?
	Reshape_6Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_6/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_6d
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_3/axis?

GatherV2_3GatherV2Reshape_6:output:0
add_16:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_3[
mul_17Mulsub:z:0	sub_4:z:0*
T0*%
_output_shapes
:???2
mul_17^
mul_18Mul
mul_17:z:0	sub_5:z:0*
T0*%
_output_shapes
:???2
mul_18o
ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_3/dim?
ExpandDims_3
ExpandDims
mul_18:z:0ExpandDims_3/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_3w
mul_19MulExpandDims_3:output:0GatherV2_3:output:0*
T0*)
_output_shapes
:???2
mul_19e
add_17AddV2
add_14:z:0
mul_19:z:0*
T0*)
_output_shapes
:???2
add_17W
mul_20/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_20/yf
mul_20Mul
Cast_4:y:0mul_20/y:output:0*
T0*%
_output_shapes
:???2
mul_20a
add_18AddV2
Cast_5:y:0
mul_20:z:0*
T0*%
_output_shapes
:???2
add_18X
mul_21/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_21/yf
mul_21Mul
Cast_6:y:0mul_21/y:output:0*
T0*%
_output_shapes
:???2
mul_21a
add_19AddV2
add_18:z:0
mul_21:z:0*
T0*%
_output_shapes
:???2
add_19s
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_7/shape?
	Reshape_7Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_7/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_7d
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_4/axis?

GatherV2_4GatherV2Reshape_7:output:0
add_19:z:0GatherV2_4/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_4]
mul_22Mul	sub_3:z:0	sub_1:z:0*
T0*%
_output_shapes
:???2
mul_22^
mul_23Mul
mul_22:z:0	sub_2:z:0*
T0*%
_output_shapes
:???2
mul_23o
ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_4/dim?
ExpandDims_4
ExpandDims
mul_23:z:0ExpandDims_4/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_4w
mul_24MulExpandDims_4:output:0GatherV2_4:output:0*
T0*)
_output_shapes
:???2
mul_24e
add_20AddV2
add_17:z:0
mul_24:z:0*
T0*)
_output_shapes
:???2
add_20W
mul_25/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_25/yf
mul_25Mul
Cast_4:y:0mul_25/y:output:0*
T0*%
_output_shapes
:???2
mul_25a
add_21AddV2
Cast_8:y:0
mul_25:z:0*
T0*%
_output_shapes
:???2
add_21X
mul_26/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_26/yf
mul_26Mul
Cast_6:y:0mul_26/y:output:0*
T0*%
_output_shapes
:???2
mul_26a
add_22AddV2
add_21:z:0
mul_26:z:0*
T0*%
_output_shapes
:???2
add_22s
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_8/shape?
	Reshape_8Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_8/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_8d
GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_5/axis?

GatherV2_5GatherV2Reshape_8:output:0
add_22:z:0GatherV2_5/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_5]
mul_27Mul	sub_3:z:0	sub_1:z:0*
T0*%
_output_shapes
:???2
mul_27^
mul_28Mul
mul_27:z:0	sub_5:z:0*
T0*%
_output_shapes
:???2
mul_28o
ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_5/dim?
ExpandDims_5
ExpandDims
mul_28:z:0ExpandDims_5/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_5w
mul_29MulExpandDims_5:output:0GatherV2_5:output:0*
T0*)
_output_shapes
:???2
mul_29e
add_23AddV2
add_20:z:0
mul_29:z:0*
T0*)
_output_shapes
:???2
add_23W
mul_30/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_30/yf
mul_30Mul
Cast_7:y:0mul_30/y:output:0*
T0*%
_output_shapes
:???2
mul_30a
add_24AddV2
Cast_5:y:0
mul_30:z:0*
T0*%
_output_shapes
:???2
add_24X
mul_31/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_31/yf
mul_31Mul
Cast_6:y:0mul_31/y:output:0*
T0*%
_output_shapes
:???2
mul_31a
add_25AddV2
add_24:z:0
mul_31:z:0*
T0*%
_output_shapes
:???2
add_25s
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_9/shape?
	Reshape_9Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_9/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_9d
GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_6/axis?

GatherV2_6GatherV2Reshape_9:output:0
add_25:z:0GatherV2_6/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_6]
mul_32Mul	sub_3:z:0	sub_4:z:0*
T0*%
_output_shapes
:???2
mul_32^
mul_33Mul
mul_32:z:0	sub_2:z:0*
T0*%
_output_shapes
:???2
mul_33o
ExpandDims_6/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_6/dim?
ExpandDims_6
ExpandDims
mul_33:z:0ExpandDims_6/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_6w
mul_34MulExpandDims_6:output:0GatherV2_6:output:0*
T0*)
_output_shapes
:???2
mul_34e
add_26AddV2
add_23:z:0
mul_34:z:0*
T0*)
_output_shapes
:???2
add_26W
mul_35/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_35/yf
mul_35Mul
Cast_7:y:0mul_35/y:output:0*
T0*%
_output_shapes
:???2
mul_35a
add_27AddV2
Cast_8:y:0
mul_35:z:0*
T0*%
_output_shapes
:???2
add_27X
mul_36/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_36/yf
mul_36Mul
Cast_6:y:0mul_36/y:output:0*
T0*%
_output_shapes
:???2
mul_36a
add_28AddV2
add_27:z:0
mul_36:z:0*
T0*%
_output_shapes
:???2
add_28u
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_10/shape?

Reshape_10Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_10/shape:output:0*
T0*!
_output_shapes
:???2

Reshape_10d
GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_7/axis?

GatherV2_7GatherV2Reshape_10:output:0
add_28:z:0GatherV2_7/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_7]
mul_37Mul	sub_3:z:0	sub_4:z:0*
T0*%
_output_shapes
:???2
mul_37^
mul_38Mul
mul_37:z:0	sub_5:z:0*
T0*%
_output_shapes
:???2
mul_38o
ExpandDims_7/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_7/dim?
ExpandDims_7
ExpandDims
mul_38:z:0ExpandDims_7/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_7w
mul_39MulExpandDims_7:output:0GatherV2_7:output:0*
T0*)
_output_shapes
:???2
mul_39e
add_29AddV2
add_26:z:0
mul_39:z:0*
T0*)
_output_shapes
:???2
add_29?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder
add_29:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemV
add_30/yConst*
_output_shapes
: *
dtype0*
value	B :2

add_30/yZ
add_30AddV2placeholderadd_30/y:output:0*
T0*
_output_shapes
: 2
add_30V
add_31/yConst*
_output_shapes
: *
dtype0*
value	B :2

add_31/ye
add_31AddV2map_while_loop_counteradd_31/y:output:0*
T0*
_output_shapes
: 2
add_31M
IdentityIdentity
add_31:z:0*
T0*
_output_shapes
: 2

IdentityX

Identity_1Identitymap_strided_slice*
T0*
_output_shapes
: 2

Identity_1Q

Identity_2Identity
add_30:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0",
map_strided_slice_1map_strided_slice_1_0"?
Stensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensorUtensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_0"?
Otensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensorQtensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*!
_input_shapes
: : : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
(model_2_transformer_map_while_body_63704.
*model_2_transformer_map_while_loop_counter)
%model_2_transformer_map_strided_slice
placeholder
placeholder_1-
)model_2_transformer_map_strided_slice_1_0i
etensorarrayv2read_tensorlistgetitem_model_2_transformer_map_tensorarrayunstack_tensorlistfromtensor_0m
itensorarrayv2read_1_tensorlistgetitem_model_2_transformer_map_tensorarrayunstack_1_tensorlistfromtensor_0
identity

identity_1

identity_2

identity_3+
'model_2_transformer_map_strided_slice_1g
ctensorarrayv2read_tensorlistgetitem_model_2_transformer_map_tensorarrayunstack_tensorlistfromtensork
gtensorarrayv2read_1_tensorlistgetitem_model_2_transformer_map_tensorarrayunstack_1_tensorlistfromtensor?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemetensorarrayv2read_tensorlistgetitem_model_2_transformer_map_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*)
_output_shapes
:???*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
3TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      25
3TensorArrayV2Read_1/TensorListGetItem/element_shape?
%TensorArrayV2Read_1/TensorListGetItemTensorListGetItemitensorarrayv2read_1_tensorlistgetitem_model_2_transformer_map_tensorarrayunstack_1_tensorlistfromtensor_0placeholder<TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*)
_output_shapes
:???*
element_dtype02'
%TensorArrayV2Read_1/TensorListGetItem\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start]
range/limitConst*
_output_shapes
: *
dtype0*
value
B :?2
range/limit\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltav
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes	
:?2
range`
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range_1/starta
range_1/limitConst*
_output_shapes
: *
dtype0*
value
B :?2
range_1/limit`
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_1/delta?
range_1Rangerange_1/start:output:0range_1/limit:output:0range_1/delta:output:0*
_output_shapes	
:?2	
range_1`
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range_2/starta
range_2/limitConst*
_output_shapes
: *
dtype0*
value
B :?2
range_2/limit`
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_2/delta?
range_2Rangerange_2/start:output:0range_2/limit:output:0range_2/delta:output:0*
_output_shapes	
:?2	
range_2s
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape/shapes
ReshapeReshaperange:output:0Reshape/shape:output:0*
T0*#
_output_shapes
:?2	
Reshapew
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ????   2
Reshape_1/shape{
	Reshape_1Reshaperange_1:output:0Reshape_1/shape:output:0*
T0*#
_output_shapes
:?2
	Reshape_1w
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
Reshape_2/shape{
	Reshape_2Reshaperange_2:output:0Reshape_2/shape:output:0*
T0*#
_output_shapes
:?2
	Reshape_2O
SizeConst*
_output_shapes
: *
dtype0*
value
B :?2
SizeS
Size_1Const*
_output_shapes
: *
dtype0*
value
B :?2
Size_1S
Size_2Const*
_output_shapes
: *
dtype0*
value
B :?2
Size_2c
stackConst*
_output_shapes
:*
dtype0*!
valueB"   ?   ?   2
stackf
TileTileReshape:output:0stack:output:0*
T0*%
_output_shapes
:???2
Tileg
stack_1Const*
_output_shapes
:*
dtype0*!
valueB"?      ?   2	
stack_1n
Tile_1TileReshape_1:output:0stack_1:output:0*
T0*%
_output_shapes
:???2
Tile_1g
stack_2Const*
_output_shapes
:*
dtype0*!
valueB"?   ?      2	
stack_2n
Tile_2TileReshape_2:output:0stack_2:output:0*
T0*%
_output_shapes
:???2
Tile_2b
CastCastTile:output:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSlice,TensorArrayV2Read_1/TensorListGetItem:item:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slicee
addAddV2Cast:y:0strided_slice:output:0*
T0*%
_output_shapes
:???2
addh
Cast_1CastTile_1:output:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlice,TensorArrayV2Read_1/TensorListGetItem:item:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_1m
add_1AddV2
Cast_1:y:0strided_slice_1:output:0*
T0*%
_output_shapes
:???2
add_1h
Cast_2CastTile_2:output:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSlice,TensorArrayV2Read_1/TensorListGetItem:item:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_2m
add_2AddV2
Cast_2:y:0strided_slice_2:output:0*
T0*%
_output_shapes
:???2
add_2?
stack_3Packadd:z:0	add_1:z:0	add_2:z:0*
N*
T0*)
_output_shapes
:???*
axis?????????2	
stack_3]
FloorFloorstack_3:output:0*
T0*)
_output_shapes
:???2
Floor
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_3w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumstrided_slice_3:output:0 clip_by_value/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSlicestack_3:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_4{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimumstrided_slice_4:output:0"clip_by_value_1/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_1
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSlicestack_3:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_5{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimumstrided_slice_5:output:0"clip_by_value_2/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_2
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSlice	Floor:y:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_6{
clip_by_value_3/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_3/Minimum/y?
clip_by_value_3/MinimumMinimumstrided_slice_6:output:0"clip_by_value_3/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_3/Minimumk
clip_by_value_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_3/y?
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_3
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSlice	Floor:y:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_7{
clip_by_value_4/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_4/Minimum/y?
clip_by_value_4/MinimumMinimumstrided_slice_7:output:0"clip_by_value_4/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_4/Minimumk
clip_by_value_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_4/y?
clip_by_value_4Maximumclip_by_value_4/Minimum:z:0clip_by_value_4/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_4
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSlice	Floor:y:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*%
_output_shapes
:???*
ellipsis_mask*
shrink_axis_mask2
strided_slice_8{
clip_by_value_5/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_5/Minimum/y?
clip_by_value_5/MinimumMinimumstrided_slice_8:output:0"clip_by_value_5/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_5/Minimumk
clip_by_value_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_5/y?
clip_by_value_5Maximumclip_by_value_5/Minimum:z:0clip_by_value_5/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_5W
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_3/yn
add_3AddV2clip_by_value_3:z:0add_3/y:output:0*
T0*%
_output_shapes
:???2
add_3{
clip_by_value_6/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_6/Minimum/y?
clip_by_value_6/MinimumMinimum	add_3:z:0"clip_by_value_6/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_6/Minimumk
clip_by_value_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_6/y?
clip_by_value_6Maximumclip_by_value_6/Minimum:z:0clip_by_value_6/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_6W
add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_4/yn
add_4AddV2clip_by_value_4:z:0add_4/y:output:0*
T0*%
_output_shapes
:???2
add_4{
clip_by_value_7/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_7/Minimum/y?
clip_by_value_7/MinimumMinimum	add_4:z:0"clip_by_value_7/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_7/Minimumk
clip_by_value_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_7/y?
clip_by_value_7Maximumclip_by_value_7/Minimum:z:0clip_by_value_7/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_7W
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_5/yn
add_5AddV2clip_by_value_5:z:0add_5/y:output:0*
T0*%
_output_shapes
:???2
add_5{
clip_by_value_8/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?C2
clip_by_value_8/Minimum/y?
clip_by_value_8/MinimumMinimum	add_5:z:0"clip_by_value_8/Minimum/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_8/Minimumk
clip_by_value_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_8/y?
clip_by_value_8Maximumclip_by_value_8/Minimum:z:0clip_by_value_8/y:output:0*
T0*%
_output_shapes
:???2
clip_by_value_8l
Cast_3Castclip_by_value_3:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_3l
Cast_4Castclip_by_value_4:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_4l
Cast_5Castclip_by_value_5:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_5l
Cast_6Castclip_by_value_6:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_6l
Cast_7Castclip_by_value_7:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_7l
Cast_8Castclip_by_value_8:z:0*

DstT0*

SrcT0*%
_output_shapes
:???2
Cast_8i
subSubclip_by_value_6:z:0clip_by_value:z:0*
T0*%
_output_shapes
:???2
subo
sub_1Subclip_by_value_7:z:0clip_by_value_1:z:0*
T0*%
_output_shapes
:???2
sub_1o
sub_2Subclip_by_value_8:z:0clip_by_value_2:z:0*
T0*%
_output_shapes
:???2
sub_2W
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_3/x`
sub_3Subsub_3/x:output:0sub:z:0*
T0*%
_output_shapes
:???2
sub_3W
sub_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_4/xb
sub_4Subsub_4/x:output:0	sub_1:z:0*
T0*%
_output_shapes
:???2
sub_4W
sub_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_5/xb
sub_5Subsub_5/x:output:0	sub_2:z:0*
T0*%
_output_shapes
:???2
sub_5Q
mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
mul/y]
mulMul
Cast_4:y:0mul/y:output:0*
T0*%
_output_shapes
:???2
mul\
add_6AddV2
Cast_5:y:0mul:z:0*
T0*%
_output_shapes
:???2
add_6V
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2	
mul_1/yc
mul_1Mul
Cast_3:y:0mul_1/y:output:0*
T0*%
_output_shapes
:???2
mul_1]
add_7AddV2	add_6:z:0	mul_1:z:0*
T0*%
_output_shapes
:???2
add_7s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_3/shape?
	Reshape_3Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_3/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_3`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Reshape_3:output:0	add_7:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2Y
mul_2Mulsub:z:0	sub_1:z:0*
T0*%
_output_shapes
:???2
mul_2[
mul_3Mul	mul_2:z:0	sub_2:z:0*
T0*%
_output_shapes
:???2
mul_3k
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim~

ExpandDims
ExpandDims	mul_3:z:0ExpandDims/dim:output:0*
T0*)
_output_shapes
:???2

ExpandDimsq
mul_4MulExpandDims:output:0GatherV2:output:0*
T0*)
_output_shapes
:???2
mul_4W
add_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2	
add_8/xh
add_8AddV2add_8/x:output:0	mul_4:z:0*
T0*)
_output_shapes
:???2
add_8U
mul_5/yConst*
_output_shapes
: *
dtype0*
value
B :?2	
mul_5/yc
mul_5Mul
Cast_4:y:0mul_5/y:output:0*
T0*%
_output_shapes
:???2
mul_5^
add_9AddV2
Cast_8:y:0	mul_5:z:0*
T0*%
_output_shapes
:???2
add_9V
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2	
mul_6/yc
mul_6Mul
Cast_3:y:0mul_6/y:output:0*
T0*%
_output_shapes
:???2
mul_6_
add_10AddV2	add_9:z:0	mul_6:z:0*
T0*%
_output_shapes
:???2
add_10s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_4/shape?
	Reshape_4Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_4/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_4d
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axis?

GatherV2_1GatherV2Reshape_4:output:0
add_10:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_1Y
mul_7Mulsub:z:0	sub_1:z:0*
T0*%
_output_shapes
:???2
mul_7[
mul_8Mul	mul_7:z:0	sub_5:z:0*
T0*%
_output_shapes
:???2
mul_8o
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_1/dim?
ExpandDims_1
ExpandDims	mul_8:z:0ExpandDims_1/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_1u
mul_9MulExpandDims_1:output:0GatherV2_1:output:0*
T0*)
_output_shapes
:???2
mul_9c
add_11AddV2	add_8:z:0	mul_9:z:0*
T0*)
_output_shapes
:???2
add_11W
mul_10/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_10/yf
mul_10Mul
Cast_7:y:0mul_10/y:output:0*
T0*%
_output_shapes
:???2
mul_10a
add_12AddV2
Cast_5:y:0
mul_10:z:0*
T0*%
_output_shapes
:???2
add_12X
mul_11/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_11/yf
mul_11Mul
Cast_3:y:0mul_11/y:output:0*
T0*%
_output_shapes
:???2
mul_11a
add_13AddV2
add_12:z:0
mul_11:z:0*
T0*%
_output_shapes
:???2
add_13s
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_5/shape?
	Reshape_5Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_5/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_5d
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_2/axis?

GatherV2_2GatherV2Reshape_5:output:0
add_13:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_2[
mul_12Mulsub:z:0	sub_4:z:0*
T0*%
_output_shapes
:???2
mul_12^
mul_13Mul
mul_12:z:0	sub_2:z:0*
T0*%
_output_shapes
:???2
mul_13o
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_2/dim?
ExpandDims_2
ExpandDims
mul_13:z:0ExpandDims_2/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_2w
mul_14MulExpandDims_2:output:0GatherV2_2:output:0*
T0*)
_output_shapes
:???2
mul_14e
add_14AddV2
add_11:z:0
mul_14:z:0*
T0*)
_output_shapes
:???2
add_14W
mul_15/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_15/yf
mul_15Mul
Cast_7:y:0mul_15/y:output:0*
T0*%
_output_shapes
:???2
mul_15a
add_15AddV2
Cast_8:y:0
mul_15:z:0*
T0*%
_output_shapes
:???2
add_15X
mul_16/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_16/yf
mul_16Mul
Cast_3:y:0mul_16/y:output:0*
T0*%
_output_shapes
:???2
mul_16a
add_16AddV2
add_15:z:0
mul_16:z:0*
T0*%
_output_shapes
:???2
add_16s
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_6/shape?
	Reshape_6Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_6/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_6d
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_3/axis?

GatherV2_3GatherV2Reshape_6:output:0
add_16:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_3[
mul_17Mulsub:z:0	sub_4:z:0*
T0*%
_output_shapes
:???2
mul_17^
mul_18Mul
mul_17:z:0	sub_5:z:0*
T0*%
_output_shapes
:???2
mul_18o
ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_3/dim?
ExpandDims_3
ExpandDims
mul_18:z:0ExpandDims_3/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_3w
mul_19MulExpandDims_3:output:0GatherV2_3:output:0*
T0*)
_output_shapes
:???2
mul_19e
add_17AddV2
add_14:z:0
mul_19:z:0*
T0*)
_output_shapes
:???2
add_17W
mul_20/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_20/yf
mul_20Mul
Cast_4:y:0mul_20/y:output:0*
T0*%
_output_shapes
:???2
mul_20a
add_18AddV2
Cast_5:y:0
mul_20:z:0*
T0*%
_output_shapes
:???2
add_18X
mul_21/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_21/yf
mul_21Mul
Cast_6:y:0mul_21/y:output:0*
T0*%
_output_shapes
:???2
mul_21a
add_19AddV2
add_18:z:0
mul_21:z:0*
T0*%
_output_shapes
:???2
add_19s
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_7/shape?
	Reshape_7Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_7/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_7d
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_4/axis?

GatherV2_4GatherV2Reshape_7:output:0
add_19:z:0GatherV2_4/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_4]
mul_22Mul	sub_3:z:0	sub_1:z:0*
T0*%
_output_shapes
:???2
mul_22^
mul_23Mul
mul_22:z:0	sub_2:z:0*
T0*%
_output_shapes
:???2
mul_23o
ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_4/dim?
ExpandDims_4
ExpandDims
mul_23:z:0ExpandDims_4/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_4w
mul_24MulExpandDims_4:output:0GatherV2_4:output:0*
T0*)
_output_shapes
:???2
mul_24e
add_20AddV2
add_17:z:0
mul_24:z:0*
T0*)
_output_shapes
:???2
add_20W
mul_25/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_25/yf
mul_25Mul
Cast_4:y:0mul_25/y:output:0*
T0*%
_output_shapes
:???2
mul_25a
add_21AddV2
Cast_8:y:0
mul_25:z:0*
T0*%
_output_shapes
:???2
add_21X
mul_26/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_26/yf
mul_26Mul
Cast_6:y:0mul_26/y:output:0*
T0*%
_output_shapes
:???2
mul_26a
add_22AddV2
add_21:z:0
mul_26:z:0*
T0*%
_output_shapes
:???2
add_22s
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_8/shape?
	Reshape_8Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_8/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_8d
GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_5/axis?

GatherV2_5GatherV2Reshape_8:output:0
add_22:z:0GatherV2_5/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_5]
mul_27Mul	sub_3:z:0	sub_1:z:0*
T0*%
_output_shapes
:???2
mul_27^
mul_28Mul
mul_27:z:0	sub_5:z:0*
T0*%
_output_shapes
:???2
mul_28o
ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_5/dim?
ExpandDims_5
ExpandDims
mul_28:z:0ExpandDims_5/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_5w
mul_29MulExpandDims_5:output:0GatherV2_5:output:0*
T0*)
_output_shapes
:???2
mul_29e
add_23AddV2
add_20:z:0
mul_29:z:0*
T0*)
_output_shapes
:???2
add_23W
mul_30/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_30/yf
mul_30Mul
Cast_7:y:0mul_30/y:output:0*
T0*%
_output_shapes
:???2
mul_30a
add_24AddV2
Cast_5:y:0
mul_30:z:0*
T0*%
_output_shapes
:???2
add_24X
mul_31/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_31/yf
mul_31Mul
Cast_6:y:0mul_31/y:output:0*
T0*%
_output_shapes
:???2
mul_31a
add_25AddV2
add_24:z:0
mul_31:z:0*
T0*%
_output_shapes
:???2
add_25s
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_9/shape?
	Reshape_9Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_9/shape:output:0*
T0*!
_output_shapes
:???2
	Reshape_9d
GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_6/axis?

GatherV2_6GatherV2Reshape_9:output:0
add_25:z:0GatherV2_6/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_6]
mul_32Mul	sub_3:z:0	sub_4:z:0*
T0*%
_output_shapes
:???2
mul_32^
mul_33Mul
mul_32:z:0	sub_2:z:0*
T0*%
_output_shapes
:???2
mul_33o
ExpandDims_6/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_6/dim?
ExpandDims_6
ExpandDims
mul_33:z:0ExpandDims_6/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_6w
mul_34MulExpandDims_6:output:0GatherV2_6:output:0*
T0*)
_output_shapes
:???2
mul_34e
add_26AddV2
add_23:z:0
mul_34:z:0*
T0*)
_output_shapes
:???2
add_26W
mul_35/yConst*
_output_shapes
: *
dtype0*
value
B :?2

mul_35/yf
mul_35Mul
Cast_7:y:0mul_35/y:output:0*
T0*%
_output_shapes
:???2
mul_35a
add_27AddV2
Cast_8:y:0
mul_35:z:0*
T0*%
_output_shapes
:???2
add_27X
mul_36/yConst*
_output_shapes
: *
dtype0*
valueB	 :??2

mul_36/yf
mul_36Mul
Cast_6:y:0mul_36/y:output:0*
T0*%
_output_shapes
:???2
mul_36a
add_28AddV2
add_27:z:0
mul_36:z:0*
T0*%
_output_shapes
:???2
add_28u
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_10/shape?

Reshape_10Reshape*TensorArrayV2Read/TensorListGetItem:item:0Reshape_10/shape:output:0*
T0*!
_output_shapes
:???2

Reshape_10d
GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_7/axis?

GatherV2_7GatherV2Reshape_10:output:0
add_28:z:0GatherV2_7/axis:output:0*
Taxis0*
Tindices0*
Tparams0*)
_output_shapes
:???2

GatherV2_7]
mul_37Mul	sub_3:z:0	sub_4:z:0*
T0*%
_output_shapes
:???2
mul_37^
mul_38Mul
mul_37:z:0	sub_5:z:0*
T0*%
_output_shapes
:???2
mul_38o
ExpandDims_7/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_7/dim?
ExpandDims_7
ExpandDims
mul_38:z:0ExpandDims_7/dim:output:0*
T0*)
_output_shapes
:???2
ExpandDims_7w
mul_39MulExpandDims_7:output:0GatherV2_7:output:0*
T0*)
_output_shapes
:???2
mul_39e
add_29AddV2
add_26:z:0
mul_39:z:0*
T0*)
_output_shapes
:???2
add_29?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder
add_29:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemV
add_30/yConst*
_output_shapes
: *
dtype0*
value	B :2

add_30/yZ
add_30AddV2placeholderadd_30/y:output:0*
T0*
_output_shapes
: 2
add_30V
add_31/yConst*
_output_shapes
: *
dtype0*
value	B :2

add_31/yy
add_31AddV2*model_2_transformer_map_while_loop_counteradd_31/y:output:0*
T0*
_output_shapes
: 2
add_31M
IdentityIdentity
add_31:z:0*
T0*
_output_shapes
: 2

Identityl

Identity_1Identity%model_2_transformer_map_strided_slice*
T0*
_output_shapes
: 2

Identity_1Q

Identity_2Identity
add_30:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"T
'model_2_transformer_map_strided_slice_1)model_2_transformer_map_strided_slice_1_0"?
gtensorarrayv2read_1_tensorlistgetitem_model_2_transformer_map_tensorarrayunstack_1_tensorlistfromtensoritensorarrayv2read_1_tensorlistgetitem_model_2_transformer_map_tensorarrayunstack_1_tensorlistfromtensor_0"?
ctensorarrayv2read_tensorlistgetitem_model_2_transformer_map_tensorarrayunstack_tensorlistfromtensoretensorarrayv2read_tensorlistgetitem_model_2_transformer_map_tensorarrayunstack_tensorlistfromtensor_0*!
_input_shapes
: : : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "?J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
J
input_3?
serving_default_input_3:0 ????????????
J
input_4?
serving_default_input_4:0 ????????????F
transformer7
PartitionedCall:0 ????????????tensorflow/serving/predict:?O
?
layer-0
layer-1
layer-2
trainable_variables
	variables
regularization_losses
	keras_api

signatures
__call__
_default_save_signature
*&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "Model", "name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 192, 192, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 192, 192, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "SpatialTransformer", "config": {"name": "transformer", "trainable": true, "dtype": "float32", "interp_method": "linear", "ndims": 3, "inshape": [{"class_name": "TensorShape", "items": [null, 192, 192, 192, 1]}, {"class_name": "TensorShape", "items": [null, 192, 192, 192, 3]}], "single_transform": false, "indexing": "ij"}, "name": "transformer", "inbound_nodes": [[["input_3", 0, 0, {}], ["input_4", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0], ["input_4", 0, 0]], "output_layers": [["transformer", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 192, 192, 192, 1]}, {"class_name": "TensorShape", "items": [null, 192, 192, 192, 3]}], "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 192, 192, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 192, 192, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "SpatialTransformer", "config": {"name": "transformer", "trainable": true, "dtype": "float32", "interp_method": "linear", "ndims": 3, "inshape": [{"class_name": "TensorShape", "items": [null, 192, 192, 192, 1]}, {"class_name": "TensorShape", "items": [null, 192, 192, 192, 3]}], "single_transform": false, "indexing": "ij"}, "name": "transformer", "inbound_nodes": [[["input_3", 0, 0, {}], ["input_4", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0], ["input_4", 0, 0]], "output_layers": [["transformer", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 192, 192, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 192, 192, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 192, 192, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 192, 192, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
?
	inshape

trainable_variables
	variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "SpatialTransformer", "name": "transformer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "transformer", "trainable": true, "dtype": "float32", "interp_method": "linear", "ndims": 3, "inshape": [{"class_name": "TensorShape", "items": [null, 192, 192, 192, 1]}, {"class_name": "TensorShape", "items": [null, 192, 192, 192, 3]}], "single_transform": false, "indexing": "ij"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 192, 192, 192, 1]}, {"class_name": "TensorShape", "items": [null, 192, 192, 192, 3]}]}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
layer_metrics
trainable_variables
	variables
layer_regularization_losses

layers
regularization_losses
metrics
non_trainable_variables
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
layer_metrics

trainable_variables
layer_regularization_losses
	variables

layers
regularization_losses
metrics
non_trainable_variables
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
'__inference_model_2_layer_call_fn_65053
'__inference_model_2_layer_call_fn_64379
'__inference_model_2_layer_call_fn_65047
'__inference_model_2_layer_call_fn_64367?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_64006?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *l?i
g?d
0?-
input_3 ????????????
0?-
input_4 ????????????
?2?
B__inference_model_2_layer_call_and_return_conditional_losses_64354
B__inference_model_2_layer_call_and_return_conditional_losses_64348
B__inference_model_2_layer_call_and_return_conditional_losses_65041
B__inference_model_2_layer_call_and_return_conditional_losses_64714?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_transformer_layer_call_fn_65386?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_transformer_layer_call_and_return_conditional_losses_65380?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
9B7
#__inference_signature_wrapper_64387input_3input_4?
 __inference__wrapped_model_64006?v?s
l?i
g?d
0?-
input_3 ????????????
0?-
input_4 ????????????
? "H?E
C
transformer4?1
transformer ?????????????
B__inference_model_2_layer_call_and_return_conditional_losses_64348?~?{
t?q
g?d
0?-
input_3 ????????????
0?-
input_4 ????????????
p

 
? "4?1
*?'
0 ????????????
? ?
B__inference_model_2_layer_call_and_return_conditional_losses_64354?~?{
t?q
g?d
0?-
input_3 ????????????
0?-
input_4 ????????????
p 

 
? "4?1
*?'
0 ????????????
? ?
B__inference_model_2_layer_call_and_return_conditional_losses_64714???}
v?s
i?f
1?.
inputs/0 ????????????
1?.
inputs/1 ????????????
p

 
? "4?1
*?'
0 ????????????
? ?
B__inference_model_2_layer_call_and_return_conditional_losses_65041???}
v?s
i?f
1?.
inputs/0 ????????????
1?.
inputs/1 ????????????
p 

 
? "4?1
*?'
0 ????????????
? ?
'__inference_model_2_layer_call_fn_64367?~?{
t?q
g?d
0?-
input_3 ????????????
0?-
input_4 ????????????
p

 
? "'?$ ?????????????
'__inference_model_2_layer_call_fn_64379?~?{
t?q
g?d
0?-
input_3 ????????????
0?-
input_4 ????????????
p 

 
? "'?$ ?????????????
'__inference_model_2_layer_call_fn_65047???}
v?s
i?f
1?.
inputs/0 ????????????
1?.
inputs/1 ????????????
p

 
? "'?$ ?????????????
'__inference_model_2_layer_call_fn_65053???}
v?s
i?f
1?.
inputs/0 ????????????
1?.
inputs/1 ????????????
p 

 
? "'?$ ?????????????
#__inference_signature_wrapper_64387????
? 
}?z
;
input_30?-
input_3 ????????????
;
input_40?-
input_4 ????????????"H?E
C
transformer4?1
transformer ?????????????
F__inference_transformer_layer_call_and_return_conditional_losses_65380?x?u
n?k
i?f
1?.
inputs/0 ????????????
1?.
inputs/1 ????????????
? "4?1
*?'
0 ????????????
? ?
+__inference_transformer_layer_call_fn_65386?x?u
n?k
i?f
1?.
inputs/0 ????????????
1?.
inputs/1 ????????????
? "'?$ ????????????