??3
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
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
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8??,
?
conv3d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3d/kernel
{
!conv3d/kernel/Read/ReadVariableOpReadVariableOpconv3d/kernel**
_output_shapes
: *
dtype0
n
conv3d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3d/bias
g
conv3d/bias/Read/ReadVariableOpReadVariableOpconv3d/bias*
_output_shapes
: *
dtype0
?
conv3d_1_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*#
shared_nameconv3d_1_13/kernel
?
&conv3d_1_13/kernel/Read/ReadVariableOpReadVariableOpconv3d_1_13/kernel**
_output_shapes
: @*
dtype0
x
conv3d_1_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv3d_1_13/bias
q
$conv3d_1_13/bias/Read/ReadVariableOpReadVariableOpconv3d_1_13/bias*
_output_shapes
:@*
dtype0
?
conv3d_2_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*#
shared_nameconv3d_2_13/kernel
?
&conv3d_2_13/kernel/Read/ReadVariableOpReadVariableOpconv3d_2_13/kernel**
_output_shapes
:@@*
dtype0
x
conv3d_2_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv3d_2_13/bias
q
$conv3d_2_13/bias/Read/ReadVariableOpReadVariableOpconv3d_2_13/bias*
_output_shapes
:@*
dtype0
?
conv3d_3_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*#
shared_nameconv3d_3_13/kernel
?
&conv3d_3_13/kernel/Read/ReadVariableOpReadVariableOpconv3d_3_13/kernel**
_output_shapes
:@@*
dtype0
x
conv3d_3_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv3d_3_13/bias
q
$conv3d_3_13/bias/Read/ReadVariableOpReadVariableOpconv3d_3_13/bias*
_output_shapes
:@*
dtype0
?
conv3d_4_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*#
shared_nameconv3d_4_13/kernel
?
&conv3d_4_13/kernel/Read/ReadVariableOpReadVariableOpconv3d_4_13/kernel**
_output_shapes
:@@*
dtype0
x
conv3d_4_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv3d_4_13/bias
q
$conv3d_4_13/bias/Read/ReadVariableOpReadVariableOpconv3d_4_13/bias*
_output_shapes
:@*
dtype0
?
conv3d_5_13/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:?@*#
shared_nameconv3d_5_13/kernel
?
&conv3d_5_13/kernel/Read/ReadVariableOpReadVariableOpconv3d_5_13/kernel*+
_output_shapes
:?@*
dtype0
x
conv3d_5_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv3d_5_13/bias
q
$conv3d_5_13/bias/Read/ReadVariableOpReadVariableOpconv3d_5_13/bias*
_output_shapes
:@*
dtype0
?
conv3d_6_13/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:?@*#
shared_nameconv3d_6_13/kernel
?
&conv3d_6_13/kernel/Read/ReadVariableOpReadVariableOpconv3d_6_13/kernel*+
_output_shapes
:?@*
dtype0
x
conv3d_6_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv3d_6_13/bias
q
$conv3d_6_13/bias/Read/ReadVariableOpReadVariableOpconv3d_6_13/bias*
_output_shapes
:@*
dtype0
?
conv3d_7_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:`@*#
shared_nameconv3d_7_13/kernel
?
&conv3d_7_13/kernel/Read/ReadVariableOpReadVariableOpconv3d_7_13/kernel**
_output_shapes
:`@*
dtype0
x
conv3d_7_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv3d_7_13/bias
q
$conv3d_7_13/bias/Read/ReadVariableOpReadVariableOpconv3d_7_13/bias*
_output_shapes
:@*
dtype0
?
conv3d_8_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*#
shared_nameconv3d_8_13/kernel
?
&conv3d_8_13/kernel/Read/ReadVariableOpReadVariableOpconv3d_8_13/kernel**
_output_shapes
:@@*
dtype0
x
conv3d_8_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv3d_8_13/bias
q
$conv3d_8_13/bias/Read/ReadVariableOpReadVariableOpconv3d_8_13/bias*
_output_shapes
:@*
dtype0
?
conv3d_9_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:B *#
shared_nameconv3d_9_13/kernel
?
&conv3d_9_13/kernel/Read/ReadVariableOpReadVariableOpconv3d_9_13/kernel**
_output_shapes
:B *
dtype0
x
conv3d_9_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv3d_9_13/bias
q
$conv3d_9_13/bias/Read/ReadVariableOpReadVariableOpconv3d_9_13/bias*
_output_shapes
: *
dtype0
?
conv3d_10_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *$
shared_nameconv3d_10_13/kernel
?
'conv3d_10_13/kernel/Read/ReadVariableOpReadVariableOpconv3d_10_13/kernel**
_output_shapes
:  *
dtype0
z
conv3d_10_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv3d_10_13/bias
s
%conv3d_10_13/bias/Read/ReadVariableOpReadVariableOpconv3d_10_13/bias*
_output_shapes
: *
dtype0
?
disp_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedisp_10/kernel
}
"disp_10/kernel/Read/ReadVariableOpReadVariableOpdisp_10/kernel**
_output_shapes
: *
dtype0
p
disp_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedisp_10/bias
i
 disp_10/bias/Read/ReadVariableOpReadVariableOpdisp_10/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?n
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?m
value?mB?m B?m
?
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer_with_weights-4
layer-11
layer-12
layer-13
layer-14
layer_with_weights-5
layer-15
layer-16
layer-17
layer-18
layer_with_weights-6
layer-19
layer-20
layer-21
layer-22
layer_with_weights-7
layer-23
layer-24
layer_with_weights-8
layer-25
layer-26
layer-27
layer-28
layer_with_weights-9
layer-29
layer-30
 layer_with_weights-10
 layer-31
!layer-32
"layer_with_weights-11
"layer-33
#layer-34
$trainable_variables
%	variables
&regularization_losses
'	keras_api
(
signatures
 
 
R
)trainable_variables
*	variables
+regularization_losses
,	keras_api
h

-kernel
.bias
/trainable_variables
0	variables
1regularization_losses
2	keras_api
R
3trainable_variables
4	variables
5regularization_losses
6	keras_api
h

7kernel
8bias
9trainable_variables
:	variables
;regularization_losses
<	keras_api
R
=trainable_variables
>	variables
?regularization_losses
@	keras_api
h

Akernel
Bbias
Ctrainable_variables
D	variables
Eregularization_losses
F	keras_api
R
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
h

Kkernel
Lbias
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
R
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
h

Ukernel
Vbias
Wtrainable_variables
X	variables
Yregularization_losses
Z	keras_api
R
[trainable_variables
\	variables
]regularization_losses
^	keras_api
R
_trainable_variables
`	variables
aregularization_losses
b	keras_api
R
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
h

gkernel
hbias
itrainable_variables
j	variables
kregularization_losses
l	keras_api
R
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
R
qtrainable_variables
r	variables
sregularization_losses
t	keras_api
R
utrainable_variables
v	variables
wregularization_losses
x	keras_api
h

ykernel
zbias
{trainable_variables
|	variables
}regularization_losses
~	keras_api
U
trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
d
?inshape
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
-0
.1
72
83
A4
B5
K6
L7
U8
V9
g10
h11
y12
z13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?
-0
.1
72
83
A4
B5
K6
L7
U8
V9
g10
h11
y12
z13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
 
?
 ?layer_regularization_losses
$trainable_variables
?layer_metrics
?metrics
?layers
%	variables
?non_trainable_variables
&regularization_losses
 
 
 
 
?
 ?layer_regularization_losses
)trainable_variables
?layer_metrics
?metrics
?layers
*	variables
?non_trainable_variables
+regularization_losses
YW
VARIABLE_VALUEconv3d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv3d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1

-0
.1
 
?
 ?layer_regularization_losses
/trainable_variables
?layer_metrics
?metrics
?layers
0	variables
?non_trainable_variables
1regularization_losses
 
 
 
?
 ?layer_regularization_losses
3trainable_variables
?layer_metrics
?metrics
?layers
4	variables
?non_trainable_variables
5regularization_losses
^\
VARIABLE_VALUEconv3d_1_13/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv3d_1_13/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

70
81

70
81
 
?
 ?layer_regularization_losses
9trainable_variables
?layer_metrics
?metrics
?layers
:	variables
?non_trainable_variables
;regularization_losses
 
 
 
?
 ?layer_regularization_losses
=trainable_variables
?layer_metrics
?metrics
?layers
>	variables
?non_trainable_variables
?regularization_losses
^\
VARIABLE_VALUEconv3d_2_13/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv3d_2_13/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

A0
B1

A0
B1
 
?
 ?layer_regularization_losses
Ctrainable_variables
?layer_metrics
?metrics
?layers
D	variables
?non_trainable_variables
Eregularization_losses
 
 
 
?
 ?layer_regularization_losses
Gtrainable_variables
?layer_metrics
?metrics
?layers
H	variables
?non_trainable_variables
Iregularization_losses
^\
VARIABLE_VALUEconv3d_3_13/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv3d_3_13/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

K0
L1

K0
L1
 
?
 ?layer_regularization_losses
Mtrainable_variables
?layer_metrics
?metrics
?layers
N	variables
?non_trainable_variables
Oregularization_losses
 
 
 
?
 ?layer_regularization_losses
Qtrainable_variables
?layer_metrics
?metrics
?layers
R	variables
?non_trainable_variables
Sregularization_losses
^\
VARIABLE_VALUEconv3d_4_13/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv3d_4_13/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1

U0
V1
 
?
 ?layer_regularization_losses
Wtrainable_variables
?layer_metrics
?metrics
?layers
X	variables
?non_trainable_variables
Yregularization_losses
 
 
 
?
 ?layer_regularization_losses
[trainable_variables
?layer_metrics
?metrics
?layers
\	variables
?non_trainable_variables
]regularization_losses
 
 
 
?
 ?layer_regularization_losses
_trainable_variables
?layer_metrics
?metrics
?layers
`	variables
?non_trainable_variables
aregularization_losses
 
 
 
?
 ?layer_regularization_losses
ctrainable_variables
?layer_metrics
?metrics
?layers
d	variables
?non_trainable_variables
eregularization_losses
^\
VARIABLE_VALUEconv3d_5_13/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv3d_5_13/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

g0
h1

g0
h1
 
?
 ?layer_regularization_losses
itrainable_variables
?layer_metrics
?metrics
?layers
j	variables
?non_trainable_variables
kregularization_losses
 
 
 
?
 ?layer_regularization_losses
mtrainable_variables
?layer_metrics
?metrics
?layers
n	variables
?non_trainable_variables
oregularization_losses
 
 
 
?
 ?layer_regularization_losses
qtrainable_variables
?layer_metrics
?metrics
?layers
r	variables
?non_trainable_variables
sregularization_losses
 
 
 
?
 ?layer_regularization_losses
utrainable_variables
?layer_metrics
?metrics
?layers
v	variables
?non_trainable_variables
wregularization_losses
^\
VARIABLE_VALUEconv3d_6_13/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv3d_6_13/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

y0
z1

y0
z1
 
?
 ?layer_regularization_losses
{trainable_variables
?layer_metrics
?metrics
?layers
|	variables
?non_trainable_variables
}regularization_losses
 
 
 
?
 ?layer_regularization_losses
trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
^\
VARIABLE_VALUEconv3d_7_13/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv3d_7_13/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
^\
VARIABLE_VALUEconv3d_8_13/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv3d_8_13/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
^\
VARIABLE_VALUEconv3d_9_13/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv3d_9_13/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
`^
VARIABLE_VALUEconv3d_10_13/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconv3d_10_13/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
[Y
VARIABLE_VALUEdisp_10/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdisp_10/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
 
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
 
 
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_input_1Placeholder*6
_output_shapes$
": ????????????*
dtype0*+
shape": ????????????
?
serving_default_input_2Placeholder*6
_output_shapes$
": ????????????*
dtype0*+
shape": ????????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2conv3d/kernelconv3d/biasconv3d_1_13/kernelconv3d_1_13/biasconv3d_2_13/kernelconv3d_2_13/biasconv3d_3_13/kernelconv3d_3_13/biasconv3d_4_13/kernelconv3d_4_13/biasconv3d_5_13/kernelconv3d_5_13/biasconv3d_6_13/kernelconv3d_6_13/biasconv3d_7_13/kernelconv3d_7_13/biasconv3d_8_13/kernelconv3d_8_13/biasconv3d_9_13/kernelconv3d_9_13/biasconv3d_10_13/kernelconv3d_10_13/biasdisp_10/kerneldisp_10/bias*%
Tin
2*
Tout
2*X
_output_shapesF
D: ????????????: ????????????*:
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*-
f(R&
$__inference_signature_wrapper_274340
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv3d/kernel/Read/ReadVariableOpconv3d/bias/Read/ReadVariableOp&conv3d_1_13/kernel/Read/ReadVariableOp$conv3d_1_13/bias/Read/ReadVariableOp&conv3d_2_13/kernel/Read/ReadVariableOp$conv3d_2_13/bias/Read/ReadVariableOp&conv3d_3_13/kernel/Read/ReadVariableOp$conv3d_3_13/bias/Read/ReadVariableOp&conv3d_4_13/kernel/Read/ReadVariableOp$conv3d_4_13/bias/Read/ReadVariableOp&conv3d_5_13/kernel/Read/ReadVariableOp$conv3d_5_13/bias/Read/ReadVariableOp&conv3d_6_13/kernel/Read/ReadVariableOp$conv3d_6_13/bias/Read/ReadVariableOp&conv3d_7_13/kernel/Read/ReadVariableOp$conv3d_7_13/bias/Read/ReadVariableOp&conv3d_8_13/kernel/Read/ReadVariableOp$conv3d_8_13/bias/Read/ReadVariableOp&conv3d_9_13/kernel/Read/ReadVariableOp$conv3d_9_13/bias/Read/ReadVariableOp'conv3d_10_13/kernel/Read/ReadVariableOp%conv3d_10_13/bias/Read/ReadVariableOp"disp_10/kernel/Read/ReadVariableOp disp_10/bias/Read/ReadVariableOpConst*%
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*(
f#R!
__inference__traced_save_277703
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d/kernelconv3d/biasconv3d_1_13/kernelconv3d_1_13/biasconv3d_2_13/kernelconv3d_2_13/biasconv3d_3_13/kernelconv3d_3_13/biasconv3d_4_13/kernelconv3d_4_13/biasconv3d_5_13/kernelconv3d_5_13/biasconv3d_6_13/kernelconv3d_6_13/biasconv3d_7_13/kernelconv3d_7_13/biasconv3d_8_13/kernelconv3d_8_13/biasconv3d_9_13/kernelconv3d_9_13/biasconv3d_10_13/kernelconv3d_10_13/biasdisp_10/kerneldisp_10/bias*$
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference__traced_restore_277787??+
?
f
J__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_277264

inputs
identityc
	LeakyRelu	LeakyReluinputs*6
_output_shapes$
": ???????????? 2
	LeakyReluz
IdentityIdentityLeakyRelu:activations:0*
T0*6
_output_shapes$
": ???????????? 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ???????????? :^ Z
6
_output_shapes$
": ???????????? 
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_276608

inputs
identity`
	LeakyRelu	LeakyReluinputs*3
_output_shapes!
:?????????@2
	LeakyReluw
IdentityIdentityLeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
??
?
C__inference_model_1_layer_call_and_return_conditional_losses_273908
input_1
input_2
conv3d_272667
conv3d_272669
conv3d_1_272685
conv3d_1_272687
conv3d_2_272703
conv3d_2_272705
conv3d_3_272721
conv3d_3_272723
conv3d_4_272739
conv3d_4_272741
conv3d_5_272833
conv3d_5_272835
conv3d_6_272963
conv3d_6_272965
conv3d_7_273165
conv3d_7_273167
conv3d_8_273183
conv3d_8_273185
conv3d_9_273529
conv3d_9_273531
conv3d_10_273547
conv3d_10_273549
disp_273565
disp_273567
identity

identity_1??conv3d/StatefulPartitionedCall? conv3d_1/StatefulPartitionedCall?!conv3d_10/StatefulPartitionedCall? conv3d_2/StatefulPartitionedCall? conv3d_3/StatefulPartitionedCall? conv3d_4/StatefulPartitionedCall? conv3d_5/StatefulPartitionedCall? conv3d_6/StatefulPartitionedCall? conv3d_7/StatefulPartitionedCall? conv3d_8/StatefulPartitionedCall? conv3d_9/StatefulPartitionedCall?disp/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCallinput_1input_2*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2726582
concatenate/PartitionedCall?
conv3d/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv3d_272667conv3d_272669*
Tin
2*
Tout
2*3
_output_shapes!
:?????????``` *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_2724052 
conv3d/StatefulPartitionedCall?
leaky_re_lu/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????``` * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_2726772
leaky_re_lu/PartitionedCall?
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0conv3d_1_272685conv3d_1_272687*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_2724262"
 conv3d_1/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_2726952
leaky_re_lu_1/PartitionedCall?
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0conv3d_2_272703conv3d_2_272705*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_2724472"
 conv3d_2/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_2727132
leaky_re_lu_2/PartitionedCall?
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0conv3d_3_272721conv3d_3_272723*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_2724682"
 conv3d_3/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_2727312
leaky_re_lu_3/PartitionedCall?
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0conv3d_4_272739conv3d_4_272741*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_2724892"
 conv3d_4/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_2727492
leaky_re_lu_4/PartitionedCall?
up_sampling3d/PartitionedCallPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_up_sampling3d_layer_call_and_return_conditional_losses_2728092
up_sampling3d/PartitionedCall?
concatenate_1/PartitionedCallPartitionedCall&up_sampling3d/PartitionedCall:output:0&leaky_re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*4
_output_shapes"
 :??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_2728242
concatenate_1/PartitionedCall?
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv3d_5_272833conv3d_5_272835*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_2725102"
 conv3d_5/StatefulPartitionedCall?
leaky_re_lu_5/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_2728432
leaky_re_lu_5/PartitionedCall?
up_sampling3d_1/PartitionedCallPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_2729392!
up_sampling3d_1/PartitionedCall?
concatenate_2/PartitionedCallPartitionedCall(up_sampling3d_1/PartitionedCall:output:0&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*4
_output_shapes"
 :?????????000?* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_2729542
concatenate_2/PartitionedCall?
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv3d_6_272963conv3d_6_272965*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_6_layer_call_and_return_conditional_losses_2725312"
 conv3d_6/StatefulPartitionedCall?
leaky_re_lu_6/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_2729732
leaky_re_lu_6/PartitionedCall?
up_sampling3d_2/PartitionedCallPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_2731412!
up_sampling3d_2/PartitionedCall?
concatenate_3/PartitionedCallPartitionedCall(up_sampling3d_2/PartitionedCall:output:0$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????````* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_2731562
concatenate_3/PartitionedCall?
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv3d_7_273165conv3d_7_273167*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_7_layer_call_and_return_conditional_losses_2725522"
 conv3d_7/StatefulPartitionedCall?
leaky_re_lu_7/PartitionedCallPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_2731752
leaky_re_lu_7/PartitionedCall?
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0conv3d_8_273183conv3d_8_273185*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_8_layer_call_and_return_conditional_losses_2725732"
 conv3d_8/StatefulPartitionedCall?
leaky_re_lu_8/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_2731932
leaky_re_lu_8/PartitionedCall?
up_sampling3d_3/PartitionedCallPartitionedCall&leaky_re_lu_8/PartitionedCall:output:0*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_2735052!
up_sampling3d_3/PartitionedCall?
concatenate_4/PartitionedCallPartitionedCall(up_sampling3d_3/PartitionedCall:output:0$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????B* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_2735202
concatenate_4/PartitionedCall?
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv3d_9_273529conv3d_9_273531*
Tin
2*
Tout
2*6
_output_shapes$
": ???????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_9_layer_call_and_return_conditional_losses_2725942"
 conv3d_9/StatefulPartitionedCall?
leaky_re_lu_9/PartitionedCallPartitionedCall)conv3d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*6
_output_shapes$
": ???????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_2735392
leaky_re_lu_9/PartitionedCall?
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_9/PartitionedCall:output:0conv3d_10_273547conv3d_10_273549*
Tin
2*
Tout
2*6
_output_shapes$
": ???????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_conv3d_10_layer_call_and_return_conditional_losses_2726152#
!conv3d_10/StatefulPartitionedCall?
leaky_re_lu_10/PartitionedCallPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*6
_output_shapes$
": ???????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_2735572 
leaky_re_lu_10/PartitionedCall?
disp/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_10/PartitionedCall:output:0disp_273565disp_273567*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_disp_layer_call_and_return_conditional_losses_2726362
disp/StatefulPartitionedCall?
transformer/PartitionedCallPartitionedCallinput_1%disp/StatefulPartitionedCall:output:0*
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
GPU 

CPU2J 8*P
fKRI
G__inference_transformer_layer_call_and_return_conditional_losses_2738972
transformer/PartitionedCall?
IdentityIdentity$transformer/PartitionedCall:output:0^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall^disp/StatefulPartitionedCall*
T0*6
_output_shapes$
": ????????????2

Identity?

Identity_1Identity%disp/StatefulPartitionedCall:output:0^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall^disp/StatefulPartitionedCall*
T0*6
_output_shapes$
": ????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ????????????: ????????????::::::::::::::::::::::::2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall2<
disp/StatefulPartitionedCalldisp/StatefulPartitionedCall:_ [
6
_output_shapes$
": ????????????
!
_user_specified_name	input_1:_[
6
_output_shapes$
": ????????????
!
_user_specified_name	input_2:
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
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
e
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_272695

inputs
identity`
	LeakyRelu	LeakyReluinputs*3
_output_shapes!
:?????????000@2
	LeakyReluw
IdentityIdentityLeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????000@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????000@:[ W
3
_output_shapes!
:?????????000@
 
_user_specified_nameinputs
?[
g
K__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_276889

inputs
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :02
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0inputs*
T0*?
_output_shapes?
?:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@*
	num_split02
split\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15split:output:16split:output:16split:output:17split:output:17split:output:18split:output:18split:output:19split:output:19split:output:20split:output:20split:output:21split:output:21split:output:22split:output:22split:output:23split:output:23split:output:24split:output:24split:output:25split:output:25split:output:26split:output:26split:output:27split:output:27split:output:28split:output:28split:output:29split:output:29split:output:30split:output:30split:output:31split:output:31split:output:32split:output:32split:output:33split:output:33split:output:34split:output:34split:output:35split:output:35split:output:36split:output:36split:output:37split:output:37split:output:38split:output:38split:output:39split:output:39split:output:40split:output:40split:output:41split:output:41split:output:42split:output:42split:output:43split:output:43split:output:44split:output:44split:output:45split:output:45split:output:46split:output:46split:output:47split:output:47concat/axis:output:0*
N`*
T0*3
_output_shapes!
:?????????`00@2
concatT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :02	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*?
_output_shapes?
?:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@*
	num_split02	
split_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15split_1:output:16split_1:output:16split_1:output:17split_1:output:17split_1:output:18split_1:output:18split_1:output:19split_1:output:19split_1:output:20split_1:output:20split_1:output:21split_1:output:21split_1:output:22split_1:output:22split_1:output:23split_1:output:23split_1:output:24split_1:output:24split_1:output:25split_1:output:25split_1:output:26split_1:output:26split_1:output:27split_1:output:27split_1:output:28split_1:output:28split_1:output:29split_1:output:29split_1:output:30split_1:output:30split_1:output:31split_1:output:31split_1:output:32split_1:output:32split_1:output:33split_1:output:33split_1:output:34split_1:output:34split_1:output:35split_1:output:35split_1:output:36split_1:output:36split_1:output:37split_1:output:37split_1:output:38split_1:output:38split_1:output:39split_1:output:39split_1:output:40split_1:output:40split_1:output:41split_1:output:41split_1:output:42split_1:output:42split_1:output:43split_1:output:43split_1:output:44split_1:output:44split_1:output:45split_1:output:45split_1:output:46split_1:output:46split_1:output:47split_1:output:47concat_1/axis:output:0*
N`*
T0*3
_output_shapes!
:?????????``0@2

concat_1T
Const_2Const*
_output_shapes
: *
dtype0*
value	B :02	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*?
_output_shapes?
?:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@*
	num_split02	
split_2`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis?
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15split_2:output:16split_2:output:16split_2:output:17split_2:output:17split_2:output:18split_2:output:18split_2:output:19split_2:output:19split_2:output:20split_2:output:20split_2:output:21split_2:output:21split_2:output:22split_2:output:22split_2:output:23split_2:output:23split_2:output:24split_2:output:24split_2:output:25split_2:output:25split_2:output:26split_2:output:26split_2:output:27split_2:output:27split_2:output:28split_2:output:28split_2:output:29split_2:output:29split_2:output:30split_2:output:30split_2:output:31split_2:output:31split_2:output:32split_2:output:32split_2:output:33split_2:output:33split_2:output:34split_2:output:34split_2:output:35split_2:output:35split_2:output:36split_2:output:36split_2:output:37split_2:output:37split_2:output:38split_2:output:38split_2:output:39split_2:output:39split_2:output:40split_2:output:40split_2:output:41split_2:output:41split_2:output:42split_2:output:42split_2:output:43split_2:output:43split_2:output:44split_2:output:44split_2:output:45split_2:output:45split_2:output:46split_2:output:46split_2:output:47split_2:output:47concat_2/axis:output:0*
N`*
T0*3
_output_shapes!
:?????????```@2

concat_2q
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:?????????```@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????000@:[ W
3
_output_shapes!
:?????????000@
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_276518

inputs
identity`
	LeakyRelu	LeakyReluinputs*3
_output_shapes!
:?????????@2
	LeakyReluw
IdentityIdentityLeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_272749

inputs
identity`
	LeakyRelu	LeakyReluinputs*3
_output_shapes!
:?????????@2
	LeakyReluw
IdentityIdentityLeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
u
I__inference_concatenate_1_layer_call_and_return_conditional_losses_276597
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????2
concatp
IdentityIdentityconcat:output:0*
T0*4
_output_shapes"
 :??????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????@:?????????@:] Y
3
_output_shapes!
:?????????@
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:?????????@
"
_user_specified_name
inputs/1
?

?
D__inference_conv3d_7_layer_call_and_return_conditional_losses_272552

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:`@*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????@*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8????????????????????????????????????`:::v r
N
_output_shapes<
::8????????????????????????????????????`
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
J
.__inference_leaky_re_lu_3_layer_call_fn_276523

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_2727312
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
??
?
C__inference_model_1_layer_call_and_return_conditional_losses_274086

inputs
inputs_1
conv3d_274004
conv3d_274006
conv3d_1_274010
conv3d_1_274012
conv3d_2_274016
conv3d_2_274018
conv3d_3_274022
conv3d_3_274024
conv3d_4_274028
conv3d_4_274030
conv3d_5_274036
conv3d_5_274038
conv3d_6_274044
conv3d_6_274046
conv3d_7_274052
conv3d_7_274054
conv3d_8_274058
conv3d_8_274060
conv3d_9_274066
conv3d_9_274068
conv3d_10_274072
conv3d_10_274074
disp_274078
disp_274080
identity

identity_1??conv3d/StatefulPartitionedCall? conv3d_1/StatefulPartitionedCall?!conv3d_10/StatefulPartitionedCall? conv3d_2/StatefulPartitionedCall? conv3d_3/StatefulPartitionedCall? conv3d_4/StatefulPartitionedCall? conv3d_5/StatefulPartitionedCall? conv3d_6/StatefulPartitionedCall? conv3d_7/StatefulPartitionedCall? conv3d_8/StatefulPartitionedCall? conv3d_9/StatefulPartitionedCall?disp/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2726582
concatenate/PartitionedCall?
conv3d/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv3d_274004conv3d_274006*
Tin
2*
Tout
2*3
_output_shapes!
:?????????``` *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_2724052 
conv3d/StatefulPartitionedCall?
leaky_re_lu/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????``` * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_2726772
leaky_re_lu/PartitionedCall?
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0conv3d_1_274010conv3d_1_274012*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_2724262"
 conv3d_1/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_2726952
leaky_re_lu_1/PartitionedCall?
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0conv3d_2_274016conv3d_2_274018*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_2724472"
 conv3d_2/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_2727132
leaky_re_lu_2/PartitionedCall?
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0conv3d_3_274022conv3d_3_274024*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_2724682"
 conv3d_3/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_2727312
leaky_re_lu_3/PartitionedCall?
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0conv3d_4_274028conv3d_4_274030*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_2724892"
 conv3d_4/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_2727492
leaky_re_lu_4/PartitionedCall?
up_sampling3d/PartitionedCallPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_up_sampling3d_layer_call_and_return_conditional_losses_2728092
up_sampling3d/PartitionedCall?
concatenate_1/PartitionedCallPartitionedCall&up_sampling3d/PartitionedCall:output:0&leaky_re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*4
_output_shapes"
 :??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_2728242
concatenate_1/PartitionedCall?
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv3d_5_274036conv3d_5_274038*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_2725102"
 conv3d_5/StatefulPartitionedCall?
leaky_re_lu_5/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_2728432
leaky_re_lu_5/PartitionedCall?
up_sampling3d_1/PartitionedCallPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_2729392!
up_sampling3d_1/PartitionedCall?
concatenate_2/PartitionedCallPartitionedCall(up_sampling3d_1/PartitionedCall:output:0&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*4
_output_shapes"
 :?????????000?* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_2729542
concatenate_2/PartitionedCall?
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv3d_6_274044conv3d_6_274046*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_6_layer_call_and_return_conditional_losses_2725312"
 conv3d_6/StatefulPartitionedCall?
leaky_re_lu_6/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_2729732
leaky_re_lu_6/PartitionedCall?
up_sampling3d_2/PartitionedCallPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_2731412!
up_sampling3d_2/PartitionedCall?
concatenate_3/PartitionedCallPartitionedCall(up_sampling3d_2/PartitionedCall:output:0$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????````* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_2731562
concatenate_3/PartitionedCall?
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv3d_7_274052conv3d_7_274054*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_7_layer_call_and_return_conditional_losses_2725522"
 conv3d_7/StatefulPartitionedCall?
leaky_re_lu_7/PartitionedCallPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_2731752
leaky_re_lu_7/PartitionedCall?
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0conv3d_8_274058conv3d_8_274060*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_8_layer_call_and_return_conditional_losses_2725732"
 conv3d_8/StatefulPartitionedCall?
leaky_re_lu_8/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_2731932
leaky_re_lu_8/PartitionedCall?
up_sampling3d_3/PartitionedCallPartitionedCall&leaky_re_lu_8/PartitionedCall:output:0*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_2735052!
up_sampling3d_3/PartitionedCall?
concatenate_4/PartitionedCallPartitionedCall(up_sampling3d_3/PartitionedCall:output:0$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????B* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_2735202
concatenate_4/PartitionedCall?
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv3d_9_274066conv3d_9_274068*
Tin
2*
Tout
2*6
_output_shapes$
": ???????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_9_layer_call_and_return_conditional_losses_2725942"
 conv3d_9/StatefulPartitionedCall?
leaky_re_lu_9/PartitionedCallPartitionedCall)conv3d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*6
_output_shapes$
": ???????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_2735392
leaky_re_lu_9/PartitionedCall?
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_9/PartitionedCall:output:0conv3d_10_274072conv3d_10_274074*
Tin
2*
Tout
2*6
_output_shapes$
": ???????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_conv3d_10_layer_call_and_return_conditional_losses_2726152#
!conv3d_10/StatefulPartitionedCall?
leaky_re_lu_10/PartitionedCallPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*6
_output_shapes$
": ???????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_2735572 
leaky_re_lu_10/PartitionedCall?
disp/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_10/PartitionedCall:output:0disp_274078disp_274080*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_disp_layer_call_and_return_conditional_losses_2726362
disp/StatefulPartitionedCall?
transformer/PartitionedCallPartitionedCallinputs%disp/StatefulPartitionedCall:output:0*
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
GPU 

CPU2J 8*P
fKRI
G__inference_transformer_layer_call_and_return_conditional_losses_2738972
transformer/PartitionedCall?
IdentityIdentity$transformer/PartitionedCall:output:0^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall^disp/StatefulPartitionedCall*
T0*6
_output_shapes$
": ????????????2

Identity?

Identity_1Identity%disp/StatefulPartitionedCall:output:0^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall^disp/StatefulPartitionedCall*
T0*6
_output_shapes$
": ????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ????????????: ????????????::::::::::::::::::::::::2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall2<
disp/StatefulPartitionedCalldisp/StatefulPartitionedCall:^ Z
6
_output_shapes$
": ????????????
 
_user_specified_nameinputs:^Z
6
_output_shapes$
": ????????????
 
_user_specified_nameinputs:
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
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
e
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_272731

inputs
identity`
	LeakyRelu	LeakyReluinputs*3
_output_shapes!
:?????????@2
	LeakyReluw
IdentityIdentityLeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
??
?	
C__inference_model_1_layer_call_and_return_conditional_losses_276358
inputs_0
inputs_1)
%conv3d_conv3d_readvariableop_resource*
&conv3d_biasadd_readvariableop_resource+
'conv3d_1_conv3d_readvariableop_resource,
(conv3d_1_biasadd_readvariableop_resource+
'conv3d_2_conv3d_readvariableop_resource,
(conv3d_2_biasadd_readvariableop_resource+
'conv3d_3_conv3d_readvariableop_resource,
(conv3d_3_biasadd_readvariableop_resource+
'conv3d_4_conv3d_readvariableop_resource,
(conv3d_4_biasadd_readvariableop_resource+
'conv3d_5_conv3d_readvariableop_resource,
(conv3d_5_biasadd_readvariableop_resource+
'conv3d_6_conv3d_readvariableop_resource,
(conv3d_6_biasadd_readvariableop_resource+
'conv3d_7_conv3d_readvariableop_resource,
(conv3d_7_biasadd_readvariableop_resource+
'conv3d_8_conv3d_readvariableop_resource,
(conv3d_8_biasadd_readvariableop_resource+
'conv3d_9_conv3d_readvariableop_resource,
(conv3d_9_biasadd_readvariableop_resource,
(conv3d_10_conv3d_readvariableop_resource-
)conv3d_10_biasadd_readvariableop_resource'
#disp_conv3d_readvariableop_resource(
$disp_biasadd_readvariableop_resource
identity

identity_1?t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2inputs_0inputs_1 concatenate/concat/axis:output:0*
N*
T0*6
_output_shapes$
": ????????????2
concatenate/concat?
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02
conv3d/Conv3D/ReadVariableOp?
conv3d/Conv3DConv3Dconcatenate/concat:output:0$conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????``` *
paddingSAME*
strides	
2
conv3d/Conv3D?
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv3d/BiasAdd/ReadVariableOp?
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????``` 2
conv3d/BiasAdd?
leaky_re_lu/LeakyRelu	LeakyReluconv3d/BiasAdd:output:0*3
_output_shapes!
:?????????``` 2
leaky_re_lu/LeakyRelu?
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype02 
conv3d_1/Conv3D/ReadVariableOp?
conv3d_1/Conv3DConv3D#leaky_re_lu/LeakyRelu:activations:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????000@*
paddingSAME*
strides	
2
conv3d_1/Conv3D?
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_1/BiasAdd/ReadVariableOp?
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????000@2
conv3d_1/BiasAdd?
leaky_re_lu_1/LeakyRelu	LeakyReluconv3d_1/BiasAdd:output:0*3
_output_shapes!
:?????????000@2
leaky_re_lu_1/LeakyRelu?
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype02 
conv3d_2/Conv3D/ReadVariableOp?
conv3d_2/Conv3DConv3D%leaky_re_lu_1/LeakyRelu:activations:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
2
conv3d_2/Conv3D?
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_2/BiasAdd/ReadVariableOp?
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@2
conv3d_2/BiasAdd?
leaky_re_lu_2/LeakyRelu	LeakyReluconv3d_2/BiasAdd:output:0*3
_output_shapes!
:?????????@2
leaky_re_lu_2/LeakyRelu?
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype02 
conv3d_3/Conv3D/ReadVariableOp?
conv3d_3/Conv3DConv3D%leaky_re_lu_2/LeakyRelu:activations:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
2
conv3d_3/Conv3D?
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_3/BiasAdd/ReadVariableOp?
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@2
conv3d_3/BiasAdd?
leaky_re_lu_3/LeakyRelu	LeakyReluconv3d_3/BiasAdd:output:0*3
_output_shapes!
:?????????@2
leaky_re_lu_3/LeakyRelu?
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype02 
conv3d_4/Conv3D/ReadVariableOp?
conv3d_4/Conv3DConv3D%leaky_re_lu_3/LeakyRelu:activations:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
2
conv3d_4/Conv3D?
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_4/BiasAdd/ReadVariableOp?
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@2
conv3d_4/BiasAdd?
leaky_re_lu_4/LeakyRelu	LeakyReluconv3d_4/BiasAdd:output:0*3
_output_shapes!
:?????????@2
leaky_re_lu_4/LeakyRelul
up_sampling3d/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/Const?
up_sampling3d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/split/split_dim?
up_sampling3d/splitSplit&up_sampling3d/split/split_dim:output:0%leaky_re_lu_4/LeakyRelu:activations:0*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2
up_sampling3d/splitx
up_sampling3d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/concat/axis?
up_sampling3d/concatConcatV2up_sampling3d/split:output:0up_sampling3d/split:output:0up_sampling3d/split:output:1up_sampling3d/split:output:1up_sampling3d/split:output:2up_sampling3d/split:output:2up_sampling3d/split:output:3up_sampling3d/split:output:3up_sampling3d/split:output:4up_sampling3d/split:output:4up_sampling3d/split:output:5up_sampling3d/split:output:5up_sampling3d/split:output:6up_sampling3d/split:output:6up_sampling3d/split:output:7up_sampling3d/split:output:7up_sampling3d/split:output:8up_sampling3d/split:output:8up_sampling3d/split:output:9up_sampling3d/split:output:9up_sampling3d/split:output:10up_sampling3d/split:output:10up_sampling3d/split:output:11up_sampling3d/split:output:11"up_sampling3d/concat/axis:output:0*
N*
T0*3
_output_shapes!
:?????????@2
up_sampling3d/concatp
up_sampling3d/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/Const_1?
up_sampling3d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
up_sampling3d/split_1/split_dim?
up_sampling3d/split_1Split(up_sampling3d/split_1/split_dim:output:0up_sampling3d/concat:output:0*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2
up_sampling3d/split_1|
up_sampling3d/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/concat_1/axis?
up_sampling3d/concat_1ConcatV2up_sampling3d/split_1:output:0up_sampling3d/split_1:output:0up_sampling3d/split_1:output:1up_sampling3d/split_1:output:1up_sampling3d/split_1:output:2up_sampling3d/split_1:output:2up_sampling3d/split_1:output:3up_sampling3d/split_1:output:3up_sampling3d/split_1:output:4up_sampling3d/split_1:output:4up_sampling3d/split_1:output:5up_sampling3d/split_1:output:5up_sampling3d/split_1:output:6up_sampling3d/split_1:output:6up_sampling3d/split_1:output:7up_sampling3d/split_1:output:7up_sampling3d/split_1:output:8up_sampling3d/split_1:output:8up_sampling3d/split_1:output:9up_sampling3d/split_1:output:9up_sampling3d/split_1:output:10up_sampling3d/split_1:output:10up_sampling3d/split_1:output:11up_sampling3d/split_1:output:11$up_sampling3d/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:?????????@2
up_sampling3d/concat_1p
up_sampling3d/Const_2Const*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/Const_2?
up_sampling3d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
up_sampling3d/split_2/split_dim?
up_sampling3d/split_2Split(up_sampling3d/split_2/split_dim:output:0up_sampling3d/concat_1:output:0*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2
up_sampling3d/split_2|
up_sampling3d/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/concat_2/axis?
up_sampling3d/concat_2ConcatV2up_sampling3d/split_2:output:0up_sampling3d/split_2:output:0up_sampling3d/split_2:output:1up_sampling3d/split_2:output:1up_sampling3d/split_2:output:2up_sampling3d/split_2:output:2up_sampling3d/split_2:output:3up_sampling3d/split_2:output:3up_sampling3d/split_2:output:4up_sampling3d/split_2:output:4up_sampling3d/split_2:output:5up_sampling3d/split_2:output:5up_sampling3d/split_2:output:6up_sampling3d/split_2:output:6up_sampling3d/split_2:output:7up_sampling3d/split_2:output:7up_sampling3d/split_2:output:8up_sampling3d/split_2:output:8up_sampling3d/split_2:output:9up_sampling3d/split_2:output:9up_sampling3d/split_2:output:10up_sampling3d/split_2:output:10up_sampling3d/split_2:output:11up_sampling3d/split_2:output:11$up_sampling3d/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:?????????@2
up_sampling3d/concat_2x
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2up_sampling3d/concat_2:output:0%leaky_re_lu_2/LeakyRelu:activations:0"concatenate_1/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????2
concatenate_1/concat?
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource*+
_output_shapes
:?@*
dtype02 
conv3d_5/Conv3D/ReadVariableOp?
conv3d_5/Conv3DConv3Dconcatenate_1/concat:output:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
2
conv3d_5/Conv3D?
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_5/BiasAdd/ReadVariableOp?
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@2
conv3d_5/BiasAdd?
leaky_re_lu_5/LeakyRelu	LeakyReluconv3d_5/BiasAdd:output:0*3
_output_shapes!
:?????????@2
leaky_re_lu_5/LeakyRelup
up_sampling3d_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_1/Const?
up_sampling3d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
up_sampling3d_1/split/split_dim?
up_sampling3d_1/splitSplit(up_sampling3d_1/split/split_dim:output:0%leaky_re_lu_5/LeakyRelu:activations:0*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2
up_sampling3d_1/split|
up_sampling3d_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_1/concat/axis?
up_sampling3d_1/concatConcatV2up_sampling3d_1/split:output:0up_sampling3d_1/split:output:0up_sampling3d_1/split:output:1up_sampling3d_1/split:output:1up_sampling3d_1/split:output:2up_sampling3d_1/split:output:2up_sampling3d_1/split:output:3up_sampling3d_1/split:output:3up_sampling3d_1/split:output:4up_sampling3d_1/split:output:4up_sampling3d_1/split:output:5up_sampling3d_1/split:output:5up_sampling3d_1/split:output:6up_sampling3d_1/split:output:6up_sampling3d_1/split:output:7up_sampling3d_1/split:output:7up_sampling3d_1/split:output:8up_sampling3d_1/split:output:8up_sampling3d_1/split:output:9up_sampling3d_1/split:output:9up_sampling3d_1/split:output:10up_sampling3d_1/split:output:10up_sampling3d_1/split:output:11up_sampling3d_1/split:output:11up_sampling3d_1/split:output:12up_sampling3d_1/split:output:12up_sampling3d_1/split:output:13up_sampling3d_1/split:output:13up_sampling3d_1/split:output:14up_sampling3d_1/split:output:14up_sampling3d_1/split:output:15up_sampling3d_1/split:output:15up_sampling3d_1/split:output:16up_sampling3d_1/split:output:16up_sampling3d_1/split:output:17up_sampling3d_1/split:output:17up_sampling3d_1/split:output:18up_sampling3d_1/split:output:18up_sampling3d_1/split:output:19up_sampling3d_1/split:output:19up_sampling3d_1/split:output:20up_sampling3d_1/split:output:20up_sampling3d_1/split:output:21up_sampling3d_1/split:output:21up_sampling3d_1/split:output:22up_sampling3d_1/split:output:22up_sampling3d_1/split:output:23up_sampling3d_1/split:output:23$up_sampling3d_1/concat/axis:output:0*
N0*
T0*3
_output_shapes!
:?????????0@2
up_sampling3d_1/concatt
up_sampling3d_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_1/Const_1?
!up_sampling3d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!up_sampling3d_1/split_1/split_dim?
up_sampling3d_1/split_1Split*up_sampling3d_1/split_1/split_dim:output:0up_sampling3d_1/concat:output:0*
T0*?
_output_shapes?
?:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@*
	num_split2
up_sampling3d_1/split_1?
up_sampling3d_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_1/concat_1/axis?
up_sampling3d_1/concat_1ConcatV2 up_sampling3d_1/split_1:output:0 up_sampling3d_1/split_1:output:0 up_sampling3d_1/split_1:output:1 up_sampling3d_1/split_1:output:1 up_sampling3d_1/split_1:output:2 up_sampling3d_1/split_1:output:2 up_sampling3d_1/split_1:output:3 up_sampling3d_1/split_1:output:3 up_sampling3d_1/split_1:output:4 up_sampling3d_1/split_1:output:4 up_sampling3d_1/split_1:output:5 up_sampling3d_1/split_1:output:5 up_sampling3d_1/split_1:output:6 up_sampling3d_1/split_1:output:6 up_sampling3d_1/split_1:output:7 up_sampling3d_1/split_1:output:7 up_sampling3d_1/split_1:output:8 up_sampling3d_1/split_1:output:8 up_sampling3d_1/split_1:output:9 up_sampling3d_1/split_1:output:9!up_sampling3d_1/split_1:output:10!up_sampling3d_1/split_1:output:10!up_sampling3d_1/split_1:output:11!up_sampling3d_1/split_1:output:11!up_sampling3d_1/split_1:output:12!up_sampling3d_1/split_1:output:12!up_sampling3d_1/split_1:output:13!up_sampling3d_1/split_1:output:13!up_sampling3d_1/split_1:output:14!up_sampling3d_1/split_1:output:14!up_sampling3d_1/split_1:output:15!up_sampling3d_1/split_1:output:15!up_sampling3d_1/split_1:output:16!up_sampling3d_1/split_1:output:16!up_sampling3d_1/split_1:output:17!up_sampling3d_1/split_1:output:17!up_sampling3d_1/split_1:output:18!up_sampling3d_1/split_1:output:18!up_sampling3d_1/split_1:output:19!up_sampling3d_1/split_1:output:19!up_sampling3d_1/split_1:output:20!up_sampling3d_1/split_1:output:20!up_sampling3d_1/split_1:output:21!up_sampling3d_1/split_1:output:21!up_sampling3d_1/split_1:output:22!up_sampling3d_1/split_1:output:22!up_sampling3d_1/split_1:output:23!up_sampling3d_1/split_1:output:23&up_sampling3d_1/concat_1/axis:output:0*
N0*
T0*3
_output_shapes!
:?????????00@2
up_sampling3d_1/concat_1t
up_sampling3d_1/Const_2Const*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_1/Const_2?
!up_sampling3d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!up_sampling3d_1/split_2/split_dim?
up_sampling3d_1/split_2Split*up_sampling3d_1/split_2/split_dim:output:0!up_sampling3d_1/concat_1:output:0*
T0*?
_output_shapes?
?:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@*
	num_split2
up_sampling3d_1/split_2?
up_sampling3d_1/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_1/concat_2/axis?
up_sampling3d_1/concat_2ConcatV2 up_sampling3d_1/split_2:output:0 up_sampling3d_1/split_2:output:0 up_sampling3d_1/split_2:output:1 up_sampling3d_1/split_2:output:1 up_sampling3d_1/split_2:output:2 up_sampling3d_1/split_2:output:2 up_sampling3d_1/split_2:output:3 up_sampling3d_1/split_2:output:3 up_sampling3d_1/split_2:output:4 up_sampling3d_1/split_2:output:4 up_sampling3d_1/split_2:output:5 up_sampling3d_1/split_2:output:5 up_sampling3d_1/split_2:output:6 up_sampling3d_1/split_2:output:6 up_sampling3d_1/split_2:output:7 up_sampling3d_1/split_2:output:7 up_sampling3d_1/split_2:output:8 up_sampling3d_1/split_2:output:8 up_sampling3d_1/split_2:output:9 up_sampling3d_1/split_2:output:9!up_sampling3d_1/split_2:output:10!up_sampling3d_1/split_2:output:10!up_sampling3d_1/split_2:output:11!up_sampling3d_1/split_2:output:11!up_sampling3d_1/split_2:output:12!up_sampling3d_1/split_2:output:12!up_sampling3d_1/split_2:output:13!up_sampling3d_1/split_2:output:13!up_sampling3d_1/split_2:output:14!up_sampling3d_1/split_2:output:14!up_sampling3d_1/split_2:output:15!up_sampling3d_1/split_2:output:15!up_sampling3d_1/split_2:output:16!up_sampling3d_1/split_2:output:16!up_sampling3d_1/split_2:output:17!up_sampling3d_1/split_2:output:17!up_sampling3d_1/split_2:output:18!up_sampling3d_1/split_2:output:18!up_sampling3d_1/split_2:output:19!up_sampling3d_1/split_2:output:19!up_sampling3d_1/split_2:output:20!up_sampling3d_1/split_2:output:20!up_sampling3d_1/split_2:output:21!up_sampling3d_1/split_2:output:21!up_sampling3d_1/split_2:output:22!up_sampling3d_1/split_2:output:22!up_sampling3d_1/split_2:output:23!up_sampling3d_1/split_2:output:23&up_sampling3d_1/concat_2/axis:output:0*
N0*
T0*3
_output_shapes!
:?????????000@2
up_sampling3d_1/concat_2x
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axis?
concatenate_2/concatConcatV2!up_sampling3d_1/concat_2:output:0%leaky_re_lu_1/LeakyRelu:activations:0"concatenate_2/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :?????????000?2
concatenate_2/concat?
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource*+
_output_shapes
:?@*
dtype02 
conv3d_6/Conv3D/ReadVariableOp?
conv3d_6/Conv3DConv3Dconcatenate_2/concat:output:0&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????000@*
paddingSAME*
strides	
2
conv3d_6/Conv3D?
conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_6/BiasAdd/ReadVariableOp?
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????000@2
conv3d_6/BiasAdd?
leaky_re_lu_6/LeakyRelu	LeakyReluconv3d_6/BiasAdd:output:0*3
_output_shapes!
:?????????000@2
leaky_re_lu_6/LeakyRelup
up_sampling3d_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :02
up_sampling3d_2/Const?
up_sampling3d_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
up_sampling3d_2/split/split_dim?
up_sampling3d_2/splitSplit(up_sampling3d_2/split/split_dim:output:0%leaky_re_lu_6/LeakyRelu:activations:0*
T0*?
_output_shapes?
?:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@*
	num_split02
up_sampling3d_2/split|
up_sampling3d_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_2/concat/axis?
up_sampling3d_2/concatConcatV2up_sampling3d_2/split:output:0up_sampling3d_2/split:output:0up_sampling3d_2/split:output:1up_sampling3d_2/split:output:1up_sampling3d_2/split:output:2up_sampling3d_2/split:output:2up_sampling3d_2/split:output:3up_sampling3d_2/split:output:3up_sampling3d_2/split:output:4up_sampling3d_2/split:output:4up_sampling3d_2/split:output:5up_sampling3d_2/split:output:5up_sampling3d_2/split:output:6up_sampling3d_2/split:output:6up_sampling3d_2/split:output:7up_sampling3d_2/split:output:7up_sampling3d_2/split:output:8up_sampling3d_2/split:output:8up_sampling3d_2/split:output:9up_sampling3d_2/split:output:9up_sampling3d_2/split:output:10up_sampling3d_2/split:output:10up_sampling3d_2/split:output:11up_sampling3d_2/split:output:11up_sampling3d_2/split:output:12up_sampling3d_2/split:output:12up_sampling3d_2/split:output:13up_sampling3d_2/split:output:13up_sampling3d_2/split:output:14up_sampling3d_2/split:output:14up_sampling3d_2/split:output:15up_sampling3d_2/split:output:15up_sampling3d_2/split:output:16up_sampling3d_2/split:output:16up_sampling3d_2/split:output:17up_sampling3d_2/split:output:17up_sampling3d_2/split:output:18up_sampling3d_2/split:output:18up_sampling3d_2/split:output:19up_sampling3d_2/split:output:19up_sampling3d_2/split:output:20up_sampling3d_2/split:output:20up_sampling3d_2/split:output:21up_sampling3d_2/split:output:21up_sampling3d_2/split:output:22up_sampling3d_2/split:output:22up_sampling3d_2/split:output:23up_sampling3d_2/split:output:23up_sampling3d_2/split:output:24up_sampling3d_2/split:output:24up_sampling3d_2/split:output:25up_sampling3d_2/split:output:25up_sampling3d_2/split:output:26up_sampling3d_2/split:output:26up_sampling3d_2/split:output:27up_sampling3d_2/split:output:27up_sampling3d_2/split:output:28up_sampling3d_2/split:output:28up_sampling3d_2/split:output:29up_sampling3d_2/split:output:29up_sampling3d_2/split:output:30up_sampling3d_2/split:output:30up_sampling3d_2/split:output:31up_sampling3d_2/split:output:31up_sampling3d_2/split:output:32up_sampling3d_2/split:output:32up_sampling3d_2/split:output:33up_sampling3d_2/split:output:33up_sampling3d_2/split:output:34up_sampling3d_2/split:output:34up_sampling3d_2/split:output:35up_sampling3d_2/split:output:35up_sampling3d_2/split:output:36up_sampling3d_2/split:output:36up_sampling3d_2/split:output:37up_sampling3d_2/split:output:37up_sampling3d_2/split:output:38up_sampling3d_2/split:output:38up_sampling3d_2/split:output:39up_sampling3d_2/split:output:39up_sampling3d_2/split:output:40up_sampling3d_2/split:output:40up_sampling3d_2/split:output:41up_sampling3d_2/split:output:41up_sampling3d_2/split:output:42up_sampling3d_2/split:output:42up_sampling3d_2/split:output:43up_sampling3d_2/split:output:43up_sampling3d_2/split:output:44up_sampling3d_2/split:output:44up_sampling3d_2/split:output:45up_sampling3d_2/split:output:45up_sampling3d_2/split:output:46up_sampling3d_2/split:output:46up_sampling3d_2/split:output:47up_sampling3d_2/split:output:47$up_sampling3d_2/concat/axis:output:0*
N`*
T0*3
_output_shapes!
:?????????`00@2
up_sampling3d_2/concatt
up_sampling3d_2/Const_1Const*
_output_shapes
: *
dtype0*
value	B :02
up_sampling3d_2/Const_1?
!up_sampling3d_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!up_sampling3d_2/split_1/split_dim?
up_sampling3d_2/split_1Split*up_sampling3d_2/split_1/split_dim:output:0up_sampling3d_2/concat:output:0*
T0*?
_output_shapes?
?:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@*
	num_split02
up_sampling3d_2/split_1?
up_sampling3d_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_2/concat_1/axis?
up_sampling3d_2/concat_1ConcatV2 up_sampling3d_2/split_1:output:0 up_sampling3d_2/split_1:output:0 up_sampling3d_2/split_1:output:1 up_sampling3d_2/split_1:output:1 up_sampling3d_2/split_1:output:2 up_sampling3d_2/split_1:output:2 up_sampling3d_2/split_1:output:3 up_sampling3d_2/split_1:output:3 up_sampling3d_2/split_1:output:4 up_sampling3d_2/split_1:output:4 up_sampling3d_2/split_1:output:5 up_sampling3d_2/split_1:output:5 up_sampling3d_2/split_1:output:6 up_sampling3d_2/split_1:output:6 up_sampling3d_2/split_1:output:7 up_sampling3d_2/split_1:output:7 up_sampling3d_2/split_1:output:8 up_sampling3d_2/split_1:output:8 up_sampling3d_2/split_1:output:9 up_sampling3d_2/split_1:output:9!up_sampling3d_2/split_1:output:10!up_sampling3d_2/split_1:output:10!up_sampling3d_2/split_1:output:11!up_sampling3d_2/split_1:output:11!up_sampling3d_2/split_1:output:12!up_sampling3d_2/split_1:output:12!up_sampling3d_2/split_1:output:13!up_sampling3d_2/split_1:output:13!up_sampling3d_2/split_1:output:14!up_sampling3d_2/split_1:output:14!up_sampling3d_2/split_1:output:15!up_sampling3d_2/split_1:output:15!up_sampling3d_2/split_1:output:16!up_sampling3d_2/split_1:output:16!up_sampling3d_2/split_1:output:17!up_sampling3d_2/split_1:output:17!up_sampling3d_2/split_1:output:18!up_sampling3d_2/split_1:output:18!up_sampling3d_2/split_1:output:19!up_sampling3d_2/split_1:output:19!up_sampling3d_2/split_1:output:20!up_sampling3d_2/split_1:output:20!up_sampling3d_2/split_1:output:21!up_sampling3d_2/split_1:output:21!up_sampling3d_2/split_1:output:22!up_sampling3d_2/split_1:output:22!up_sampling3d_2/split_1:output:23!up_sampling3d_2/split_1:output:23!up_sampling3d_2/split_1:output:24!up_sampling3d_2/split_1:output:24!up_sampling3d_2/split_1:output:25!up_sampling3d_2/split_1:output:25!up_sampling3d_2/split_1:output:26!up_sampling3d_2/split_1:output:26!up_sampling3d_2/split_1:output:27!up_sampling3d_2/split_1:output:27!up_sampling3d_2/split_1:output:28!up_sampling3d_2/split_1:output:28!up_sampling3d_2/split_1:output:29!up_sampling3d_2/split_1:output:29!up_sampling3d_2/split_1:output:30!up_sampling3d_2/split_1:output:30!up_sampling3d_2/split_1:output:31!up_sampling3d_2/split_1:output:31!up_sampling3d_2/split_1:output:32!up_sampling3d_2/split_1:output:32!up_sampling3d_2/split_1:output:33!up_sampling3d_2/split_1:output:33!up_sampling3d_2/split_1:output:34!up_sampling3d_2/split_1:output:34!up_sampling3d_2/split_1:output:35!up_sampling3d_2/split_1:output:35!up_sampling3d_2/split_1:output:36!up_sampling3d_2/split_1:output:36!up_sampling3d_2/split_1:output:37!up_sampling3d_2/split_1:output:37!up_sampling3d_2/split_1:output:38!up_sampling3d_2/split_1:output:38!up_sampling3d_2/split_1:output:39!up_sampling3d_2/split_1:output:39!up_sampling3d_2/split_1:output:40!up_sampling3d_2/split_1:output:40!up_sampling3d_2/split_1:output:41!up_sampling3d_2/split_1:output:41!up_sampling3d_2/split_1:output:42!up_sampling3d_2/split_1:output:42!up_sampling3d_2/split_1:output:43!up_sampling3d_2/split_1:output:43!up_sampling3d_2/split_1:output:44!up_sampling3d_2/split_1:output:44!up_sampling3d_2/split_1:output:45!up_sampling3d_2/split_1:output:45!up_sampling3d_2/split_1:output:46!up_sampling3d_2/split_1:output:46!up_sampling3d_2/split_1:output:47!up_sampling3d_2/split_1:output:47&up_sampling3d_2/concat_1/axis:output:0*
N`*
T0*3
_output_shapes!
:?????????``0@2
up_sampling3d_2/concat_1t
up_sampling3d_2/Const_2Const*
_output_shapes
: *
dtype0*
value	B :02
up_sampling3d_2/Const_2?
!up_sampling3d_2/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!up_sampling3d_2/split_2/split_dim?
up_sampling3d_2/split_2Split*up_sampling3d_2/split_2/split_dim:output:0!up_sampling3d_2/concat_1:output:0*
T0*?
_output_shapes?
?:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@*
	num_split02
up_sampling3d_2/split_2?
up_sampling3d_2/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_2/concat_2/axis?
up_sampling3d_2/concat_2ConcatV2 up_sampling3d_2/split_2:output:0 up_sampling3d_2/split_2:output:0 up_sampling3d_2/split_2:output:1 up_sampling3d_2/split_2:output:1 up_sampling3d_2/split_2:output:2 up_sampling3d_2/split_2:output:2 up_sampling3d_2/split_2:output:3 up_sampling3d_2/split_2:output:3 up_sampling3d_2/split_2:output:4 up_sampling3d_2/split_2:output:4 up_sampling3d_2/split_2:output:5 up_sampling3d_2/split_2:output:5 up_sampling3d_2/split_2:output:6 up_sampling3d_2/split_2:output:6 up_sampling3d_2/split_2:output:7 up_sampling3d_2/split_2:output:7 up_sampling3d_2/split_2:output:8 up_sampling3d_2/split_2:output:8 up_sampling3d_2/split_2:output:9 up_sampling3d_2/split_2:output:9!up_sampling3d_2/split_2:output:10!up_sampling3d_2/split_2:output:10!up_sampling3d_2/split_2:output:11!up_sampling3d_2/split_2:output:11!up_sampling3d_2/split_2:output:12!up_sampling3d_2/split_2:output:12!up_sampling3d_2/split_2:output:13!up_sampling3d_2/split_2:output:13!up_sampling3d_2/split_2:output:14!up_sampling3d_2/split_2:output:14!up_sampling3d_2/split_2:output:15!up_sampling3d_2/split_2:output:15!up_sampling3d_2/split_2:output:16!up_sampling3d_2/split_2:output:16!up_sampling3d_2/split_2:output:17!up_sampling3d_2/split_2:output:17!up_sampling3d_2/split_2:output:18!up_sampling3d_2/split_2:output:18!up_sampling3d_2/split_2:output:19!up_sampling3d_2/split_2:output:19!up_sampling3d_2/split_2:output:20!up_sampling3d_2/split_2:output:20!up_sampling3d_2/split_2:output:21!up_sampling3d_2/split_2:output:21!up_sampling3d_2/split_2:output:22!up_sampling3d_2/split_2:output:22!up_sampling3d_2/split_2:output:23!up_sampling3d_2/split_2:output:23!up_sampling3d_2/split_2:output:24!up_sampling3d_2/split_2:output:24!up_sampling3d_2/split_2:output:25!up_sampling3d_2/split_2:output:25!up_sampling3d_2/split_2:output:26!up_sampling3d_2/split_2:output:26!up_sampling3d_2/split_2:output:27!up_sampling3d_2/split_2:output:27!up_sampling3d_2/split_2:output:28!up_sampling3d_2/split_2:output:28!up_sampling3d_2/split_2:output:29!up_sampling3d_2/split_2:output:29!up_sampling3d_2/split_2:output:30!up_sampling3d_2/split_2:output:30!up_sampling3d_2/split_2:output:31!up_sampling3d_2/split_2:output:31!up_sampling3d_2/split_2:output:32!up_sampling3d_2/split_2:output:32!up_sampling3d_2/split_2:output:33!up_sampling3d_2/split_2:output:33!up_sampling3d_2/split_2:output:34!up_sampling3d_2/split_2:output:34!up_sampling3d_2/split_2:output:35!up_sampling3d_2/split_2:output:35!up_sampling3d_2/split_2:output:36!up_sampling3d_2/split_2:output:36!up_sampling3d_2/split_2:output:37!up_sampling3d_2/split_2:output:37!up_sampling3d_2/split_2:output:38!up_sampling3d_2/split_2:output:38!up_sampling3d_2/split_2:output:39!up_sampling3d_2/split_2:output:39!up_sampling3d_2/split_2:output:40!up_sampling3d_2/split_2:output:40!up_sampling3d_2/split_2:output:41!up_sampling3d_2/split_2:output:41!up_sampling3d_2/split_2:output:42!up_sampling3d_2/split_2:output:42!up_sampling3d_2/split_2:output:43!up_sampling3d_2/split_2:output:43!up_sampling3d_2/split_2:output:44!up_sampling3d_2/split_2:output:44!up_sampling3d_2/split_2:output:45!up_sampling3d_2/split_2:output:45!up_sampling3d_2/split_2:output:46!up_sampling3d_2/split_2:output:46!up_sampling3d_2/split_2:output:47!up_sampling3d_2/split_2:output:47&up_sampling3d_2/concat_2/axis:output:0*
N`*
T0*3
_output_shapes!
:?????????```@2
up_sampling3d_2/concat_2x
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axis?
concatenate_3/concatConcatV2!up_sampling3d_2/concat_2:output:0#leaky_re_lu/LeakyRelu:activations:0"concatenate_3/concat/axis:output:0*
N*
T0*3
_output_shapes!
:?????????````2
concatenate_3/concat?
conv3d_7/Conv3D/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource**
_output_shapes
:`@*
dtype02 
conv3d_7/Conv3D/ReadVariableOp?
conv3d_7/Conv3DConv3Dconcatenate_3/concat:output:0&conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????```@*
paddingSAME*
strides	
2
conv3d_7/Conv3D?
conv3d_7/BiasAdd/ReadVariableOpReadVariableOp(conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_7/BiasAdd/ReadVariableOp?
conv3d_7/BiasAddBiasAddconv3d_7/Conv3D:output:0'conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????```@2
conv3d_7/BiasAdd?
leaky_re_lu_7/LeakyRelu	LeakyReluconv3d_7/BiasAdd:output:0*3
_output_shapes!
:?????????```@2
leaky_re_lu_7/LeakyRelu?
conv3d_8/Conv3D/ReadVariableOpReadVariableOp'conv3d_8_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype02 
conv3d_8/Conv3D/ReadVariableOp?
conv3d_8/Conv3DConv3D%leaky_re_lu_7/LeakyRelu:activations:0&conv3d_8/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????```@*
paddingSAME*
strides	
2
conv3d_8/Conv3D?
conv3d_8/BiasAdd/ReadVariableOpReadVariableOp(conv3d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_8/BiasAdd/ReadVariableOp?
conv3d_8/BiasAddBiasAddconv3d_8/Conv3D:output:0'conv3d_8/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????```@2
conv3d_8/BiasAdd?
leaky_re_lu_8/LeakyRelu	LeakyReluconv3d_8/BiasAdd:output:0*3
_output_shapes!
:?????????```@2
leaky_re_lu_8/LeakyRelup
up_sampling3d_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :`2
up_sampling3d_3/Const?
up_sampling3d_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
up_sampling3d_3/split/split_dim?
up_sampling3d_3/splitSplit(up_sampling3d_3/split/split_dim:output:0%leaky_re_lu_8/LeakyRelu:activations:0*
T0*?
_output_shapes?
?:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@*
	num_split`2
up_sampling3d_3/split|
up_sampling3d_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_3/concat/axis?2
up_sampling3d_3/concatConcatV2up_sampling3d_3/split:output:0up_sampling3d_3/split:output:0up_sampling3d_3/split:output:1up_sampling3d_3/split:output:1up_sampling3d_3/split:output:2up_sampling3d_3/split:output:2up_sampling3d_3/split:output:3up_sampling3d_3/split:output:3up_sampling3d_3/split:output:4up_sampling3d_3/split:output:4up_sampling3d_3/split:output:5up_sampling3d_3/split:output:5up_sampling3d_3/split:output:6up_sampling3d_3/split:output:6up_sampling3d_3/split:output:7up_sampling3d_3/split:output:7up_sampling3d_3/split:output:8up_sampling3d_3/split:output:8up_sampling3d_3/split:output:9up_sampling3d_3/split:output:9up_sampling3d_3/split:output:10up_sampling3d_3/split:output:10up_sampling3d_3/split:output:11up_sampling3d_3/split:output:11up_sampling3d_3/split:output:12up_sampling3d_3/split:output:12up_sampling3d_3/split:output:13up_sampling3d_3/split:output:13up_sampling3d_3/split:output:14up_sampling3d_3/split:output:14up_sampling3d_3/split:output:15up_sampling3d_3/split:output:15up_sampling3d_3/split:output:16up_sampling3d_3/split:output:16up_sampling3d_3/split:output:17up_sampling3d_3/split:output:17up_sampling3d_3/split:output:18up_sampling3d_3/split:output:18up_sampling3d_3/split:output:19up_sampling3d_3/split:output:19up_sampling3d_3/split:output:20up_sampling3d_3/split:output:20up_sampling3d_3/split:output:21up_sampling3d_3/split:output:21up_sampling3d_3/split:output:22up_sampling3d_3/split:output:22up_sampling3d_3/split:output:23up_sampling3d_3/split:output:23up_sampling3d_3/split:output:24up_sampling3d_3/split:output:24up_sampling3d_3/split:output:25up_sampling3d_3/split:output:25up_sampling3d_3/split:output:26up_sampling3d_3/split:output:26up_sampling3d_3/split:output:27up_sampling3d_3/split:output:27up_sampling3d_3/split:output:28up_sampling3d_3/split:output:28up_sampling3d_3/split:output:29up_sampling3d_3/split:output:29up_sampling3d_3/split:output:30up_sampling3d_3/split:output:30up_sampling3d_3/split:output:31up_sampling3d_3/split:output:31up_sampling3d_3/split:output:32up_sampling3d_3/split:output:32up_sampling3d_3/split:output:33up_sampling3d_3/split:output:33up_sampling3d_3/split:output:34up_sampling3d_3/split:output:34up_sampling3d_3/split:output:35up_sampling3d_3/split:output:35up_sampling3d_3/split:output:36up_sampling3d_3/split:output:36up_sampling3d_3/split:output:37up_sampling3d_3/split:output:37up_sampling3d_3/split:output:38up_sampling3d_3/split:output:38up_sampling3d_3/split:output:39up_sampling3d_3/split:output:39up_sampling3d_3/split:output:40up_sampling3d_3/split:output:40up_sampling3d_3/split:output:41up_sampling3d_3/split:output:41up_sampling3d_3/split:output:42up_sampling3d_3/split:output:42up_sampling3d_3/split:output:43up_sampling3d_3/split:output:43up_sampling3d_3/split:output:44up_sampling3d_3/split:output:44up_sampling3d_3/split:output:45up_sampling3d_3/split:output:45up_sampling3d_3/split:output:46up_sampling3d_3/split:output:46up_sampling3d_3/split:output:47up_sampling3d_3/split:output:47up_sampling3d_3/split:output:48up_sampling3d_3/split:output:48up_sampling3d_3/split:output:49up_sampling3d_3/split:output:49up_sampling3d_3/split:output:50up_sampling3d_3/split:output:50up_sampling3d_3/split:output:51up_sampling3d_3/split:output:51up_sampling3d_3/split:output:52up_sampling3d_3/split:output:52up_sampling3d_3/split:output:53up_sampling3d_3/split:output:53up_sampling3d_3/split:output:54up_sampling3d_3/split:output:54up_sampling3d_3/split:output:55up_sampling3d_3/split:output:55up_sampling3d_3/split:output:56up_sampling3d_3/split:output:56up_sampling3d_3/split:output:57up_sampling3d_3/split:output:57up_sampling3d_3/split:output:58up_sampling3d_3/split:output:58up_sampling3d_3/split:output:59up_sampling3d_3/split:output:59up_sampling3d_3/split:output:60up_sampling3d_3/split:output:60up_sampling3d_3/split:output:61up_sampling3d_3/split:output:61up_sampling3d_3/split:output:62up_sampling3d_3/split:output:62up_sampling3d_3/split:output:63up_sampling3d_3/split:output:63up_sampling3d_3/split:output:64up_sampling3d_3/split:output:64up_sampling3d_3/split:output:65up_sampling3d_3/split:output:65up_sampling3d_3/split:output:66up_sampling3d_3/split:output:66up_sampling3d_3/split:output:67up_sampling3d_3/split:output:67up_sampling3d_3/split:output:68up_sampling3d_3/split:output:68up_sampling3d_3/split:output:69up_sampling3d_3/split:output:69up_sampling3d_3/split:output:70up_sampling3d_3/split:output:70up_sampling3d_3/split:output:71up_sampling3d_3/split:output:71up_sampling3d_3/split:output:72up_sampling3d_3/split:output:72up_sampling3d_3/split:output:73up_sampling3d_3/split:output:73up_sampling3d_3/split:output:74up_sampling3d_3/split:output:74up_sampling3d_3/split:output:75up_sampling3d_3/split:output:75up_sampling3d_3/split:output:76up_sampling3d_3/split:output:76up_sampling3d_3/split:output:77up_sampling3d_3/split:output:77up_sampling3d_3/split:output:78up_sampling3d_3/split:output:78up_sampling3d_3/split:output:79up_sampling3d_3/split:output:79up_sampling3d_3/split:output:80up_sampling3d_3/split:output:80up_sampling3d_3/split:output:81up_sampling3d_3/split:output:81up_sampling3d_3/split:output:82up_sampling3d_3/split:output:82up_sampling3d_3/split:output:83up_sampling3d_3/split:output:83up_sampling3d_3/split:output:84up_sampling3d_3/split:output:84up_sampling3d_3/split:output:85up_sampling3d_3/split:output:85up_sampling3d_3/split:output:86up_sampling3d_3/split:output:86up_sampling3d_3/split:output:87up_sampling3d_3/split:output:87up_sampling3d_3/split:output:88up_sampling3d_3/split:output:88up_sampling3d_3/split:output:89up_sampling3d_3/split:output:89up_sampling3d_3/split:output:90up_sampling3d_3/split:output:90up_sampling3d_3/split:output:91up_sampling3d_3/split:output:91up_sampling3d_3/split:output:92up_sampling3d_3/split:output:92up_sampling3d_3/split:output:93up_sampling3d_3/split:output:93up_sampling3d_3/split:output:94up_sampling3d_3/split:output:94up_sampling3d_3/split:output:95up_sampling3d_3/split:output:95$up_sampling3d_3/concat/axis:output:0*
N?*
T0*4
_output_shapes"
 :??????????``@2
up_sampling3d_3/concatt
up_sampling3d_3/Const_1Const*
_output_shapes
: *
dtype0*
value	B :`2
up_sampling3d_3/Const_1?
!up_sampling3d_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!up_sampling3d_3/split_1/split_dim?
up_sampling3d_3/split_1Split*up_sampling3d_3/split_1/split_dim:output:0up_sampling3d_3/concat:output:0*
T0*?
_output_shapes?
?:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@*
	num_split`2
up_sampling3d_3/split_1?
up_sampling3d_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_3/concat_1/axis?5
up_sampling3d_3/concat_1ConcatV2 up_sampling3d_3/split_1:output:0 up_sampling3d_3/split_1:output:0 up_sampling3d_3/split_1:output:1 up_sampling3d_3/split_1:output:1 up_sampling3d_3/split_1:output:2 up_sampling3d_3/split_1:output:2 up_sampling3d_3/split_1:output:3 up_sampling3d_3/split_1:output:3 up_sampling3d_3/split_1:output:4 up_sampling3d_3/split_1:output:4 up_sampling3d_3/split_1:output:5 up_sampling3d_3/split_1:output:5 up_sampling3d_3/split_1:output:6 up_sampling3d_3/split_1:output:6 up_sampling3d_3/split_1:output:7 up_sampling3d_3/split_1:output:7 up_sampling3d_3/split_1:output:8 up_sampling3d_3/split_1:output:8 up_sampling3d_3/split_1:output:9 up_sampling3d_3/split_1:output:9!up_sampling3d_3/split_1:output:10!up_sampling3d_3/split_1:output:10!up_sampling3d_3/split_1:output:11!up_sampling3d_3/split_1:output:11!up_sampling3d_3/split_1:output:12!up_sampling3d_3/split_1:output:12!up_sampling3d_3/split_1:output:13!up_sampling3d_3/split_1:output:13!up_sampling3d_3/split_1:output:14!up_sampling3d_3/split_1:output:14!up_sampling3d_3/split_1:output:15!up_sampling3d_3/split_1:output:15!up_sampling3d_3/split_1:output:16!up_sampling3d_3/split_1:output:16!up_sampling3d_3/split_1:output:17!up_sampling3d_3/split_1:output:17!up_sampling3d_3/split_1:output:18!up_sampling3d_3/split_1:output:18!up_sampling3d_3/split_1:output:19!up_sampling3d_3/split_1:output:19!up_sampling3d_3/split_1:output:20!up_sampling3d_3/split_1:output:20!up_sampling3d_3/split_1:output:21!up_sampling3d_3/split_1:output:21!up_sampling3d_3/split_1:output:22!up_sampling3d_3/split_1:output:22!up_sampling3d_3/split_1:output:23!up_sampling3d_3/split_1:output:23!up_sampling3d_3/split_1:output:24!up_sampling3d_3/split_1:output:24!up_sampling3d_3/split_1:output:25!up_sampling3d_3/split_1:output:25!up_sampling3d_3/split_1:output:26!up_sampling3d_3/split_1:output:26!up_sampling3d_3/split_1:output:27!up_sampling3d_3/split_1:output:27!up_sampling3d_3/split_1:output:28!up_sampling3d_3/split_1:output:28!up_sampling3d_3/split_1:output:29!up_sampling3d_3/split_1:output:29!up_sampling3d_3/split_1:output:30!up_sampling3d_3/split_1:output:30!up_sampling3d_3/split_1:output:31!up_sampling3d_3/split_1:output:31!up_sampling3d_3/split_1:output:32!up_sampling3d_3/split_1:output:32!up_sampling3d_3/split_1:output:33!up_sampling3d_3/split_1:output:33!up_sampling3d_3/split_1:output:34!up_sampling3d_3/split_1:output:34!up_sampling3d_3/split_1:output:35!up_sampling3d_3/split_1:output:35!up_sampling3d_3/split_1:output:36!up_sampling3d_3/split_1:output:36!up_sampling3d_3/split_1:output:37!up_sampling3d_3/split_1:output:37!up_sampling3d_3/split_1:output:38!up_sampling3d_3/split_1:output:38!up_sampling3d_3/split_1:output:39!up_sampling3d_3/split_1:output:39!up_sampling3d_3/split_1:output:40!up_sampling3d_3/split_1:output:40!up_sampling3d_3/split_1:output:41!up_sampling3d_3/split_1:output:41!up_sampling3d_3/split_1:output:42!up_sampling3d_3/split_1:output:42!up_sampling3d_3/split_1:output:43!up_sampling3d_3/split_1:output:43!up_sampling3d_3/split_1:output:44!up_sampling3d_3/split_1:output:44!up_sampling3d_3/split_1:output:45!up_sampling3d_3/split_1:output:45!up_sampling3d_3/split_1:output:46!up_sampling3d_3/split_1:output:46!up_sampling3d_3/split_1:output:47!up_sampling3d_3/split_1:output:47!up_sampling3d_3/split_1:output:48!up_sampling3d_3/split_1:output:48!up_sampling3d_3/split_1:output:49!up_sampling3d_3/split_1:output:49!up_sampling3d_3/split_1:output:50!up_sampling3d_3/split_1:output:50!up_sampling3d_3/split_1:output:51!up_sampling3d_3/split_1:output:51!up_sampling3d_3/split_1:output:52!up_sampling3d_3/split_1:output:52!up_sampling3d_3/split_1:output:53!up_sampling3d_3/split_1:output:53!up_sampling3d_3/split_1:output:54!up_sampling3d_3/split_1:output:54!up_sampling3d_3/split_1:output:55!up_sampling3d_3/split_1:output:55!up_sampling3d_3/split_1:output:56!up_sampling3d_3/split_1:output:56!up_sampling3d_3/split_1:output:57!up_sampling3d_3/split_1:output:57!up_sampling3d_3/split_1:output:58!up_sampling3d_3/split_1:output:58!up_sampling3d_3/split_1:output:59!up_sampling3d_3/split_1:output:59!up_sampling3d_3/split_1:output:60!up_sampling3d_3/split_1:output:60!up_sampling3d_3/split_1:output:61!up_sampling3d_3/split_1:output:61!up_sampling3d_3/split_1:output:62!up_sampling3d_3/split_1:output:62!up_sampling3d_3/split_1:output:63!up_sampling3d_3/split_1:output:63!up_sampling3d_3/split_1:output:64!up_sampling3d_3/split_1:output:64!up_sampling3d_3/split_1:output:65!up_sampling3d_3/split_1:output:65!up_sampling3d_3/split_1:output:66!up_sampling3d_3/split_1:output:66!up_sampling3d_3/split_1:output:67!up_sampling3d_3/split_1:output:67!up_sampling3d_3/split_1:output:68!up_sampling3d_3/split_1:output:68!up_sampling3d_3/split_1:output:69!up_sampling3d_3/split_1:output:69!up_sampling3d_3/split_1:output:70!up_sampling3d_3/split_1:output:70!up_sampling3d_3/split_1:output:71!up_sampling3d_3/split_1:output:71!up_sampling3d_3/split_1:output:72!up_sampling3d_3/split_1:output:72!up_sampling3d_3/split_1:output:73!up_sampling3d_3/split_1:output:73!up_sampling3d_3/split_1:output:74!up_sampling3d_3/split_1:output:74!up_sampling3d_3/split_1:output:75!up_sampling3d_3/split_1:output:75!up_sampling3d_3/split_1:output:76!up_sampling3d_3/split_1:output:76!up_sampling3d_3/split_1:output:77!up_sampling3d_3/split_1:output:77!up_sampling3d_3/split_1:output:78!up_sampling3d_3/split_1:output:78!up_sampling3d_3/split_1:output:79!up_sampling3d_3/split_1:output:79!up_sampling3d_3/split_1:output:80!up_sampling3d_3/split_1:output:80!up_sampling3d_3/split_1:output:81!up_sampling3d_3/split_1:output:81!up_sampling3d_3/split_1:output:82!up_sampling3d_3/split_1:output:82!up_sampling3d_3/split_1:output:83!up_sampling3d_3/split_1:output:83!up_sampling3d_3/split_1:output:84!up_sampling3d_3/split_1:output:84!up_sampling3d_3/split_1:output:85!up_sampling3d_3/split_1:output:85!up_sampling3d_3/split_1:output:86!up_sampling3d_3/split_1:output:86!up_sampling3d_3/split_1:output:87!up_sampling3d_3/split_1:output:87!up_sampling3d_3/split_1:output:88!up_sampling3d_3/split_1:output:88!up_sampling3d_3/split_1:output:89!up_sampling3d_3/split_1:output:89!up_sampling3d_3/split_1:output:90!up_sampling3d_3/split_1:output:90!up_sampling3d_3/split_1:output:91!up_sampling3d_3/split_1:output:91!up_sampling3d_3/split_1:output:92!up_sampling3d_3/split_1:output:92!up_sampling3d_3/split_1:output:93!up_sampling3d_3/split_1:output:93!up_sampling3d_3/split_1:output:94!up_sampling3d_3/split_1:output:94!up_sampling3d_3/split_1:output:95!up_sampling3d_3/split_1:output:95&up_sampling3d_3/concat_1/axis:output:0*
N?*
T0*5
_output_shapes#
!:???????????`@2
up_sampling3d_3/concat_1t
up_sampling3d_3/Const_2Const*
_output_shapes
: *
dtype0*
value	B :`2
up_sampling3d_3/Const_2?
!up_sampling3d_3/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!up_sampling3d_3/split_2/split_dim?
up_sampling3d_3/split_2Split*up_sampling3d_3/split_2/split_dim:output:0!up_sampling3d_3/concat_1:output:0*
T0*?
_output_shapes?
?:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@*
	num_split`2
up_sampling3d_3/split_2?
up_sampling3d_3/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_3/concat_2/axis?5
up_sampling3d_3/concat_2ConcatV2 up_sampling3d_3/split_2:output:0 up_sampling3d_3/split_2:output:0 up_sampling3d_3/split_2:output:1 up_sampling3d_3/split_2:output:1 up_sampling3d_3/split_2:output:2 up_sampling3d_3/split_2:output:2 up_sampling3d_3/split_2:output:3 up_sampling3d_3/split_2:output:3 up_sampling3d_3/split_2:output:4 up_sampling3d_3/split_2:output:4 up_sampling3d_3/split_2:output:5 up_sampling3d_3/split_2:output:5 up_sampling3d_3/split_2:output:6 up_sampling3d_3/split_2:output:6 up_sampling3d_3/split_2:output:7 up_sampling3d_3/split_2:output:7 up_sampling3d_3/split_2:output:8 up_sampling3d_3/split_2:output:8 up_sampling3d_3/split_2:output:9 up_sampling3d_3/split_2:output:9!up_sampling3d_3/split_2:output:10!up_sampling3d_3/split_2:output:10!up_sampling3d_3/split_2:output:11!up_sampling3d_3/split_2:output:11!up_sampling3d_3/split_2:output:12!up_sampling3d_3/split_2:output:12!up_sampling3d_3/split_2:output:13!up_sampling3d_3/split_2:output:13!up_sampling3d_3/split_2:output:14!up_sampling3d_3/split_2:output:14!up_sampling3d_3/split_2:output:15!up_sampling3d_3/split_2:output:15!up_sampling3d_3/split_2:output:16!up_sampling3d_3/split_2:output:16!up_sampling3d_3/split_2:output:17!up_sampling3d_3/split_2:output:17!up_sampling3d_3/split_2:output:18!up_sampling3d_3/split_2:output:18!up_sampling3d_3/split_2:output:19!up_sampling3d_3/split_2:output:19!up_sampling3d_3/split_2:output:20!up_sampling3d_3/split_2:output:20!up_sampling3d_3/split_2:output:21!up_sampling3d_3/split_2:output:21!up_sampling3d_3/split_2:output:22!up_sampling3d_3/split_2:output:22!up_sampling3d_3/split_2:output:23!up_sampling3d_3/split_2:output:23!up_sampling3d_3/split_2:output:24!up_sampling3d_3/split_2:output:24!up_sampling3d_3/split_2:output:25!up_sampling3d_3/split_2:output:25!up_sampling3d_3/split_2:output:26!up_sampling3d_3/split_2:output:26!up_sampling3d_3/split_2:output:27!up_sampling3d_3/split_2:output:27!up_sampling3d_3/split_2:output:28!up_sampling3d_3/split_2:output:28!up_sampling3d_3/split_2:output:29!up_sampling3d_3/split_2:output:29!up_sampling3d_3/split_2:output:30!up_sampling3d_3/split_2:output:30!up_sampling3d_3/split_2:output:31!up_sampling3d_3/split_2:output:31!up_sampling3d_3/split_2:output:32!up_sampling3d_3/split_2:output:32!up_sampling3d_3/split_2:output:33!up_sampling3d_3/split_2:output:33!up_sampling3d_3/split_2:output:34!up_sampling3d_3/split_2:output:34!up_sampling3d_3/split_2:output:35!up_sampling3d_3/split_2:output:35!up_sampling3d_3/split_2:output:36!up_sampling3d_3/split_2:output:36!up_sampling3d_3/split_2:output:37!up_sampling3d_3/split_2:output:37!up_sampling3d_3/split_2:output:38!up_sampling3d_3/split_2:output:38!up_sampling3d_3/split_2:output:39!up_sampling3d_3/split_2:output:39!up_sampling3d_3/split_2:output:40!up_sampling3d_3/split_2:output:40!up_sampling3d_3/split_2:output:41!up_sampling3d_3/split_2:output:41!up_sampling3d_3/split_2:output:42!up_sampling3d_3/split_2:output:42!up_sampling3d_3/split_2:output:43!up_sampling3d_3/split_2:output:43!up_sampling3d_3/split_2:output:44!up_sampling3d_3/split_2:output:44!up_sampling3d_3/split_2:output:45!up_sampling3d_3/split_2:output:45!up_sampling3d_3/split_2:output:46!up_sampling3d_3/split_2:output:46!up_sampling3d_3/split_2:output:47!up_sampling3d_3/split_2:output:47!up_sampling3d_3/split_2:output:48!up_sampling3d_3/split_2:output:48!up_sampling3d_3/split_2:output:49!up_sampling3d_3/split_2:output:49!up_sampling3d_3/split_2:output:50!up_sampling3d_3/split_2:output:50!up_sampling3d_3/split_2:output:51!up_sampling3d_3/split_2:output:51!up_sampling3d_3/split_2:output:52!up_sampling3d_3/split_2:output:52!up_sampling3d_3/split_2:output:53!up_sampling3d_3/split_2:output:53!up_sampling3d_3/split_2:output:54!up_sampling3d_3/split_2:output:54!up_sampling3d_3/split_2:output:55!up_sampling3d_3/split_2:output:55!up_sampling3d_3/split_2:output:56!up_sampling3d_3/split_2:output:56!up_sampling3d_3/split_2:output:57!up_sampling3d_3/split_2:output:57!up_sampling3d_3/split_2:output:58!up_sampling3d_3/split_2:output:58!up_sampling3d_3/split_2:output:59!up_sampling3d_3/split_2:output:59!up_sampling3d_3/split_2:output:60!up_sampling3d_3/split_2:output:60!up_sampling3d_3/split_2:output:61!up_sampling3d_3/split_2:output:61!up_sampling3d_3/split_2:output:62!up_sampling3d_3/split_2:output:62!up_sampling3d_3/split_2:output:63!up_sampling3d_3/split_2:output:63!up_sampling3d_3/split_2:output:64!up_sampling3d_3/split_2:output:64!up_sampling3d_3/split_2:output:65!up_sampling3d_3/split_2:output:65!up_sampling3d_3/split_2:output:66!up_sampling3d_3/split_2:output:66!up_sampling3d_3/split_2:output:67!up_sampling3d_3/split_2:output:67!up_sampling3d_3/split_2:output:68!up_sampling3d_3/split_2:output:68!up_sampling3d_3/split_2:output:69!up_sampling3d_3/split_2:output:69!up_sampling3d_3/split_2:output:70!up_sampling3d_3/split_2:output:70!up_sampling3d_3/split_2:output:71!up_sampling3d_3/split_2:output:71!up_sampling3d_3/split_2:output:72!up_sampling3d_3/split_2:output:72!up_sampling3d_3/split_2:output:73!up_sampling3d_3/split_2:output:73!up_sampling3d_3/split_2:output:74!up_sampling3d_3/split_2:output:74!up_sampling3d_3/split_2:output:75!up_sampling3d_3/split_2:output:75!up_sampling3d_3/split_2:output:76!up_sampling3d_3/split_2:output:76!up_sampling3d_3/split_2:output:77!up_sampling3d_3/split_2:output:77!up_sampling3d_3/split_2:output:78!up_sampling3d_3/split_2:output:78!up_sampling3d_3/split_2:output:79!up_sampling3d_3/split_2:output:79!up_sampling3d_3/split_2:output:80!up_sampling3d_3/split_2:output:80!up_sampling3d_3/split_2:output:81!up_sampling3d_3/split_2:output:81!up_sampling3d_3/split_2:output:82!up_sampling3d_3/split_2:output:82!up_sampling3d_3/split_2:output:83!up_sampling3d_3/split_2:output:83!up_sampling3d_3/split_2:output:84!up_sampling3d_3/split_2:output:84!up_sampling3d_3/split_2:output:85!up_sampling3d_3/split_2:output:85!up_sampling3d_3/split_2:output:86!up_sampling3d_3/split_2:output:86!up_sampling3d_3/split_2:output:87!up_sampling3d_3/split_2:output:87!up_sampling3d_3/split_2:output:88!up_sampling3d_3/split_2:output:88!up_sampling3d_3/split_2:output:89!up_sampling3d_3/split_2:output:89!up_sampling3d_3/split_2:output:90!up_sampling3d_3/split_2:output:90!up_sampling3d_3/split_2:output:91!up_sampling3d_3/split_2:output:91!up_sampling3d_3/split_2:output:92!up_sampling3d_3/split_2:output:92!up_sampling3d_3/split_2:output:93!up_sampling3d_3/split_2:output:93!up_sampling3d_3/split_2:output:94!up_sampling3d_3/split_2:output:94!up_sampling3d_3/split_2:output:95!up_sampling3d_3/split_2:output:95&up_sampling3d_3/concat_2/axis:output:0*
N?*
T0*6
_output_shapes$
": ????????????@2
up_sampling3d_3/concat_2x
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_4/concat/axis?
concatenate_4/concatConcatV2!up_sampling3d_3/concat_2:output:0concatenate/concat:output:0"concatenate_4/concat/axis:output:0*
N*
T0*6
_output_shapes$
": ????????????B2
concatenate_4/concat?
conv3d_9/Conv3D/ReadVariableOpReadVariableOp'conv3d_9_conv3d_readvariableop_resource**
_output_shapes
:B *
dtype02 
conv3d_9/Conv3D/ReadVariableOp?
conv3d_9/Conv3DConv3Dconcatenate_4/concat:output:0&conv3d_9/Conv3D/ReadVariableOp:value:0*
T0*6
_output_shapes$
": ???????????? *
paddingSAME*
strides	
2
conv3d_9/Conv3D?
conv3d_9/BiasAdd/ReadVariableOpReadVariableOp(conv3d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv3d_9/BiasAdd/ReadVariableOp?
conv3d_9/BiasAddBiasAddconv3d_9/Conv3D:output:0'conv3d_9/BiasAdd/ReadVariableOp:value:0*
T0*6
_output_shapes$
": ???????????? 2
conv3d_9/BiasAdd?
leaky_re_lu_9/LeakyRelu	LeakyReluconv3d_9/BiasAdd:output:0*6
_output_shapes$
": ???????????? 2
leaky_re_lu_9/LeakyRelu?
conv3d_10/Conv3D/ReadVariableOpReadVariableOp(conv3d_10_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype02!
conv3d_10/Conv3D/ReadVariableOp?
conv3d_10/Conv3DConv3D%leaky_re_lu_9/LeakyRelu:activations:0'conv3d_10/Conv3D/ReadVariableOp:value:0*
T0*6
_output_shapes$
": ???????????? *
paddingSAME*
strides	
2
conv3d_10/Conv3D?
 conv3d_10/BiasAdd/ReadVariableOpReadVariableOp)conv3d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3d_10/BiasAdd/ReadVariableOp?
conv3d_10/BiasAddBiasAddconv3d_10/Conv3D:output:0(conv3d_10/BiasAdd/ReadVariableOp:value:0*
T0*6
_output_shapes$
": ???????????? 2
conv3d_10/BiasAdd?
leaky_re_lu_10/LeakyRelu	LeakyReluconv3d_10/BiasAdd:output:0*6
_output_shapes$
": ???????????? 2
leaky_re_lu_10/LeakyRelu?
disp/Conv3D/ReadVariableOpReadVariableOp#disp_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02
disp/Conv3D/ReadVariableOp?
disp/Conv3DConv3D&leaky_re_lu_10/LeakyRelu:activations:0"disp/Conv3D/ReadVariableOp:value:0*
T0*6
_output_shapes$
": ????????????*
paddingSAME*
strides	
2
disp/Conv3D?
disp/BiasAdd/ReadVariableOpReadVariableOp$disp_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
disp/BiasAdd/ReadVariableOp?
disp/BiasAddBiasAdddisp/Conv3D:output:0#disp/BiasAdd/ReadVariableOp:value:0*
T0*6
_output_shapes$
": ????????????2
disp/BiasAdd?
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
transformer/Reshape_1Reshapedisp/BiasAdd:output:0$transformer/Reshape_1/shape:output:0*
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
 *-
body%R#
!transformer_map_while_body_276055*-
cond%R#
!transformer_map_while_cond_276054*!
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

Identity|

Identity_1Identitydisp/BiasAdd:output:0*
T0*6
_output_shapes$
": ????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ????????????: ????????????:::::::::::::::::::::::::` \
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/0:`\
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/1:
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
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
s
I__inference_concatenate_1_layer_call_and_return_conditional_losses_272824

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????2
concatp
IdentityIdentityconcat:output:0*
T0*4
_output_shapes"
 :??????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????@:?????????@:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs:[W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_276912

inputs
identity`
	LeakyRelu	LeakyReluinputs*3
_output_shapes!
:?????????```@2
	LeakyReluw
IdentityIdentityLeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????```@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????```@:[ W
3
_output_shapes!
:?????????```@
 
_user_specified_nameinputs
?A
?

__inference__traced_save_277703
file_prefix,
(savev2_conv3d_kernel_read_readvariableop*
&savev2_conv3d_bias_read_readvariableop1
-savev2_conv3d_1_13_kernel_read_readvariableop/
+savev2_conv3d_1_13_bias_read_readvariableop1
-savev2_conv3d_2_13_kernel_read_readvariableop/
+savev2_conv3d_2_13_bias_read_readvariableop1
-savev2_conv3d_3_13_kernel_read_readvariableop/
+savev2_conv3d_3_13_bias_read_readvariableop1
-savev2_conv3d_4_13_kernel_read_readvariableop/
+savev2_conv3d_4_13_bias_read_readvariableop1
-savev2_conv3d_5_13_kernel_read_readvariableop/
+savev2_conv3d_5_13_bias_read_readvariableop1
-savev2_conv3d_6_13_kernel_read_readvariableop/
+savev2_conv3d_6_13_bias_read_readvariableop1
-savev2_conv3d_7_13_kernel_read_readvariableop/
+savev2_conv3d_7_13_bias_read_readvariableop1
-savev2_conv3d_8_13_kernel_read_readvariableop/
+savev2_conv3d_8_13_bias_read_readvariableop1
-savev2_conv3d_9_13_kernel_read_readvariableop/
+savev2_conv3d_9_13_bias_read_readvariableop2
.savev2_conv3d_10_13_kernel_read_readvariableop0
,savev2_conv3d_10_13_bias_read_readvariableop-
)savev2_disp_10_kernel_read_readvariableop+
'savev2_disp_10_bias_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
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
value3B1 B+_temp_bb7478fb5ba14fb19977e8e35267ff9a/part2	
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
value	B :2

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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?

value?
B?
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv3d_kernel_read_readvariableop&savev2_conv3d_bias_read_readvariableop-savev2_conv3d_1_13_kernel_read_readvariableop+savev2_conv3d_1_13_bias_read_readvariableop-savev2_conv3d_2_13_kernel_read_readvariableop+savev2_conv3d_2_13_bias_read_readvariableop-savev2_conv3d_3_13_kernel_read_readvariableop+savev2_conv3d_3_13_bias_read_readvariableop-savev2_conv3d_4_13_kernel_read_readvariableop+savev2_conv3d_4_13_bias_read_readvariableop-savev2_conv3d_5_13_kernel_read_readvariableop+savev2_conv3d_5_13_bias_read_readvariableop-savev2_conv3d_6_13_kernel_read_readvariableop+savev2_conv3d_6_13_bias_read_readvariableop-savev2_conv3d_7_13_kernel_read_readvariableop+savev2_conv3d_7_13_bias_read_readvariableop-savev2_conv3d_8_13_kernel_read_readvariableop+savev2_conv3d_8_13_bias_read_readvariableop-savev2_conv3d_9_13_kernel_read_readvariableop+savev2_conv3d_9_13_bias_read_readvariableop.savev2_conv3d_10_13_kernel_read_readvariableop,savev2_conv3d_10_13_bias_read_readvariableop)savev2_disp_10_kernel_read_readvariableop'savev2_disp_10_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *&
dtypes
22
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : @:@:@@:@:@@:@:@@:@:?@:@:?@:@:`@:@:@@:@:B : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_output_shapes
: : 

_output_shapes
: :0,
*
_output_shapes
: @: 

_output_shapes
:@:0,
*
_output_shapes
:@@: 

_output_shapes
:@:0,
*
_output_shapes
:@@: 

_output_shapes
:@:0	,
*
_output_shapes
:@@: 


_output_shapes
:@:1-
+
_output_shapes
:?@: 

_output_shapes
:@:1-
+
_output_shapes
:?@: 

_output_shapes
:@:0,
*
_output_shapes
:`@: 

_output_shapes
:@:0,
*
_output_shapes
:@@: 

_output_shapes
:@:0,
*
_output_shapes
:B : 

_output_shapes
: :0,
*
_output_shapes
:  : 

_output_shapes
: :0,
*
_output_shapes
: : 

_output_shapes
::

_output_shapes
: 
?i
?
"__inference__traced_restore_277787
file_prefix"
assignvariableop_conv3d_kernel"
assignvariableop_1_conv3d_bias)
%assignvariableop_2_conv3d_1_13_kernel'
#assignvariableop_3_conv3d_1_13_bias)
%assignvariableop_4_conv3d_2_13_kernel'
#assignvariableop_5_conv3d_2_13_bias)
%assignvariableop_6_conv3d_3_13_kernel'
#assignvariableop_7_conv3d_3_13_bias)
%assignvariableop_8_conv3d_4_13_kernel'
#assignvariableop_9_conv3d_4_13_bias*
&assignvariableop_10_conv3d_5_13_kernel(
$assignvariableop_11_conv3d_5_13_bias*
&assignvariableop_12_conv3d_6_13_kernel(
$assignvariableop_13_conv3d_6_13_bias*
&assignvariableop_14_conv3d_7_13_kernel(
$assignvariableop_15_conv3d_7_13_bias*
&assignvariableop_16_conv3d_8_13_kernel(
$assignvariableop_17_conv3d_8_13_bias*
&assignvariableop_18_conv3d_9_13_kernel(
$assignvariableop_19_conv3d_9_13_bias+
'assignvariableop_20_conv3d_10_13_kernel)
%assignvariableop_21_conv3d_10_13_bias&
"assignvariableop_22_disp_10_kernel$
 assignvariableop_23_disp_10_bias
identity_25??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?

value?
B?
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*t
_output_shapesb
`::::::::::::::::::::::::*&
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv3d_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv3d_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp%assignvariableop_2_conv3d_1_13_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp#assignvariableop_3_conv3d_1_13_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_conv3d_2_13_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_conv3d_2_13_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_conv3d_3_13_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_conv3d_3_13_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_conv3d_4_13_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv3d_4_13_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_conv3d_5_13_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv3d_5_13_biasIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_conv3d_6_13_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv3d_6_13_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_conv3d_7_13_kernelIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv3d_7_13_biasIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_conv3d_8_13_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp$assignvariableop_17_conv3d_8_13_biasIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp&assignvariableop_18_conv3d_9_13_kernelIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp$assignvariableop_19_conv3d_9_13_biasIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_conv3d_10_13_kernelIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp%assignvariableop_21_conv3d_10_13_biasIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp"assignvariableop_22_disp_10_kernelIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp assignvariableop_23_disp_10_biasIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24?
Identity_25IdentityIdentity_24:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_25"#
identity_25Identity_25:output:0*u
_input_shapesd
b: ::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:
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
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
)model_1_transformer_map_while_cond_272090.
*model_1_transformer_map_while_loop_counter)
%model_1_transformer_map_strided_slice
placeholder
placeholder_1.
*less_model_1_transformer_map_strided_sliceF
Bmodel_1_transformer_map_while_cond_272090___redundant_placeholder0F
Bmodel_1_transformer_map_while_cond_272090___redundant_placeholder1
identity
n
LessLessplaceholder*less_model_1_transformer_map_strided_slice*
T0*
_output_shapes
: 2
Less?
Less_1Less*model_1_transformer_map_while_loop_counter%model_1_transformer_map_strided_slice*
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
?
~
)__inference_conv3d_5_layer_call_fn_272520

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8????????????????????????????????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_2725102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:9?????????????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:w s
O
_output_shapes=
;:9?????????????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
f
J__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_273557

inputs
identityc
	LeakyRelu	LeakyReluinputs*6
_output_shapes$
": ???????????? 2
	LeakyReluz
IdentityIdentityLeakyRelu:activations:0*
T0*6
_output_shapes$
": ???????????? 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ???????????? :^ Z
6
_output_shapes$
": ???????????? 
 
_user_specified_nameinputs
??
?
)model_1_transformer_map_while_body_272091.
*model_1_transformer_map_while_loop_counter)
%model_1_transformer_map_strided_slice
placeholder
placeholder_1-
)model_1_transformer_map_strided_slice_1_0i
etensorarrayv2read_tensorlistgetitem_model_1_transformer_map_tensorarrayunstack_tensorlistfromtensor_0m
itensorarrayv2read_1_tensorlistgetitem_model_1_transformer_map_tensorarrayunstack_1_tensorlistfromtensor_0
identity

identity_1

identity_2

identity_3+
'model_1_transformer_map_strided_slice_1g
ctensorarrayv2read_tensorlistgetitem_model_1_transformer_map_tensorarrayunstack_tensorlistfromtensork
gtensorarrayv2read_1_tensorlistgetitem_model_1_transformer_map_tensorarrayunstack_1_tensorlistfromtensor?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemetensorarrayv2read_tensorlistgetitem_model_1_transformer_map_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*)
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
%TensorArrayV2Read_1/TensorListGetItemTensorListGetItemitensorarrayv2read_1_tensorlistgetitem_model_1_transformer_map_tensorarrayunstack_1_tensorlistfromtensor_0placeholder<TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*)
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
add_31AddV2*model_1_transformer_map_while_loop_counteradd_31/y:output:0*
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

Identity_1Identity%model_1_transformer_map_strided_slice*
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
'model_1_transformer_map_strided_slice_1)model_1_transformer_map_strided_slice_1_0"?
gtensorarrayv2read_1_tensorlistgetitem_model_1_transformer_map_tensorarrayunstack_1_tensorlistfromtensoritensorarrayv2read_1_tensorlistgetitem_model_1_transformer_map_tensorarrayunstack_1_tensorlistfromtensor_0"?
ctensorarrayv2read_tensorlistgetitem_model_1_transformer_map_tensorarrayunstack_tensorlistfromtensoretensorarrayv2read_tensorlistgetitem_model_1_transformer_map_tensorarrayunstack_tensorlistfromtensor_0*!
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
u
I__inference_concatenate_3_layer_call_and_return_conditional_losses_276901
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*3
_output_shapes!
:?????????````2
concato
IdentityIdentityconcat:output:0*
T0*3
_output_shapes!
:?????????````2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????```@:?????????``` :] Y
3
_output_shapes!
:?????????```@
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:?????????``` 
"
_user_specified_name
inputs/1
?
Z
.__inference_concatenate_2_layer_call_fn_276719
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*4
_output_shapes"
 :?????????000?* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_2729542
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :?????????000?2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????000@:?????????000@:] Y
3
_output_shapes!
:?????????000@
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:?????????000@
"
_user_specified_name
inputs/1
??
g
K__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_273505

inputs
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :`2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0inputs*
T0*?
_output_shapes?
?:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@*
	num_split`2
split\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15split:output:16split:output:16split:output:17split:output:17split:output:18split:output:18split:output:19split:output:19split:output:20split:output:20split:output:21split:output:21split:output:22split:output:22split:output:23split:output:23split:output:24split:output:24split:output:25split:output:25split:output:26split:output:26split:output:27split:output:27split:output:28split:output:28split:output:29split:output:29split:output:30split:output:30split:output:31split:output:31split:output:32split:output:32split:output:33split:output:33split:output:34split:output:34split:output:35split:output:35split:output:36split:output:36split:output:37split:output:37split:output:38split:output:38split:output:39split:output:39split:output:40split:output:40split:output:41split:output:41split:output:42split:output:42split:output:43split:output:43split:output:44split:output:44split:output:45split:output:45split:output:46split:output:46split:output:47split:output:47split:output:48split:output:48split:output:49split:output:49split:output:50split:output:50split:output:51split:output:51split:output:52split:output:52split:output:53split:output:53split:output:54split:output:54split:output:55split:output:55split:output:56split:output:56split:output:57split:output:57split:output:58split:output:58split:output:59split:output:59split:output:60split:output:60split:output:61split:output:61split:output:62split:output:62split:output:63split:output:63split:output:64split:output:64split:output:65split:output:65split:output:66split:output:66split:output:67split:output:67split:output:68split:output:68split:output:69split:output:69split:output:70split:output:70split:output:71split:output:71split:output:72split:output:72split:output:73split:output:73split:output:74split:output:74split:output:75split:output:75split:output:76split:output:76split:output:77split:output:77split:output:78split:output:78split:output:79split:output:79split:output:80split:output:80split:output:81split:output:81split:output:82split:output:82split:output:83split:output:83split:output:84split:output:84split:output:85split:output:85split:output:86split:output:86split:output:87split:output:87split:output:88split:output:88split:output:89split:output:89split:output:90split:output:90split:output:91split:output:91split:output:92split:output:92split:output:93split:output:93split:output:94split:output:94split:output:95split:output:95concat/axis:output:0*
N?*
T0*4
_output_shapes"
 :??????????``@2
concatT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :`2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*?
_output_shapes?
?:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@*
	num_split`2	
split_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15split_1:output:16split_1:output:16split_1:output:17split_1:output:17split_1:output:18split_1:output:18split_1:output:19split_1:output:19split_1:output:20split_1:output:20split_1:output:21split_1:output:21split_1:output:22split_1:output:22split_1:output:23split_1:output:23split_1:output:24split_1:output:24split_1:output:25split_1:output:25split_1:output:26split_1:output:26split_1:output:27split_1:output:27split_1:output:28split_1:output:28split_1:output:29split_1:output:29split_1:output:30split_1:output:30split_1:output:31split_1:output:31split_1:output:32split_1:output:32split_1:output:33split_1:output:33split_1:output:34split_1:output:34split_1:output:35split_1:output:35split_1:output:36split_1:output:36split_1:output:37split_1:output:37split_1:output:38split_1:output:38split_1:output:39split_1:output:39split_1:output:40split_1:output:40split_1:output:41split_1:output:41split_1:output:42split_1:output:42split_1:output:43split_1:output:43split_1:output:44split_1:output:44split_1:output:45split_1:output:45split_1:output:46split_1:output:46split_1:output:47split_1:output:47split_1:output:48split_1:output:48split_1:output:49split_1:output:49split_1:output:50split_1:output:50split_1:output:51split_1:output:51split_1:output:52split_1:output:52split_1:output:53split_1:output:53split_1:output:54split_1:output:54split_1:output:55split_1:output:55split_1:output:56split_1:output:56split_1:output:57split_1:output:57split_1:output:58split_1:output:58split_1:output:59split_1:output:59split_1:output:60split_1:output:60split_1:output:61split_1:output:61split_1:output:62split_1:output:62split_1:output:63split_1:output:63split_1:output:64split_1:output:64split_1:output:65split_1:output:65split_1:output:66split_1:output:66split_1:output:67split_1:output:67split_1:output:68split_1:output:68split_1:output:69split_1:output:69split_1:output:70split_1:output:70split_1:output:71split_1:output:71split_1:output:72split_1:output:72split_1:output:73split_1:output:73split_1:output:74split_1:output:74split_1:output:75split_1:output:75split_1:output:76split_1:output:76split_1:output:77split_1:output:77split_1:output:78split_1:output:78split_1:output:79split_1:output:79split_1:output:80split_1:output:80split_1:output:81split_1:output:81split_1:output:82split_1:output:82split_1:output:83split_1:output:83split_1:output:84split_1:output:84split_1:output:85split_1:output:85split_1:output:86split_1:output:86split_1:output:87split_1:output:87split_1:output:88split_1:output:88split_1:output:89split_1:output:89split_1:output:90split_1:output:90split_1:output:91split_1:output:91split_1:output:92split_1:output:92split_1:output:93split_1:output:93split_1:output:94split_1:output:94split_1:output:95split_1:output:95concat_1/axis:output:0*
N?*
T0*5
_output_shapes#
!:???????????`@2

concat_1T
Const_2Const*
_output_shapes
: *
dtype0*
value	B :`2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*?
_output_shapes?
?:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@*
	num_split`2	
split_2`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis?
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15split_2:output:16split_2:output:16split_2:output:17split_2:output:17split_2:output:18split_2:output:18split_2:output:19split_2:output:19split_2:output:20split_2:output:20split_2:output:21split_2:output:21split_2:output:22split_2:output:22split_2:output:23split_2:output:23split_2:output:24split_2:output:24split_2:output:25split_2:output:25split_2:output:26split_2:output:26split_2:output:27split_2:output:27split_2:output:28split_2:output:28split_2:output:29split_2:output:29split_2:output:30split_2:output:30split_2:output:31split_2:output:31split_2:output:32split_2:output:32split_2:output:33split_2:output:33split_2:output:34split_2:output:34split_2:output:35split_2:output:35split_2:output:36split_2:output:36split_2:output:37split_2:output:37split_2:output:38split_2:output:38split_2:output:39split_2:output:39split_2:output:40split_2:output:40split_2:output:41split_2:output:41split_2:output:42split_2:output:42split_2:output:43split_2:output:43split_2:output:44split_2:output:44split_2:output:45split_2:output:45split_2:output:46split_2:output:46split_2:output:47split_2:output:47split_2:output:48split_2:output:48split_2:output:49split_2:output:49split_2:output:50split_2:output:50split_2:output:51split_2:output:51split_2:output:52split_2:output:52split_2:output:53split_2:output:53split_2:output:54split_2:output:54split_2:output:55split_2:output:55split_2:output:56split_2:output:56split_2:output:57split_2:output:57split_2:output:58split_2:output:58split_2:output:59split_2:output:59split_2:output:60split_2:output:60split_2:output:61split_2:output:61split_2:output:62split_2:output:62split_2:output:63split_2:output:63split_2:output:64split_2:output:64split_2:output:65split_2:output:65split_2:output:66split_2:output:66split_2:output:67split_2:output:67split_2:output:68split_2:output:68split_2:output:69split_2:output:69split_2:output:70split_2:output:70split_2:output:71split_2:output:71split_2:output:72split_2:output:72split_2:output:73split_2:output:73split_2:output:74split_2:output:74split_2:output:75split_2:output:75split_2:output:76split_2:output:76split_2:output:77split_2:output:77split_2:output:78split_2:output:78split_2:output:79split_2:output:79split_2:output:80split_2:output:80split_2:output:81split_2:output:81split_2:output:82split_2:output:82split_2:output:83split_2:output:83split_2:output:84split_2:output:84split_2:output:85split_2:output:85split_2:output:86split_2:output:86split_2:output:87split_2:output:87split_2:output:88split_2:output:88split_2:output:89split_2:output:89split_2:output:90split_2:output:90split_2:output:91split_2:output:91split_2:output:92split_2:output:92split_2:output:93split_2:output:93split_2:output:94split_2:output:94split_2:output:95split_2:output:95concat_2/axis:output:0*
N?*
T0*6
_output_shapes$
": ????????????@2

concat_2t
IdentityIdentityconcat_2:output:0*
T0*6
_output_shapes$
": ????????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????```@:[ W
3
_output_shapes!
:?????????```@
 
_user_specified_nameinputs
?
~
)__inference_conv3d_9_layer_call_fn_272604

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8???????????????????????????????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_9_layer_call_and_return_conditional_losses_2725942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8????????????????????????????????????B::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????B
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
s
G__inference_concatenate_layer_call_and_return_conditional_losses_276477
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*6
_output_shapes$
": ????????????2
concatr
IdentityIdentityconcat:output:0*
T0*6
_output_shapes$
": ????????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D: ????????????: ????????????:` \
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/0:`\
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/1
??
g
K__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_277231

inputs
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :`2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0inputs*
T0*?
_output_shapes?
?:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@*
	num_split`2
split\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15split:output:16split:output:16split:output:17split:output:17split:output:18split:output:18split:output:19split:output:19split:output:20split:output:20split:output:21split:output:21split:output:22split:output:22split:output:23split:output:23split:output:24split:output:24split:output:25split:output:25split:output:26split:output:26split:output:27split:output:27split:output:28split:output:28split:output:29split:output:29split:output:30split:output:30split:output:31split:output:31split:output:32split:output:32split:output:33split:output:33split:output:34split:output:34split:output:35split:output:35split:output:36split:output:36split:output:37split:output:37split:output:38split:output:38split:output:39split:output:39split:output:40split:output:40split:output:41split:output:41split:output:42split:output:42split:output:43split:output:43split:output:44split:output:44split:output:45split:output:45split:output:46split:output:46split:output:47split:output:47split:output:48split:output:48split:output:49split:output:49split:output:50split:output:50split:output:51split:output:51split:output:52split:output:52split:output:53split:output:53split:output:54split:output:54split:output:55split:output:55split:output:56split:output:56split:output:57split:output:57split:output:58split:output:58split:output:59split:output:59split:output:60split:output:60split:output:61split:output:61split:output:62split:output:62split:output:63split:output:63split:output:64split:output:64split:output:65split:output:65split:output:66split:output:66split:output:67split:output:67split:output:68split:output:68split:output:69split:output:69split:output:70split:output:70split:output:71split:output:71split:output:72split:output:72split:output:73split:output:73split:output:74split:output:74split:output:75split:output:75split:output:76split:output:76split:output:77split:output:77split:output:78split:output:78split:output:79split:output:79split:output:80split:output:80split:output:81split:output:81split:output:82split:output:82split:output:83split:output:83split:output:84split:output:84split:output:85split:output:85split:output:86split:output:86split:output:87split:output:87split:output:88split:output:88split:output:89split:output:89split:output:90split:output:90split:output:91split:output:91split:output:92split:output:92split:output:93split:output:93split:output:94split:output:94split:output:95split:output:95concat/axis:output:0*
N?*
T0*4
_output_shapes"
 :??????????``@2
concatT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :`2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*?
_output_shapes?
?:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@*
	num_split`2	
split_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15split_1:output:16split_1:output:16split_1:output:17split_1:output:17split_1:output:18split_1:output:18split_1:output:19split_1:output:19split_1:output:20split_1:output:20split_1:output:21split_1:output:21split_1:output:22split_1:output:22split_1:output:23split_1:output:23split_1:output:24split_1:output:24split_1:output:25split_1:output:25split_1:output:26split_1:output:26split_1:output:27split_1:output:27split_1:output:28split_1:output:28split_1:output:29split_1:output:29split_1:output:30split_1:output:30split_1:output:31split_1:output:31split_1:output:32split_1:output:32split_1:output:33split_1:output:33split_1:output:34split_1:output:34split_1:output:35split_1:output:35split_1:output:36split_1:output:36split_1:output:37split_1:output:37split_1:output:38split_1:output:38split_1:output:39split_1:output:39split_1:output:40split_1:output:40split_1:output:41split_1:output:41split_1:output:42split_1:output:42split_1:output:43split_1:output:43split_1:output:44split_1:output:44split_1:output:45split_1:output:45split_1:output:46split_1:output:46split_1:output:47split_1:output:47split_1:output:48split_1:output:48split_1:output:49split_1:output:49split_1:output:50split_1:output:50split_1:output:51split_1:output:51split_1:output:52split_1:output:52split_1:output:53split_1:output:53split_1:output:54split_1:output:54split_1:output:55split_1:output:55split_1:output:56split_1:output:56split_1:output:57split_1:output:57split_1:output:58split_1:output:58split_1:output:59split_1:output:59split_1:output:60split_1:output:60split_1:output:61split_1:output:61split_1:output:62split_1:output:62split_1:output:63split_1:output:63split_1:output:64split_1:output:64split_1:output:65split_1:output:65split_1:output:66split_1:output:66split_1:output:67split_1:output:67split_1:output:68split_1:output:68split_1:output:69split_1:output:69split_1:output:70split_1:output:70split_1:output:71split_1:output:71split_1:output:72split_1:output:72split_1:output:73split_1:output:73split_1:output:74split_1:output:74split_1:output:75split_1:output:75split_1:output:76split_1:output:76split_1:output:77split_1:output:77split_1:output:78split_1:output:78split_1:output:79split_1:output:79split_1:output:80split_1:output:80split_1:output:81split_1:output:81split_1:output:82split_1:output:82split_1:output:83split_1:output:83split_1:output:84split_1:output:84split_1:output:85split_1:output:85split_1:output:86split_1:output:86split_1:output:87split_1:output:87split_1:output:88split_1:output:88split_1:output:89split_1:output:89split_1:output:90split_1:output:90split_1:output:91split_1:output:91split_1:output:92split_1:output:92split_1:output:93split_1:output:93split_1:output:94split_1:output:94split_1:output:95split_1:output:95concat_1/axis:output:0*
N?*
T0*5
_output_shapes#
!:???????????`@2

concat_1T
Const_2Const*
_output_shapes
: *
dtype0*
value	B :`2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*?
_output_shapes?
?:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@*
	num_split`2	
split_2`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis?
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15split_2:output:16split_2:output:16split_2:output:17split_2:output:17split_2:output:18split_2:output:18split_2:output:19split_2:output:19split_2:output:20split_2:output:20split_2:output:21split_2:output:21split_2:output:22split_2:output:22split_2:output:23split_2:output:23split_2:output:24split_2:output:24split_2:output:25split_2:output:25split_2:output:26split_2:output:26split_2:output:27split_2:output:27split_2:output:28split_2:output:28split_2:output:29split_2:output:29split_2:output:30split_2:output:30split_2:output:31split_2:output:31split_2:output:32split_2:output:32split_2:output:33split_2:output:33split_2:output:34split_2:output:34split_2:output:35split_2:output:35split_2:output:36split_2:output:36split_2:output:37split_2:output:37split_2:output:38split_2:output:38split_2:output:39split_2:output:39split_2:output:40split_2:output:40split_2:output:41split_2:output:41split_2:output:42split_2:output:42split_2:output:43split_2:output:43split_2:output:44split_2:output:44split_2:output:45split_2:output:45split_2:output:46split_2:output:46split_2:output:47split_2:output:47split_2:output:48split_2:output:48split_2:output:49split_2:output:49split_2:output:50split_2:output:50split_2:output:51split_2:output:51split_2:output:52split_2:output:52split_2:output:53split_2:output:53split_2:output:54split_2:output:54split_2:output:55split_2:output:55split_2:output:56split_2:output:56split_2:output:57split_2:output:57split_2:output:58split_2:output:58split_2:output:59split_2:output:59split_2:output:60split_2:output:60split_2:output:61split_2:output:61split_2:output:62split_2:output:62split_2:output:63split_2:output:63split_2:output:64split_2:output:64split_2:output:65split_2:output:65split_2:output:66split_2:output:66split_2:output:67split_2:output:67split_2:output:68split_2:output:68split_2:output:69split_2:output:69split_2:output:70split_2:output:70split_2:output:71split_2:output:71split_2:output:72split_2:output:72split_2:output:73split_2:output:73split_2:output:74split_2:output:74split_2:output:75split_2:output:75split_2:output:76split_2:output:76split_2:output:77split_2:output:77split_2:output:78split_2:output:78split_2:output:79split_2:output:79split_2:output:80split_2:output:80split_2:output:81split_2:output:81split_2:output:82split_2:output:82split_2:output:83split_2:output:83split_2:output:84split_2:output:84split_2:output:85split_2:output:85split_2:output:86split_2:output:86split_2:output:87split_2:output:87split_2:output:88split_2:output:88split_2:output:89split_2:output:89split_2:output:90split_2:output:90split_2:output:91split_2:output:91split_2:output:92split_2:output:92split_2:output:93split_2:output:93split_2:output:94split_2:output:94split_2:output:95split_2:output:95concat_2/axis:output:0*
N?*
T0*6
_output_shapes$
": ????????????@2

concat_2t
IdentityIdentityconcat_2:output:0*
T0*6
_output_shapes$
": ????????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????```@:[ W
3
_output_shapes!
:?????????```@
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_272843

inputs
identity`
	LeakyRelu	LeakyReluinputs*3
_output_shapes!
:?????????@2
	LeakyReluw
IdentityIdentityLeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
X
,__inference_concatenate_layer_call_fn_276483
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2726582
PartitionedCall{
IdentityIdentityPartitionedCall:output:0*
T0*6
_output_shapes$
": ????????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D: ????????????: ????????????:` \
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/0:`\
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/1
?
s
I__inference_concatenate_4_layer_call_and_return_conditional_losses_273520

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*6
_output_shapes$
": ????????????B2
concatr
IdentityIdentityconcat:output:0*
T0*6
_output_shapes$
": ????????????B2

Identity"
identityIdentity:output:0*W
_input_shapesF
D: ????????????@: ????????????:^ Z
6
_output_shapes$
": ????????????@
 
_user_specified_nameinputs:^Z
6
_output_shapes$
": ????????????
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_8_layer_call_fn_276927

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_2731932
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????```@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????```@:[ W
3
_output_shapes!
:?????????```@
 
_user_specified_nameinputs
?
z
%__inference_disp_layer_call_fn_272646

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8????????????????????????????????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_disp_layer_call_and_return_conditional_losses_2726362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8???????????????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
e
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_276724

inputs
identity`
	LeakyRelu	LeakyReluinputs*3
_output_shapes!
:?????????000@2
	LeakyReluw
IdentityIdentityLeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????000@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????000@:[ W
3
_output_shapes!
:?????????000@
 
_user_specified_nameinputs
?5
g
K__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_276701

inputs
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0inputs*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2
split\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15split:output:16split:output:16split:output:17split:output:17split:output:18split:output:18split:output:19split:output:19split:output:20split:output:20split:output:21split:output:21split:output:22split:output:22split:output:23split:output:23concat/axis:output:0*
N0*
T0*3
_output_shapes!
:?????????0@2
concatT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*?
_output_shapes?
?:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@*
	num_split2	
split_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15split_1:output:16split_1:output:16split_1:output:17split_1:output:17split_1:output:18split_1:output:18split_1:output:19split_1:output:19split_1:output:20split_1:output:20split_1:output:21split_1:output:21split_1:output:22split_1:output:22split_1:output:23split_1:output:23concat_1/axis:output:0*
N0*
T0*3
_output_shapes!
:?????????00@2

concat_1T
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*?
_output_shapes?
?:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@*
	num_split2	
split_2`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis?
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15split_2:output:16split_2:output:16split_2:output:17split_2:output:17split_2:output:18split_2:output:18split_2:output:19split_2:output:19split_2:output:20split_2:output:20split_2:output:21split_2:output:21split_2:output:22split_2:output:22split_2:output:23split_2:output:23concat_2/axis:output:0*
N0*
T0*3
_output_shapes!
:?????????000@2

concat_2q
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:?????????000@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
?
!transformer_map_while_cond_276054&
"transformer_map_while_loop_counter!
transformer_map_strided_slice
placeholder
placeholder_1&
"less_transformer_map_strided_slice>
:transformer_map_while_cond_276054___redundant_placeholder0>
:transformer_map_while_cond_276054___redundant_placeholder1
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
?

?
D__inference_conv3d_4_layer_call_and_return_conditional_losses_272489

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????@*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8????????????????????????????????????@:::v r
N
_output_shapes<
::8????????????????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
!transformer_map_while_cond_275045&
"transformer_map_while_loop_counter!
transformer_map_strided_slice
placeholder
placeholder_1&
"less_transformer_map_strided_slice>
:transformer_map_while_cond_275045___redundant_placeholder0>
:transformer_map_while_cond_275045___redundant_placeholder1
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
??
?
map_while_body_273595
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
?

?
D__inference_conv3d_9_layer_call_and_return_conditional_losses_272594

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:B *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8???????????????????????????????????? *
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8????????????????????????????????????B:::v r
N
_output_shapes<
::8????????????????????????????????????B
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
e
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_273539

inputs
identityc
	LeakyRelu	LeakyReluinputs*6
_output_shapes$
": ???????????? 2
	LeakyReluz
IdentityIdentityLeakyRelu:activations:0*
T0*6
_output_shapes$
": ???????????? 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ???????????? :^ Z
6
_output_shapes$
": ???????????? 
 
_user_specified_nameinputs
?
s
I__inference_concatenate_2_layer_call_and_return_conditional_losses_272954

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*4
_output_shapes"
 :?????????000?2
concatp
IdentityIdentityconcat:output:0*
T0*4
_output_shapes"
 :?????????000?2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????000@:?????????000@:[ W
3
_output_shapes!
:?????????000@
 
_user_specified_nameinputs:[W
3
_output_shapes!
:?????????000@
 
_user_specified_nameinputs
?
?
(__inference_model_1_layer_call_fn_274282
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*X
_output_shapesF
D: ????????????: ????????????*:
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2742292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*6
_output_shapes$
": ????????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*6
_output_shapes$
": ????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ????????????: ????????????::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
6
_output_shapes$
": ????????????
!
_user_specified_name	input_1:_[
6
_output_shapes$
": ????????????
!
_user_specified_name	input_2:
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
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
e
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_273193

inputs
identity`
	LeakyRelu	LeakyReluinputs*3
_output_shapes!
:?????????```@2
	LeakyReluw
IdentityIdentityLeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????```@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????```@:[ W
3
_output_shapes!
:?????????```@
 
_user_specified_nameinputs
?[
g
K__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_273141

inputs
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :02
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0inputs*
T0*?
_output_shapes?
?:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@*
	num_split02
split\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15split:output:16split:output:16split:output:17split:output:17split:output:18split:output:18split:output:19split:output:19split:output:20split:output:20split:output:21split:output:21split:output:22split:output:22split:output:23split:output:23split:output:24split:output:24split:output:25split:output:25split:output:26split:output:26split:output:27split:output:27split:output:28split:output:28split:output:29split:output:29split:output:30split:output:30split:output:31split:output:31split:output:32split:output:32split:output:33split:output:33split:output:34split:output:34split:output:35split:output:35split:output:36split:output:36split:output:37split:output:37split:output:38split:output:38split:output:39split:output:39split:output:40split:output:40split:output:41split:output:41split:output:42split:output:42split:output:43split:output:43split:output:44split:output:44split:output:45split:output:45split:output:46split:output:46split:output:47split:output:47concat/axis:output:0*
N`*
T0*3
_output_shapes!
:?????????`00@2
concatT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :02	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*?
_output_shapes?
?:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@*
	num_split02	
split_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15split_1:output:16split_1:output:16split_1:output:17split_1:output:17split_1:output:18split_1:output:18split_1:output:19split_1:output:19split_1:output:20split_1:output:20split_1:output:21split_1:output:21split_1:output:22split_1:output:22split_1:output:23split_1:output:23split_1:output:24split_1:output:24split_1:output:25split_1:output:25split_1:output:26split_1:output:26split_1:output:27split_1:output:27split_1:output:28split_1:output:28split_1:output:29split_1:output:29split_1:output:30split_1:output:30split_1:output:31split_1:output:31split_1:output:32split_1:output:32split_1:output:33split_1:output:33split_1:output:34split_1:output:34split_1:output:35split_1:output:35split_1:output:36split_1:output:36split_1:output:37split_1:output:37split_1:output:38split_1:output:38split_1:output:39split_1:output:39split_1:output:40split_1:output:40split_1:output:41split_1:output:41split_1:output:42split_1:output:42split_1:output:43split_1:output:43split_1:output:44split_1:output:44split_1:output:45split_1:output:45split_1:output:46split_1:output:46split_1:output:47split_1:output:47concat_1/axis:output:0*
N`*
T0*3
_output_shapes!
:?????????``0@2

concat_1T
Const_2Const*
_output_shapes
: *
dtype0*
value	B :02	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*?
_output_shapes?
?:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@*
	num_split02	
split_2`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis?
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15split_2:output:16split_2:output:16split_2:output:17split_2:output:17split_2:output:18split_2:output:18split_2:output:19split_2:output:19split_2:output:20split_2:output:20split_2:output:21split_2:output:21split_2:output:22split_2:output:22split_2:output:23split_2:output:23split_2:output:24split_2:output:24split_2:output:25split_2:output:25split_2:output:26split_2:output:26split_2:output:27split_2:output:27split_2:output:28split_2:output:28split_2:output:29split_2:output:29split_2:output:30split_2:output:30split_2:output:31split_2:output:31split_2:output:32split_2:output:32split_2:output:33split_2:output:33split_2:output:34split_2:output:34split_2:output:35split_2:output:35split_2:output:36split_2:output:36split_2:output:37split_2:output:37split_2:output:38split_2:output:38split_2:output:39split_2:output:39split_2:output:40split_2:output:40split_2:output:41split_2:output:41split_2:output:42split_2:output:42split_2:output:43split_2:output:43split_2:output:44split_2:output:44split_2:output:45split_2:output:45split_2:output:46split_2:output:46split_2:output:47split_2:output:47concat_2/axis:output:0*
N`*
T0*3
_output_shapes!
:?????????```@2

concat_2q
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:?????????```@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????000@:[ W
3
_output_shapes!
:?????????000@
 
_user_specified_nameinputs
?	
?
@__inference_disp_layer_call_and_return_conditional_losses_272636

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8???????????????????????????????????? :::v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
~
)__inference_conv3d_3_layer_call_fn_272478

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8????????????????????????????????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_2724682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8????????????????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
L
0__inference_up_sampling3d_2_layer_call_fn_276894

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_2731412
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????```@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????000@:[ W
3
_output_shapes!
:?????????000@
 
_user_specified_nameinputs
?
s
I__inference_concatenate_3_layer_call_and_return_conditional_losses_273156

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*3
_output_shapes!
:?????????````2
concato
IdentityIdentityconcat:output:0*
T0*3
_output_shapes!
:?????????````2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????```@:?????????``` :[ W
3
_output_shapes!
:?????????```@
 
_user_specified_nameinputs:[W
3
_output_shapes!
:?????????``` 
 
_user_specified_nameinputs
??
?	
C__inference_model_1_layer_call_and_return_conditional_losses_275349
inputs_0
inputs_1)
%conv3d_conv3d_readvariableop_resource*
&conv3d_biasadd_readvariableop_resource+
'conv3d_1_conv3d_readvariableop_resource,
(conv3d_1_biasadd_readvariableop_resource+
'conv3d_2_conv3d_readvariableop_resource,
(conv3d_2_biasadd_readvariableop_resource+
'conv3d_3_conv3d_readvariableop_resource,
(conv3d_3_biasadd_readvariableop_resource+
'conv3d_4_conv3d_readvariableop_resource,
(conv3d_4_biasadd_readvariableop_resource+
'conv3d_5_conv3d_readvariableop_resource,
(conv3d_5_biasadd_readvariableop_resource+
'conv3d_6_conv3d_readvariableop_resource,
(conv3d_6_biasadd_readvariableop_resource+
'conv3d_7_conv3d_readvariableop_resource,
(conv3d_7_biasadd_readvariableop_resource+
'conv3d_8_conv3d_readvariableop_resource,
(conv3d_8_biasadd_readvariableop_resource+
'conv3d_9_conv3d_readvariableop_resource,
(conv3d_9_biasadd_readvariableop_resource,
(conv3d_10_conv3d_readvariableop_resource-
)conv3d_10_biasadd_readvariableop_resource'
#disp_conv3d_readvariableop_resource(
$disp_biasadd_readvariableop_resource
identity

identity_1?t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2inputs_0inputs_1 concatenate/concat/axis:output:0*
N*
T0*6
_output_shapes$
": ????????????2
concatenate/concat?
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02
conv3d/Conv3D/ReadVariableOp?
conv3d/Conv3DConv3Dconcatenate/concat:output:0$conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????``` *
paddingSAME*
strides	
2
conv3d/Conv3D?
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv3d/BiasAdd/ReadVariableOp?
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????``` 2
conv3d/BiasAdd?
leaky_re_lu/LeakyRelu	LeakyReluconv3d/BiasAdd:output:0*3
_output_shapes!
:?????????``` 2
leaky_re_lu/LeakyRelu?
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype02 
conv3d_1/Conv3D/ReadVariableOp?
conv3d_1/Conv3DConv3D#leaky_re_lu/LeakyRelu:activations:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????000@*
paddingSAME*
strides	
2
conv3d_1/Conv3D?
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_1/BiasAdd/ReadVariableOp?
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????000@2
conv3d_1/BiasAdd?
leaky_re_lu_1/LeakyRelu	LeakyReluconv3d_1/BiasAdd:output:0*3
_output_shapes!
:?????????000@2
leaky_re_lu_1/LeakyRelu?
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype02 
conv3d_2/Conv3D/ReadVariableOp?
conv3d_2/Conv3DConv3D%leaky_re_lu_1/LeakyRelu:activations:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
2
conv3d_2/Conv3D?
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_2/BiasAdd/ReadVariableOp?
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@2
conv3d_2/BiasAdd?
leaky_re_lu_2/LeakyRelu	LeakyReluconv3d_2/BiasAdd:output:0*3
_output_shapes!
:?????????@2
leaky_re_lu_2/LeakyRelu?
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype02 
conv3d_3/Conv3D/ReadVariableOp?
conv3d_3/Conv3DConv3D%leaky_re_lu_2/LeakyRelu:activations:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
2
conv3d_3/Conv3D?
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_3/BiasAdd/ReadVariableOp?
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@2
conv3d_3/BiasAdd?
leaky_re_lu_3/LeakyRelu	LeakyReluconv3d_3/BiasAdd:output:0*3
_output_shapes!
:?????????@2
leaky_re_lu_3/LeakyRelu?
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype02 
conv3d_4/Conv3D/ReadVariableOp?
conv3d_4/Conv3DConv3D%leaky_re_lu_3/LeakyRelu:activations:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
2
conv3d_4/Conv3D?
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_4/BiasAdd/ReadVariableOp?
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@2
conv3d_4/BiasAdd?
leaky_re_lu_4/LeakyRelu	LeakyReluconv3d_4/BiasAdd:output:0*3
_output_shapes!
:?????????@2
leaky_re_lu_4/LeakyRelul
up_sampling3d/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/Const?
up_sampling3d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/split/split_dim?
up_sampling3d/splitSplit&up_sampling3d/split/split_dim:output:0%leaky_re_lu_4/LeakyRelu:activations:0*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2
up_sampling3d/splitx
up_sampling3d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/concat/axis?
up_sampling3d/concatConcatV2up_sampling3d/split:output:0up_sampling3d/split:output:0up_sampling3d/split:output:1up_sampling3d/split:output:1up_sampling3d/split:output:2up_sampling3d/split:output:2up_sampling3d/split:output:3up_sampling3d/split:output:3up_sampling3d/split:output:4up_sampling3d/split:output:4up_sampling3d/split:output:5up_sampling3d/split:output:5up_sampling3d/split:output:6up_sampling3d/split:output:6up_sampling3d/split:output:7up_sampling3d/split:output:7up_sampling3d/split:output:8up_sampling3d/split:output:8up_sampling3d/split:output:9up_sampling3d/split:output:9up_sampling3d/split:output:10up_sampling3d/split:output:10up_sampling3d/split:output:11up_sampling3d/split:output:11"up_sampling3d/concat/axis:output:0*
N*
T0*3
_output_shapes!
:?????????@2
up_sampling3d/concatp
up_sampling3d/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/Const_1?
up_sampling3d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
up_sampling3d/split_1/split_dim?
up_sampling3d/split_1Split(up_sampling3d/split_1/split_dim:output:0up_sampling3d/concat:output:0*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2
up_sampling3d/split_1|
up_sampling3d/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/concat_1/axis?
up_sampling3d/concat_1ConcatV2up_sampling3d/split_1:output:0up_sampling3d/split_1:output:0up_sampling3d/split_1:output:1up_sampling3d/split_1:output:1up_sampling3d/split_1:output:2up_sampling3d/split_1:output:2up_sampling3d/split_1:output:3up_sampling3d/split_1:output:3up_sampling3d/split_1:output:4up_sampling3d/split_1:output:4up_sampling3d/split_1:output:5up_sampling3d/split_1:output:5up_sampling3d/split_1:output:6up_sampling3d/split_1:output:6up_sampling3d/split_1:output:7up_sampling3d/split_1:output:7up_sampling3d/split_1:output:8up_sampling3d/split_1:output:8up_sampling3d/split_1:output:9up_sampling3d/split_1:output:9up_sampling3d/split_1:output:10up_sampling3d/split_1:output:10up_sampling3d/split_1:output:11up_sampling3d/split_1:output:11$up_sampling3d/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:?????????@2
up_sampling3d/concat_1p
up_sampling3d/Const_2Const*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/Const_2?
up_sampling3d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
up_sampling3d/split_2/split_dim?
up_sampling3d/split_2Split(up_sampling3d/split_2/split_dim:output:0up_sampling3d/concat_1:output:0*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2
up_sampling3d/split_2|
up_sampling3d/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d/concat_2/axis?
up_sampling3d/concat_2ConcatV2up_sampling3d/split_2:output:0up_sampling3d/split_2:output:0up_sampling3d/split_2:output:1up_sampling3d/split_2:output:1up_sampling3d/split_2:output:2up_sampling3d/split_2:output:2up_sampling3d/split_2:output:3up_sampling3d/split_2:output:3up_sampling3d/split_2:output:4up_sampling3d/split_2:output:4up_sampling3d/split_2:output:5up_sampling3d/split_2:output:5up_sampling3d/split_2:output:6up_sampling3d/split_2:output:6up_sampling3d/split_2:output:7up_sampling3d/split_2:output:7up_sampling3d/split_2:output:8up_sampling3d/split_2:output:8up_sampling3d/split_2:output:9up_sampling3d/split_2:output:9up_sampling3d/split_2:output:10up_sampling3d/split_2:output:10up_sampling3d/split_2:output:11up_sampling3d/split_2:output:11$up_sampling3d/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:?????????@2
up_sampling3d/concat_2x
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2up_sampling3d/concat_2:output:0%leaky_re_lu_2/LeakyRelu:activations:0"concatenate_1/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????2
concatenate_1/concat?
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource*+
_output_shapes
:?@*
dtype02 
conv3d_5/Conv3D/ReadVariableOp?
conv3d_5/Conv3DConv3Dconcatenate_1/concat:output:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
2
conv3d_5/Conv3D?
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_5/BiasAdd/ReadVariableOp?
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@2
conv3d_5/BiasAdd?
leaky_re_lu_5/LeakyRelu	LeakyReluconv3d_5/BiasAdd:output:0*3
_output_shapes!
:?????????@2
leaky_re_lu_5/LeakyRelup
up_sampling3d_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_1/Const?
up_sampling3d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
up_sampling3d_1/split/split_dim?
up_sampling3d_1/splitSplit(up_sampling3d_1/split/split_dim:output:0%leaky_re_lu_5/LeakyRelu:activations:0*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2
up_sampling3d_1/split|
up_sampling3d_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_1/concat/axis?
up_sampling3d_1/concatConcatV2up_sampling3d_1/split:output:0up_sampling3d_1/split:output:0up_sampling3d_1/split:output:1up_sampling3d_1/split:output:1up_sampling3d_1/split:output:2up_sampling3d_1/split:output:2up_sampling3d_1/split:output:3up_sampling3d_1/split:output:3up_sampling3d_1/split:output:4up_sampling3d_1/split:output:4up_sampling3d_1/split:output:5up_sampling3d_1/split:output:5up_sampling3d_1/split:output:6up_sampling3d_1/split:output:6up_sampling3d_1/split:output:7up_sampling3d_1/split:output:7up_sampling3d_1/split:output:8up_sampling3d_1/split:output:8up_sampling3d_1/split:output:9up_sampling3d_1/split:output:9up_sampling3d_1/split:output:10up_sampling3d_1/split:output:10up_sampling3d_1/split:output:11up_sampling3d_1/split:output:11up_sampling3d_1/split:output:12up_sampling3d_1/split:output:12up_sampling3d_1/split:output:13up_sampling3d_1/split:output:13up_sampling3d_1/split:output:14up_sampling3d_1/split:output:14up_sampling3d_1/split:output:15up_sampling3d_1/split:output:15up_sampling3d_1/split:output:16up_sampling3d_1/split:output:16up_sampling3d_1/split:output:17up_sampling3d_1/split:output:17up_sampling3d_1/split:output:18up_sampling3d_1/split:output:18up_sampling3d_1/split:output:19up_sampling3d_1/split:output:19up_sampling3d_1/split:output:20up_sampling3d_1/split:output:20up_sampling3d_1/split:output:21up_sampling3d_1/split:output:21up_sampling3d_1/split:output:22up_sampling3d_1/split:output:22up_sampling3d_1/split:output:23up_sampling3d_1/split:output:23$up_sampling3d_1/concat/axis:output:0*
N0*
T0*3
_output_shapes!
:?????????0@2
up_sampling3d_1/concatt
up_sampling3d_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_1/Const_1?
!up_sampling3d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!up_sampling3d_1/split_1/split_dim?
up_sampling3d_1/split_1Split*up_sampling3d_1/split_1/split_dim:output:0up_sampling3d_1/concat:output:0*
T0*?
_output_shapes?
?:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@*
	num_split2
up_sampling3d_1/split_1?
up_sampling3d_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_1/concat_1/axis?
up_sampling3d_1/concat_1ConcatV2 up_sampling3d_1/split_1:output:0 up_sampling3d_1/split_1:output:0 up_sampling3d_1/split_1:output:1 up_sampling3d_1/split_1:output:1 up_sampling3d_1/split_1:output:2 up_sampling3d_1/split_1:output:2 up_sampling3d_1/split_1:output:3 up_sampling3d_1/split_1:output:3 up_sampling3d_1/split_1:output:4 up_sampling3d_1/split_1:output:4 up_sampling3d_1/split_1:output:5 up_sampling3d_1/split_1:output:5 up_sampling3d_1/split_1:output:6 up_sampling3d_1/split_1:output:6 up_sampling3d_1/split_1:output:7 up_sampling3d_1/split_1:output:7 up_sampling3d_1/split_1:output:8 up_sampling3d_1/split_1:output:8 up_sampling3d_1/split_1:output:9 up_sampling3d_1/split_1:output:9!up_sampling3d_1/split_1:output:10!up_sampling3d_1/split_1:output:10!up_sampling3d_1/split_1:output:11!up_sampling3d_1/split_1:output:11!up_sampling3d_1/split_1:output:12!up_sampling3d_1/split_1:output:12!up_sampling3d_1/split_1:output:13!up_sampling3d_1/split_1:output:13!up_sampling3d_1/split_1:output:14!up_sampling3d_1/split_1:output:14!up_sampling3d_1/split_1:output:15!up_sampling3d_1/split_1:output:15!up_sampling3d_1/split_1:output:16!up_sampling3d_1/split_1:output:16!up_sampling3d_1/split_1:output:17!up_sampling3d_1/split_1:output:17!up_sampling3d_1/split_1:output:18!up_sampling3d_1/split_1:output:18!up_sampling3d_1/split_1:output:19!up_sampling3d_1/split_1:output:19!up_sampling3d_1/split_1:output:20!up_sampling3d_1/split_1:output:20!up_sampling3d_1/split_1:output:21!up_sampling3d_1/split_1:output:21!up_sampling3d_1/split_1:output:22!up_sampling3d_1/split_1:output:22!up_sampling3d_1/split_1:output:23!up_sampling3d_1/split_1:output:23&up_sampling3d_1/concat_1/axis:output:0*
N0*
T0*3
_output_shapes!
:?????????00@2
up_sampling3d_1/concat_1t
up_sampling3d_1/Const_2Const*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_1/Const_2?
!up_sampling3d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!up_sampling3d_1/split_2/split_dim?
up_sampling3d_1/split_2Split*up_sampling3d_1/split_2/split_dim:output:0!up_sampling3d_1/concat_1:output:0*
T0*?
_output_shapes?
?:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@*
	num_split2
up_sampling3d_1/split_2?
up_sampling3d_1/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_1/concat_2/axis?
up_sampling3d_1/concat_2ConcatV2 up_sampling3d_1/split_2:output:0 up_sampling3d_1/split_2:output:0 up_sampling3d_1/split_2:output:1 up_sampling3d_1/split_2:output:1 up_sampling3d_1/split_2:output:2 up_sampling3d_1/split_2:output:2 up_sampling3d_1/split_2:output:3 up_sampling3d_1/split_2:output:3 up_sampling3d_1/split_2:output:4 up_sampling3d_1/split_2:output:4 up_sampling3d_1/split_2:output:5 up_sampling3d_1/split_2:output:5 up_sampling3d_1/split_2:output:6 up_sampling3d_1/split_2:output:6 up_sampling3d_1/split_2:output:7 up_sampling3d_1/split_2:output:7 up_sampling3d_1/split_2:output:8 up_sampling3d_1/split_2:output:8 up_sampling3d_1/split_2:output:9 up_sampling3d_1/split_2:output:9!up_sampling3d_1/split_2:output:10!up_sampling3d_1/split_2:output:10!up_sampling3d_1/split_2:output:11!up_sampling3d_1/split_2:output:11!up_sampling3d_1/split_2:output:12!up_sampling3d_1/split_2:output:12!up_sampling3d_1/split_2:output:13!up_sampling3d_1/split_2:output:13!up_sampling3d_1/split_2:output:14!up_sampling3d_1/split_2:output:14!up_sampling3d_1/split_2:output:15!up_sampling3d_1/split_2:output:15!up_sampling3d_1/split_2:output:16!up_sampling3d_1/split_2:output:16!up_sampling3d_1/split_2:output:17!up_sampling3d_1/split_2:output:17!up_sampling3d_1/split_2:output:18!up_sampling3d_1/split_2:output:18!up_sampling3d_1/split_2:output:19!up_sampling3d_1/split_2:output:19!up_sampling3d_1/split_2:output:20!up_sampling3d_1/split_2:output:20!up_sampling3d_1/split_2:output:21!up_sampling3d_1/split_2:output:21!up_sampling3d_1/split_2:output:22!up_sampling3d_1/split_2:output:22!up_sampling3d_1/split_2:output:23!up_sampling3d_1/split_2:output:23&up_sampling3d_1/concat_2/axis:output:0*
N0*
T0*3
_output_shapes!
:?????????000@2
up_sampling3d_1/concat_2x
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axis?
concatenate_2/concatConcatV2!up_sampling3d_1/concat_2:output:0%leaky_re_lu_1/LeakyRelu:activations:0"concatenate_2/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :?????????000?2
concatenate_2/concat?
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource*+
_output_shapes
:?@*
dtype02 
conv3d_6/Conv3D/ReadVariableOp?
conv3d_6/Conv3DConv3Dconcatenate_2/concat:output:0&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????000@*
paddingSAME*
strides	
2
conv3d_6/Conv3D?
conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_6/BiasAdd/ReadVariableOp?
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????000@2
conv3d_6/BiasAdd?
leaky_re_lu_6/LeakyRelu	LeakyReluconv3d_6/BiasAdd:output:0*3
_output_shapes!
:?????????000@2
leaky_re_lu_6/LeakyRelup
up_sampling3d_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :02
up_sampling3d_2/Const?
up_sampling3d_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
up_sampling3d_2/split/split_dim?
up_sampling3d_2/splitSplit(up_sampling3d_2/split/split_dim:output:0%leaky_re_lu_6/LeakyRelu:activations:0*
T0*?
_output_shapes?
?:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@*
	num_split02
up_sampling3d_2/split|
up_sampling3d_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_2/concat/axis?
up_sampling3d_2/concatConcatV2up_sampling3d_2/split:output:0up_sampling3d_2/split:output:0up_sampling3d_2/split:output:1up_sampling3d_2/split:output:1up_sampling3d_2/split:output:2up_sampling3d_2/split:output:2up_sampling3d_2/split:output:3up_sampling3d_2/split:output:3up_sampling3d_2/split:output:4up_sampling3d_2/split:output:4up_sampling3d_2/split:output:5up_sampling3d_2/split:output:5up_sampling3d_2/split:output:6up_sampling3d_2/split:output:6up_sampling3d_2/split:output:7up_sampling3d_2/split:output:7up_sampling3d_2/split:output:8up_sampling3d_2/split:output:8up_sampling3d_2/split:output:9up_sampling3d_2/split:output:9up_sampling3d_2/split:output:10up_sampling3d_2/split:output:10up_sampling3d_2/split:output:11up_sampling3d_2/split:output:11up_sampling3d_2/split:output:12up_sampling3d_2/split:output:12up_sampling3d_2/split:output:13up_sampling3d_2/split:output:13up_sampling3d_2/split:output:14up_sampling3d_2/split:output:14up_sampling3d_2/split:output:15up_sampling3d_2/split:output:15up_sampling3d_2/split:output:16up_sampling3d_2/split:output:16up_sampling3d_2/split:output:17up_sampling3d_2/split:output:17up_sampling3d_2/split:output:18up_sampling3d_2/split:output:18up_sampling3d_2/split:output:19up_sampling3d_2/split:output:19up_sampling3d_2/split:output:20up_sampling3d_2/split:output:20up_sampling3d_2/split:output:21up_sampling3d_2/split:output:21up_sampling3d_2/split:output:22up_sampling3d_2/split:output:22up_sampling3d_2/split:output:23up_sampling3d_2/split:output:23up_sampling3d_2/split:output:24up_sampling3d_2/split:output:24up_sampling3d_2/split:output:25up_sampling3d_2/split:output:25up_sampling3d_2/split:output:26up_sampling3d_2/split:output:26up_sampling3d_2/split:output:27up_sampling3d_2/split:output:27up_sampling3d_2/split:output:28up_sampling3d_2/split:output:28up_sampling3d_2/split:output:29up_sampling3d_2/split:output:29up_sampling3d_2/split:output:30up_sampling3d_2/split:output:30up_sampling3d_2/split:output:31up_sampling3d_2/split:output:31up_sampling3d_2/split:output:32up_sampling3d_2/split:output:32up_sampling3d_2/split:output:33up_sampling3d_2/split:output:33up_sampling3d_2/split:output:34up_sampling3d_2/split:output:34up_sampling3d_2/split:output:35up_sampling3d_2/split:output:35up_sampling3d_2/split:output:36up_sampling3d_2/split:output:36up_sampling3d_2/split:output:37up_sampling3d_2/split:output:37up_sampling3d_2/split:output:38up_sampling3d_2/split:output:38up_sampling3d_2/split:output:39up_sampling3d_2/split:output:39up_sampling3d_2/split:output:40up_sampling3d_2/split:output:40up_sampling3d_2/split:output:41up_sampling3d_2/split:output:41up_sampling3d_2/split:output:42up_sampling3d_2/split:output:42up_sampling3d_2/split:output:43up_sampling3d_2/split:output:43up_sampling3d_2/split:output:44up_sampling3d_2/split:output:44up_sampling3d_2/split:output:45up_sampling3d_2/split:output:45up_sampling3d_2/split:output:46up_sampling3d_2/split:output:46up_sampling3d_2/split:output:47up_sampling3d_2/split:output:47$up_sampling3d_2/concat/axis:output:0*
N`*
T0*3
_output_shapes!
:?????????`00@2
up_sampling3d_2/concatt
up_sampling3d_2/Const_1Const*
_output_shapes
: *
dtype0*
value	B :02
up_sampling3d_2/Const_1?
!up_sampling3d_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!up_sampling3d_2/split_1/split_dim?
up_sampling3d_2/split_1Split*up_sampling3d_2/split_1/split_dim:output:0up_sampling3d_2/concat:output:0*
T0*?
_output_shapes?
?:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@*
	num_split02
up_sampling3d_2/split_1?
up_sampling3d_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_2/concat_1/axis?
up_sampling3d_2/concat_1ConcatV2 up_sampling3d_2/split_1:output:0 up_sampling3d_2/split_1:output:0 up_sampling3d_2/split_1:output:1 up_sampling3d_2/split_1:output:1 up_sampling3d_2/split_1:output:2 up_sampling3d_2/split_1:output:2 up_sampling3d_2/split_1:output:3 up_sampling3d_2/split_1:output:3 up_sampling3d_2/split_1:output:4 up_sampling3d_2/split_1:output:4 up_sampling3d_2/split_1:output:5 up_sampling3d_2/split_1:output:5 up_sampling3d_2/split_1:output:6 up_sampling3d_2/split_1:output:6 up_sampling3d_2/split_1:output:7 up_sampling3d_2/split_1:output:7 up_sampling3d_2/split_1:output:8 up_sampling3d_2/split_1:output:8 up_sampling3d_2/split_1:output:9 up_sampling3d_2/split_1:output:9!up_sampling3d_2/split_1:output:10!up_sampling3d_2/split_1:output:10!up_sampling3d_2/split_1:output:11!up_sampling3d_2/split_1:output:11!up_sampling3d_2/split_1:output:12!up_sampling3d_2/split_1:output:12!up_sampling3d_2/split_1:output:13!up_sampling3d_2/split_1:output:13!up_sampling3d_2/split_1:output:14!up_sampling3d_2/split_1:output:14!up_sampling3d_2/split_1:output:15!up_sampling3d_2/split_1:output:15!up_sampling3d_2/split_1:output:16!up_sampling3d_2/split_1:output:16!up_sampling3d_2/split_1:output:17!up_sampling3d_2/split_1:output:17!up_sampling3d_2/split_1:output:18!up_sampling3d_2/split_1:output:18!up_sampling3d_2/split_1:output:19!up_sampling3d_2/split_1:output:19!up_sampling3d_2/split_1:output:20!up_sampling3d_2/split_1:output:20!up_sampling3d_2/split_1:output:21!up_sampling3d_2/split_1:output:21!up_sampling3d_2/split_1:output:22!up_sampling3d_2/split_1:output:22!up_sampling3d_2/split_1:output:23!up_sampling3d_2/split_1:output:23!up_sampling3d_2/split_1:output:24!up_sampling3d_2/split_1:output:24!up_sampling3d_2/split_1:output:25!up_sampling3d_2/split_1:output:25!up_sampling3d_2/split_1:output:26!up_sampling3d_2/split_1:output:26!up_sampling3d_2/split_1:output:27!up_sampling3d_2/split_1:output:27!up_sampling3d_2/split_1:output:28!up_sampling3d_2/split_1:output:28!up_sampling3d_2/split_1:output:29!up_sampling3d_2/split_1:output:29!up_sampling3d_2/split_1:output:30!up_sampling3d_2/split_1:output:30!up_sampling3d_2/split_1:output:31!up_sampling3d_2/split_1:output:31!up_sampling3d_2/split_1:output:32!up_sampling3d_2/split_1:output:32!up_sampling3d_2/split_1:output:33!up_sampling3d_2/split_1:output:33!up_sampling3d_2/split_1:output:34!up_sampling3d_2/split_1:output:34!up_sampling3d_2/split_1:output:35!up_sampling3d_2/split_1:output:35!up_sampling3d_2/split_1:output:36!up_sampling3d_2/split_1:output:36!up_sampling3d_2/split_1:output:37!up_sampling3d_2/split_1:output:37!up_sampling3d_2/split_1:output:38!up_sampling3d_2/split_1:output:38!up_sampling3d_2/split_1:output:39!up_sampling3d_2/split_1:output:39!up_sampling3d_2/split_1:output:40!up_sampling3d_2/split_1:output:40!up_sampling3d_2/split_1:output:41!up_sampling3d_2/split_1:output:41!up_sampling3d_2/split_1:output:42!up_sampling3d_2/split_1:output:42!up_sampling3d_2/split_1:output:43!up_sampling3d_2/split_1:output:43!up_sampling3d_2/split_1:output:44!up_sampling3d_2/split_1:output:44!up_sampling3d_2/split_1:output:45!up_sampling3d_2/split_1:output:45!up_sampling3d_2/split_1:output:46!up_sampling3d_2/split_1:output:46!up_sampling3d_2/split_1:output:47!up_sampling3d_2/split_1:output:47&up_sampling3d_2/concat_1/axis:output:0*
N`*
T0*3
_output_shapes!
:?????????``0@2
up_sampling3d_2/concat_1t
up_sampling3d_2/Const_2Const*
_output_shapes
: *
dtype0*
value	B :02
up_sampling3d_2/Const_2?
!up_sampling3d_2/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!up_sampling3d_2/split_2/split_dim?
up_sampling3d_2/split_2Split*up_sampling3d_2/split_2/split_dim:output:0!up_sampling3d_2/concat_1:output:0*
T0*?
_output_shapes?
?:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@*
	num_split02
up_sampling3d_2/split_2?
up_sampling3d_2/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_2/concat_2/axis?
up_sampling3d_2/concat_2ConcatV2 up_sampling3d_2/split_2:output:0 up_sampling3d_2/split_2:output:0 up_sampling3d_2/split_2:output:1 up_sampling3d_2/split_2:output:1 up_sampling3d_2/split_2:output:2 up_sampling3d_2/split_2:output:2 up_sampling3d_2/split_2:output:3 up_sampling3d_2/split_2:output:3 up_sampling3d_2/split_2:output:4 up_sampling3d_2/split_2:output:4 up_sampling3d_2/split_2:output:5 up_sampling3d_2/split_2:output:5 up_sampling3d_2/split_2:output:6 up_sampling3d_2/split_2:output:6 up_sampling3d_2/split_2:output:7 up_sampling3d_2/split_2:output:7 up_sampling3d_2/split_2:output:8 up_sampling3d_2/split_2:output:8 up_sampling3d_2/split_2:output:9 up_sampling3d_2/split_2:output:9!up_sampling3d_2/split_2:output:10!up_sampling3d_2/split_2:output:10!up_sampling3d_2/split_2:output:11!up_sampling3d_2/split_2:output:11!up_sampling3d_2/split_2:output:12!up_sampling3d_2/split_2:output:12!up_sampling3d_2/split_2:output:13!up_sampling3d_2/split_2:output:13!up_sampling3d_2/split_2:output:14!up_sampling3d_2/split_2:output:14!up_sampling3d_2/split_2:output:15!up_sampling3d_2/split_2:output:15!up_sampling3d_2/split_2:output:16!up_sampling3d_2/split_2:output:16!up_sampling3d_2/split_2:output:17!up_sampling3d_2/split_2:output:17!up_sampling3d_2/split_2:output:18!up_sampling3d_2/split_2:output:18!up_sampling3d_2/split_2:output:19!up_sampling3d_2/split_2:output:19!up_sampling3d_2/split_2:output:20!up_sampling3d_2/split_2:output:20!up_sampling3d_2/split_2:output:21!up_sampling3d_2/split_2:output:21!up_sampling3d_2/split_2:output:22!up_sampling3d_2/split_2:output:22!up_sampling3d_2/split_2:output:23!up_sampling3d_2/split_2:output:23!up_sampling3d_2/split_2:output:24!up_sampling3d_2/split_2:output:24!up_sampling3d_2/split_2:output:25!up_sampling3d_2/split_2:output:25!up_sampling3d_2/split_2:output:26!up_sampling3d_2/split_2:output:26!up_sampling3d_2/split_2:output:27!up_sampling3d_2/split_2:output:27!up_sampling3d_2/split_2:output:28!up_sampling3d_2/split_2:output:28!up_sampling3d_2/split_2:output:29!up_sampling3d_2/split_2:output:29!up_sampling3d_2/split_2:output:30!up_sampling3d_2/split_2:output:30!up_sampling3d_2/split_2:output:31!up_sampling3d_2/split_2:output:31!up_sampling3d_2/split_2:output:32!up_sampling3d_2/split_2:output:32!up_sampling3d_2/split_2:output:33!up_sampling3d_2/split_2:output:33!up_sampling3d_2/split_2:output:34!up_sampling3d_2/split_2:output:34!up_sampling3d_2/split_2:output:35!up_sampling3d_2/split_2:output:35!up_sampling3d_2/split_2:output:36!up_sampling3d_2/split_2:output:36!up_sampling3d_2/split_2:output:37!up_sampling3d_2/split_2:output:37!up_sampling3d_2/split_2:output:38!up_sampling3d_2/split_2:output:38!up_sampling3d_2/split_2:output:39!up_sampling3d_2/split_2:output:39!up_sampling3d_2/split_2:output:40!up_sampling3d_2/split_2:output:40!up_sampling3d_2/split_2:output:41!up_sampling3d_2/split_2:output:41!up_sampling3d_2/split_2:output:42!up_sampling3d_2/split_2:output:42!up_sampling3d_2/split_2:output:43!up_sampling3d_2/split_2:output:43!up_sampling3d_2/split_2:output:44!up_sampling3d_2/split_2:output:44!up_sampling3d_2/split_2:output:45!up_sampling3d_2/split_2:output:45!up_sampling3d_2/split_2:output:46!up_sampling3d_2/split_2:output:46!up_sampling3d_2/split_2:output:47!up_sampling3d_2/split_2:output:47&up_sampling3d_2/concat_2/axis:output:0*
N`*
T0*3
_output_shapes!
:?????????```@2
up_sampling3d_2/concat_2x
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axis?
concatenate_3/concatConcatV2!up_sampling3d_2/concat_2:output:0#leaky_re_lu/LeakyRelu:activations:0"concatenate_3/concat/axis:output:0*
N*
T0*3
_output_shapes!
:?????????````2
concatenate_3/concat?
conv3d_7/Conv3D/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource**
_output_shapes
:`@*
dtype02 
conv3d_7/Conv3D/ReadVariableOp?
conv3d_7/Conv3DConv3Dconcatenate_3/concat:output:0&conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????```@*
paddingSAME*
strides	
2
conv3d_7/Conv3D?
conv3d_7/BiasAdd/ReadVariableOpReadVariableOp(conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_7/BiasAdd/ReadVariableOp?
conv3d_7/BiasAddBiasAddconv3d_7/Conv3D:output:0'conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????```@2
conv3d_7/BiasAdd?
leaky_re_lu_7/LeakyRelu	LeakyReluconv3d_7/BiasAdd:output:0*3
_output_shapes!
:?????????```@2
leaky_re_lu_7/LeakyRelu?
conv3d_8/Conv3D/ReadVariableOpReadVariableOp'conv3d_8_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype02 
conv3d_8/Conv3D/ReadVariableOp?
conv3d_8/Conv3DConv3D%leaky_re_lu_7/LeakyRelu:activations:0&conv3d_8/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????```@*
paddingSAME*
strides	
2
conv3d_8/Conv3D?
conv3d_8/BiasAdd/ReadVariableOpReadVariableOp(conv3d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_8/BiasAdd/ReadVariableOp?
conv3d_8/BiasAddBiasAddconv3d_8/Conv3D:output:0'conv3d_8/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????```@2
conv3d_8/BiasAdd?
leaky_re_lu_8/LeakyRelu	LeakyReluconv3d_8/BiasAdd:output:0*3
_output_shapes!
:?????????```@2
leaky_re_lu_8/LeakyRelup
up_sampling3d_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :`2
up_sampling3d_3/Const?
up_sampling3d_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
up_sampling3d_3/split/split_dim?
up_sampling3d_3/splitSplit(up_sampling3d_3/split/split_dim:output:0%leaky_re_lu_8/LeakyRelu:activations:0*
T0*?
_output_shapes?
?:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@*
	num_split`2
up_sampling3d_3/split|
up_sampling3d_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_3/concat/axis?2
up_sampling3d_3/concatConcatV2up_sampling3d_3/split:output:0up_sampling3d_3/split:output:0up_sampling3d_3/split:output:1up_sampling3d_3/split:output:1up_sampling3d_3/split:output:2up_sampling3d_3/split:output:2up_sampling3d_3/split:output:3up_sampling3d_3/split:output:3up_sampling3d_3/split:output:4up_sampling3d_3/split:output:4up_sampling3d_3/split:output:5up_sampling3d_3/split:output:5up_sampling3d_3/split:output:6up_sampling3d_3/split:output:6up_sampling3d_3/split:output:7up_sampling3d_3/split:output:7up_sampling3d_3/split:output:8up_sampling3d_3/split:output:8up_sampling3d_3/split:output:9up_sampling3d_3/split:output:9up_sampling3d_3/split:output:10up_sampling3d_3/split:output:10up_sampling3d_3/split:output:11up_sampling3d_3/split:output:11up_sampling3d_3/split:output:12up_sampling3d_3/split:output:12up_sampling3d_3/split:output:13up_sampling3d_3/split:output:13up_sampling3d_3/split:output:14up_sampling3d_3/split:output:14up_sampling3d_3/split:output:15up_sampling3d_3/split:output:15up_sampling3d_3/split:output:16up_sampling3d_3/split:output:16up_sampling3d_3/split:output:17up_sampling3d_3/split:output:17up_sampling3d_3/split:output:18up_sampling3d_3/split:output:18up_sampling3d_3/split:output:19up_sampling3d_3/split:output:19up_sampling3d_3/split:output:20up_sampling3d_3/split:output:20up_sampling3d_3/split:output:21up_sampling3d_3/split:output:21up_sampling3d_3/split:output:22up_sampling3d_3/split:output:22up_sampling3d_3/split:output:23up_sampling3d_3/split:output:23up_sampling3d_3/split:output:24up_sampling3d_3/split:output:24up_sampling3d_3/split:output:25up_sampling3d_3/split:output:25up_sampling3d_3/split:output:26up_sampling3d_3/split:output:26up_sampling3d_3/split:output:27up_sampling3d_3/split:output:27up_sampling3d_3/split:output:28up_sampling3d_3/split:output:28up_sampling3d_3/split:output:29up_sampling3d_3/split:output:29up_sampling3d_3/split:output:30up_sampling3d_3/split:output:30up_sampling3d_3/split:output:31up_sampling3d_3/split:output:31up_sampling3d_3/split:output:32up_sampling3d_3/split:output:32up_sampling3d_3/split:output:33up_sampling3d_3/split:output:33up_sampling3d_3/split:output:34up_sampling3d_3/split:output:34up_sampling3d_3/split:output:35up_sampling3d_3/split:output:35up_sampling3d_3/split:output:36up_sampling3d_3/split:output:36up_sampling3d_3/split:output:37up_sampling3d_3/split:output:37up_sampling3d_3/split:output:38up_sampling3d_3/split:output:38up_sampling3d_3/split:output:39up_sampling3d_3/split:output:39up_sampling3d_3/split:output:40up_sampling3d_3/split:output:40up_sampling3d_3/split:output:41up_sampling3d_3/split:output:41up_sampling3d_3/split:output:42up_sampling3d_3/split:output:42up_sampling3d_3/split:output:43up_sampling3d_3/split:output:43up_sampling3d_3/split:output:44up_sampling3d_3/split:output:44up_sampling3d_3/split:output:45up_sampling3d_3/split:output:45up_sampling3d_3/split:output:46up_sampling3d_3/split:output:46up_sampling3d_3/split:output:47up_sampling3d_3/split:output:47up_sampling3d_3/split:output:48up_sampling3d_3/split:output:48up_sampling3d_3/split:output:49up_sampling3d_3/split:output:49up_sampling3d_3/split:output:50up_sampling3d_3/split:output:50up_sampling3d_3/split:output:51up_sampling3d_3/split:output:51up_sampling3d_3/split:output:52up_sampling3d_3/split:output:52up_sampling3d_3/split:output:53up_sampling3d_3/split:output:53up_sampling3d_3/split:output:54up_sampling3d_3/split:output:54up_sampling3d_3/split:output:55up_sampling3d_3/split:output:55up_sampling3d_3/split:output:56up_sampling3d_3/split:output:56up_sampling3d_3/split:output:57up_sampling3d_3/split:output:57up_sampling3d_3/split:output:58up_sampling3d_3/split:output:58up_sampling3d_3/split:output:59up_sampling3d_3/split:output:59up_sampling3d_3/split:output:60up_sampling3d_3/split:output:60up_sampling3d_3/split:output:61up_sampling3d_3/split:output:61up_sampling3d_3/split:output:62up_sampling3d_3/split:output:62up_sampling3d_3/split:output:63up_sampling3d_3/split:output:63up_sampling3d_3/split:output:64up_sampling3d_3/split:output:64up_sampling3d_3/split:output:65up_sampling3d_3/split:output:65up_sampling3d_3/split:output:66up_sampling3d_3/split:output:66up_sampling3d_3/split:output:67up_sampling3d_3/split:output:67up_sampling3d_3/split:output:68up_sampling3d_3/split:output:68up_sampling3d_3/split:output:69up_sampling3d_3/split:output:69up_sampling3d_3/split:output:70up_sampling3d_3/split:output:70up_sampling3d_3/split:output:71up_sampling3d_3/split:output:71up_sampling3d_3/split:output:72up_sampling3d_3/split:output:72up_sampling3d_3/split:output:73up_sampling3d_3/split:output:73up_sampling3d_3/split:output:74up_sampling3d_3/split:output:74up_sampling3d_3/split:output:75up_sampling3d_3/split:output:75up_sampling3d_3/split:output:76up_sampling3d_3/split:output:76up_sampling3d_3/split:output:77up_sampling3d_3/split:output:77up_sampling3d_3/split:output:78up_sampling3d_3/split:output:78up_sampling3d_3/split:output:79up_sampling3d_3/split:output:79up_sampling3d_3/split:output:80up_sampling3d_3/split:output:80up_sampling3d_3/split:output:81up_sampling3d_3/split:output:81up_sampling3d_3/split:output:82up_sampling3d_3/split:output:82up_sampling3d_3/split:output:83up_sampling3d_3/split:output:83up_sampling3d_3/split:output:84up_sampling3d_3/split:output:84up_sampling3d_3/split:output:85up_sampling3d_3/split:output:85up_sampling3d_3/split:output:86up_sampling3d_3/split:output:86up_sampling3d_3/split:output:87up_sampling3d_3/split:output:87up_sampling3d_3/split:output:88up_sampling3d_3/split:output:88up_sampling3d_3/split:output:89up_sampling3d_3/split:output:89up_sampling3d_3/split:output:90up_sampling3d_3/split:output:90up_sampling3d_3/split:output:91up_sampling3d_3/split:output:91up_sampling3d_3/split:output:92up_sampling3d_3/split:output:92up_sampling3d_3/split:output:93up_sampling3d_3/split:output:93up_sampling3d_3/split:output:94up_sampling3d_3/split:output:94up_sampling3d_3/split:output:95up_sampling3d_3/split:output:95$up_sampling3d_3/concat/axis:output:0*
N?*
T0*4
_output_shapes"
 :??????????``@2
up_sampling3d_3/concatt
up_sampling3d_3/Const_1Const*
_output_shapes
: *
dtype0*
value	B :`2
up_sampling3d_3/Const_1?
!up_sampling3d_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!up_sampling3d_3/split_1/split_dim?
up_sampling3d_3/split_1Split*up_sampling3d_3/split_1/split_dim:output:0up_sampling3d_3/concat:output:0*
T0*?
_output_shapes?
?:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@*
	num_split`2
up_sampling3d_3/split_1?
up_sampling3d_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_3/concat_1/axis?5
up_sampling3d_3/concat_1ConcatV2 up_sampling3d_3/split_1:output:0 up_sampling3d_3/split_1:output:0 up_sampling3d_3/split_1:output:1 up_sampling3d_3/split_1:output:1 up_sampling3d_3/split_1:output:2 up_sampling3d_3/split_1:output:2 up_sampling3d_3/split_1:output:3 up_sampling3d_3/split_1:output:3 up_sampling3d_3/split_1:output:4 up_sampling3d_3/split_1:output:4 up_sampling3d_3/split_1:output:5 up_sampling3d_3/split_1:output:5 up_sampling3d_3/split_1:output:6 up_sampling3d_3/split_1:output:6 up_sampling3d_3/split_1:output:7 up_sampling3d_3/split_1:output:7 up_sampling3d_3/split_1:output:8 up_sampling3d_3/split_1:output:8 up_sampling3d_3/split_1:output:9 up_sampling3d_3/split_1:output:9!up_sampling3d_3/split_1:output:10!up_sampling3d_3/split_1:output:10!up_sampling3d_3/split_1:output:11!up_sampling3d_3/split_1:output:11!up_sampling3d_3/split_1:output:12!up_sampling3d_3/split_1:output:12!up_sampling3d_3/split_1:output:13!up_sampling3d_3/split_1:output:13!up_sampling3d_3/split_1:output:14!up_sampling3d_3/split_1:output:14!up_sampling3d_3/split_1:output:15!up_sampling3d_3/split_1:output:15!up_sampling3d_3/split_1:output:16!up_sampling3d_3/split_1:output:16!up_sampling3d_3/split_1:output:17!up_sampling3d_3/split_1:output:17!up_sampling3d_3/split_1:output:18!up_sampling3d_3/split_1:output:18!up_sampling3d_3/split_1:output:19!up_sampling3d_3/split_1:output:19!up_sampling3d_3/split_1:output:20!up_sampling3d_3/split_1:output:20!up_sampling3d_3/split_1:output:21!up_sampling3d_3/split_1:output:21!up_sampling3d_3/split_1:output:22!up_sampling3d_3/split_1:output:22!up_sampling3d_3/split_1:output:23!up_sampling3d_3/split_1:output:23!up_sampling3d_3/split_1:output:24!up_sampling3d_3/split_1:output:24!up_sampling3d_3/split_1:output:25!up_sampling3d_3/split_1:output:25!up_sampling3d_3/split_1:output:26!up_sampling3d_3/split_1:output:26!up_sampling3d_3/split_1:output:27!up_sampling3d_3/split_1:output:27!up_sampling3d_3/split_1:output:28!up_sampling3d_3/split_1:output:28!up_sampling3d_3/split_1:output:29!up_sampling3d_3/split_1:output:29!up_sampling3d_3/split_1:output:30!up_sampling3d_3/split_1:output:30!up_sampling3d_3/split_1:output:31!up_sampling3d_3/split_1:output:31!up_sampling3d_3/split_1:output:32!up_sampling3d_3/split_1:output:32!up_sampling3d_3/split_1:output:33!up_sampling3d_3/split_1:output:33!up_sampling3d_3/split_1:output:34!up_sampling3d_3/split_1:output:34!up_sampling3d_3/split_1:output:35!up_sampling3d_3/split_1:output:35!up_sampling3d_3/split_1:output:36!up_sampling3d_3/split_1:output:36!up_sampling3d_3/split_1:output:37!up_sampling3d_3/split_1:output:37!up_sampling3d_3/split_1:output:38!up_sampling3d_3/split_1:output:38!up_sampling3d_3/split_1:output:39!up_sampling3d_3/split_1:output:39!up_sampling3d_3/split_1:output:40!up_sampling3d_3/split_1:output:40!up_sampling3d_3/split_1:output:41!up_sampling3d_3/split_1:output:41!up_sampling3d_3/split_1:output:42!up_sampling3d_3/split_1:output:42!up_sampling3d_3/split_1:output:43!up_sampling3d_3/split_1:output:43!up_sampling3d_3/split_1:output:44!up_sampling3d_3/split_1:output:44!up_sampling3d_3/split_1:output:45!up_sampling3d_3/split_1:output:45!up_sampling3d_3/split_1:output:46!up_sampling3d_3/split_1:output:46!up_sampling3d_3/split_1:output:47!up_sampling3d_3/split_1:output:47!up_sampling3d_3/split_1:output:48!up_sampling3d_3/split_1:output:48!up_sampling3d_3/split_1:output:49!up_sampling3d_3/split_1:output:49!up_sampling3d_3/split_1:output:50!up_sampling3d_3/split_1:output:50!up_sampling3d_3/split_1:output:51!up_sampling3d_3/split_1:output:51!up_sampling3d_3/split_1:output:52!up_sampling3d_3/split_1:output:52!up_sampling3d_3/split_1:output:53!up_sampling3d_3/split_1:output:53!up_sampling3d_3/split_1:output:54!up_sampling3d_3/split_1:output:54!up_sampling3d_3/split_1:output:55!up_sampling3d_3/split_1:output:55!up_sampling3d_3/split_1:output:56!up_sampling3d_3/split_1:output:56!up_sampling3d_3/split_1:output:57!up_sampling3d_3/split_1:output:57!up_sampling3d_3/split_1:output:58!up_sampling3d_3/split_1:output:58!up_sampling3d_3/split_1:output:59!up_sampling3d_3/split_1:output:59!up_sampling3d_3/split_1:output:60!up_sampling3d_3/split_1:output:60!up_sampling3d_3/split_1:output:61!up_sampling3d_3/split_1:output:61!up_sampling3d_3/split_1:output:62!up_sampling3d_3/split_1:output:62!up_sampling3d_3/split_1:output:63!up_sampling3d_3/split_1:output:63!up_sampling3d_3/split_1:output:64!up_sampling3d_3/split_1:output:64!up_sampling3d_3/split_1:output:65!up_sampling3d_3/split_1:output:65!up_sampling3d_3/split_1:output:66!up_sampling3d_3/split_1:output:66!up_sampling3d_3/split_1:output:67!up_sampling3d_3/split_1:output:67!up_sampling3d_3/split_1:output:68!up_sampling3d_3/split_1:output:68!up_sampling3d_3/split_1:output:69!up_sampling3d_3/split_1:output:69!up_sampling3d_3/split_1:output:70!up_sampling3d_3/split_1:output:70!up_sampling3d_3/split_1:output:71!up_sampling3d_3/split_1:output:71!up_sampling3d_3/split_1:output:72!up_sampling3d_3/split_1:output:72!up_sampling3d_3/split_1:output:73!up_sampling3d_3/split_1:output:73!up_sampling3d_3/split_1:output:74!up_sampling3d_3/split_1:output:74!up_sampling3d_3/split_1:output:75!up_sampling3d_3/split_1:output:75!up_sampling3d_3/split_1:output:76!up_sampling3d_3/split_1:output:76!up_sampling3d_3/split_1:output:77!up_sampling3d_3/split_1:output:77!up_sampling3d_3/split_1:output:78!up_sampling3d_3/split_1:output:78!up_sampling3d_3/split_1:output:79!up_sampling3d_3/split_1:output:79!up_sampling3d_3/split_1:output:80!up_sampling3d_3/split_1:output:80!up_sampling3d_3/split_1:output:81!up_sampling3d_3/split_1:output:81!up_sampling3d_3/split_1:output:82!up_sampling3d_3/split_1:output:82!up_sampling3d_3/split_1:output:83!up_sampling3d_3/split_1:output:83!up_sampling3d_3/split_1:output:84!up_sampling3d_3/split_1:output:84!up_sampling3d_3/split_1:output:85!up_sampling3d_3/split_1:output:85!up_sampling3d_3/split_1:output:86!up_sampling3d_3/split_1:output:86!up_sampling3d_3/split_1:output:87!up_sampling3d_3/split_1:output:87!up_sampling3d_3/split_1:output:88!up_sampling3d_3/split_1:output:88!up_sampling3d_3/split_1:output:89!up_sampling3d_3/split_1:output:89!up_sampling3d_3/split_1:output:90!up_sampling3d_3/split_1:output:90!up_sampling3d_3/split_1:output:91!up_sampling3d_3/split_1:output:91!up_sampling3d_3/split_1:output:92!up_sampling3d_3/split_1:output:92!up_sampling3d_3/split_1:output:93!up_sampling3d_3/split_1:output:93!up_sampling3d_3/split_1:output:94!up_sampling3d_3/split_1:output:94!up_sampling3d_3/split_1:output:95!up_sampling3d_3/split_1:output:95&up_sampling3d_3/concat_1/axis:output:0*
N?*
T0*5
_output_shapes#
!:???????????`@2
up_sampling3d_3/concat_1t
up_sampling3d_3/Const_2Const*
_output_shapes
: *
dtype0*
value	B :`2
up_sampling3d_3/Const_2?
!up_sampling3d_3/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!up_sampling3d_3/split_2/split_dim?
up_sampling3d_3/split_2Split*up_sampling3d_3/split_2/split_dim:output:0!up_sampling3d_3/concat_1:output:0*
T0*?
_output_shapes?
?:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@*
	num_split`2
up_sampling3d_3/split_2?
up_sampling3d_3/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
up_sampling3d_3/concat_2/axis?5
up_sampling3d_3/concat_2ConcatV2 up_sampling3d_3/split_2:output:0 up_sampling3d_3/split_2:output:0 up_sampling3d_3/split_2:output:1 up_sampling3d_3/split_2:output:1 up_sampling3d_3/split_2:output:2 up_sampling3d_3/split_2:output:2 up_sampling3d_3/split_2:output:3 up_sampling3d_3/split_2:output:3 up_sampling3d_3/split_2:output:4 up_sampling3d_3/split_2:output:4 up_sampling3d_3/split_2:output:5 up_sampling3d_3/split_2:output:5 up_sampling3d_3/split_2:output:6 up_sampling3d_3/split_2:output:6 up_sampling3d_3/split_2:output:7 up_sampling3d_3/split_2:output:7 up_sampling3d_3/split_2:output:8 up_sampling3d_3/split_2:output:8 up_sampling3d_3/split_2:output:9 up_sampling3d_3/split_2:output:9!up_sampling3d_3/split_2:output:10!up_sampling3d_3/split_2:output:10!up_sampling3d_3/split_2:output:11!up_sampling3d_3/split_2:output:11!up_sampling3d_3/split_2:output:12!up_sampling3d_3/split_2:output:12!up_sampling3d_3/split_2:output:13!up_sampling3d_3/split_2:output:13!up_sampling3d_3/split_2:output:14!up_sampling3d_3/split_2:output:14!up_sampling3d_3/split_2:output:15!up_sampling3d_3/split_2:output:15!up_sampling3d_3/split_2:output:16!up_sampling3d_3/split_2:output:16!up_sampling3d_3/split_2:output:17!up_sampling3d_3/split_2:output:17!up_sampling3d_3/split_2:output:18!up_sampling3d_3/split_2:output:18!up_sampling3d_3/split_2:output:19!up_sampling3d_3/split_2:output:19!up_sampling3d_3/split_2:output:20!up_sampling3d_3/split_2:output:20!up_sampling3d_3/split_2:output:21!up_sampling3d_3/split_2:output:21!up_sampling3d_3/split_2:output:22!up_sampling3d_3/split_2:output:22!up_sampling3d_3/split_2:output:23!up_sampling3d_3/split_2:output:23!up_sampling3d_3/split_2:output:24!up_sampling3d_3/split_2:output:24!up_sampling3d_3/split_2:output:25!up_sampling3d_3/split_2:output:25!up_sampling3d_3/split_2:output:26!up_sampling3d_3/split_2:output:26!up_sampling3d_3/split_2:output:27!up_sampling3d_3/split_2:output:27!up_sampling3d_3/split_2:output:28!up_sampling3d_3/split_2:output:28!up_sampling3d_3/split_2:output:29!up_sampling3d_3/split_2:output:29!up_sampling3d_3/split_2:output:30!up_sampling3d_3/split_2:output:30!up_sampling3d_3/split_2:output:31!up_sampling3d_3/split_2:output:31!up_sampling3d_3/split_2:output:32!up_sampling3d_3/split_2:output:32!up_sampling3d_3/split_2:output:33!up_sampling3d_3/split_2:output:33!up_sampling3d_3/split_2:output:34!up_sampling3d_3/split_2:output:34!up_sampling3d_3/split_2:output:35!up_sampling3d_3/split_2:output:35!up_sampling3d_3/split_2:output:36!up_sampling3d_3/split_2:output:36!up_sampling3d_3/split_2:output:37!up_sampling3d_3/split_2:output:37!up_sampling3d_3/split_2:output:38!up_sampling3d_3/split_2:output:38!up_sampling3d_3/split_2:output:39!up_sampling3d_3/split_2:output:39!up_sampling3d_3/split_2:output:40!up_sampling3d_3/split_2:output:40!up_sampling3d_3/split_2:output:41!up_sampling3d_3/split_2:output:41!up_sampling3d_3/split_2:output:42!up_sampling3d_3/split_2:output:42!up_sampling3d_3/split_2:output:43!up_sampling3d_3/split_2:output:43!up_sampling3d_3/split_2:output:44!up_sampling3d_3/split_2:output:44!up_sampling3d_3/split_2:output:45!up_sampling3d_3/split_2:output:45!up_sampling3d_3/split_2:output:46!up_sampling3d_3/split_2:output:46!up_sampling3d_3/split_2:output:47!up_sampling3d_3/split_2:output:47!up_sampling3d_3/split_2:output:48!up_sampling3d_3/split_2:output:48!up_sampling3d_3/split_2:output:49!up_sampling3d_3/split_2:output:49!up_sampling3d_3/split_2:output:50!up_sampling3d_3/split_2:output:50!up_sampling3d_3/split_2:output:51!up_sampling3d_3/split_2:output:51!up_sampling3d_3/split_2:output:52!up_sampling3d_3/split_2:output:52!up_sampling3d_3/split_2:output:53!up_sampling3d_3/split_2:output:53!up_sampling3d_3/split_2:output:54!up_sampling3d_3/split_2:output:54!up_sampling3d_3/split_2:output:55!up_sampling3d_3/split_2:output:55!up_sampling3d_3/split_2:output:56!up_sampling3d_3/split_2:output:56!up_sampling3d_3/split_2:output:57!up_sampling3d_3/split_2:output:57!up_sampling3d_3/split_2:output:58!up_sampling3d_3/split_2:output:58!up_sampling3d_3/split_2:output:59!up_sampling3d_3/split_2:output:59!up_sampling3d_3/split_2:output:60!up_sampling3d_3/split_2:output:60!up_sampling3d_3/split_2:output:61!up_sampling3d_3/split_2:output:61!up_sampling3d_3/split_2:output:62!up_sampling3d_3/split_2:output:62!up_sampling3d_3/split_2:output:63!up_sampling3d_3/split_2:output:63!up_sampling3d_3/split_2:output:64!up_sampling3d_3/split_2:output:64!up_sampling3d_3/split_2:output:65!up_sampling3d_3/split_2:output:65!up_sampling3d_3/split_2:output:66!up_sampling3d_3/split_2:output:66!up_sampling3d_3/split_2:output:67!up_sampling3d_3/split_2:output:67!up_sampling3d_3/split_2:output:68!up_sampling3d_3/split_2:output:68!up_sampling3d_3/split_2:output:69!up_sampling3d_3/split_2:output:69!up_sampling3d_3/split_2:output:70!up_sampling3d_3/split_2:output:70!up_sampling3d_3/split_2:output:71!up_sampling3d_3/split_2:output:71!up_sampling3d_3/split_2:output:72!up_sampling3d_3/split_2:output:72!up_sampling3d_3/split_2:output:73!up_sampling3d_3/split_2:output:73!up_sampling3d_3/split_2:output:74!up_sampling3d_3/split_2:output:74!up_sampling3d_3/split_2:output:75!up_sampling3d_3/split_2:output:75!up_sampling3d_3/split_2:output:76!up_sampling3d_3/split_2:output:76!up_sampling3d_3/split_2:output:77!up_sampling3d_3/split_2:output:77!up_sampling3d_3/split_2:output:78!up_sampling3d_3/split_2:output:78!up_sampling3d_3/split_2:output:79!up_sampling3d_3/split_2:output:79!up_sampling3d_3/split_2:output:80!up_sampling3d_3/split_2:output:80!up_sampling3d_3/split_2:output:81!up_sampling3d_3/split_2:output:81!up_sampling3d_3/split_2:output:82!up_sampling3d_3/split_2:output:82!up_sampling3d_3/split_2:output:83!up_sampling3d_3/split_2:output:83!up_sampling3d_3/split_2:output:84!up_sampling3d_3/split_2:output:84!up_sampling3d_3/split_2:output:85!up_sampling3d_3/split_2:output:85!up_sampling3d_3/split_2:output:86!up_sampling3d_3/split_2:output:86!up_sampling3d_3/split_2:output:87!up_sampling3d_3/split_2:output:87!up_sampling3d_3/split_2:output:88!up_sampling3d_3/split_2:output:88!up_sampling3d_3/split_2:output:89!up_sampling3d_3/split_2:output:89!up_sampling3d_3/split_2:output:90!up_sampling3d_3/split_2:output:90!up_sampling3d_3/split_2:output:91!up_sampling3d_3/split_2:output:91!up_sampling3d_3/split_2:output:92!up_sampling3d_3/split_2:output:92!up_sampling3d_3/split_2:output:93!up_sampling3d_3/split_2:output:93!up_sampling3d_3/split_2:output:94!up_sampling3d_3/split_2:output:94!up_sampling3d_3/split_2:output:95!up_sampling3d_3/split_2:output:95&up_sampling3d_3/concat_2/axis:output:0*
N?*
T0*6
_output_shapes$
": ????????????@2
up_sampling3d_3/concat_2x
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_4/concat/axis?
concatenate_4/concatConcatV2!up_sampling3d_3/concat_2:output:0concatenate/concat:output:0"concatenate_4/concat/axis:output:0*
N*
T0*6
_output_shapes$
": ????????????B2
concatenate_4/concat?
conv3d_9/Conv3D/ReadVariableOpReadVariableOp'conv3d_9_conv3d_readvariableop_resource**
_output_shapes
:B *
dtype02 
conv3d_9/Conv3D/ReadVariableOp?
conv3d_9/Conv3DConv3Dconcatenate_4/concat:output:0&conv3d_9/Conv3D/ReadVariableOp:value:0*
T0*6
_output_shapes$
": ???????????? *
paddingSAME*
strides	
2
conv3d_9/Conv3D?
conv3d_9/BiasAdd/ReadVariableOpReadVariableOp(conv3d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv3d_9/BiasAdd/ReadVariableOp?
conv3d_9/BiasAddBiasAddconv3d_9/Conv3D:output:0'conv3d_9/BiasAdd/ReadVariableOp:value:0*
T0*6
_output_shapes$
": ???????????? 2
conv3d_9/BiasAdd?
leaky_re_lu_9/LeakyRelu	LeakyReluconv3d_9/BiasAdd:output:0*6
_output_shapes$
": ???????????? 2
leaky_re_lu_9/LeakyRelu?
conv3d_10/Conv3D/ReadVariableOpReadVariableOp(conv3d_10_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype02!
conv3d_10/Conv3D/ReadVariableOp?
conv3d_10/Conv3DConv3D%leaky_re_lu_9/LeakyRelu:activations:0'conv3d_10/Conv3D/ReadVariableOp:value:0*
T0*6
_output_shapes$
": ???????????? *
paddingSAME*
strides	
2
conv3d_10/Conv3D?
 conv3d_10/BiasAdd/ReadVariableOpReadVariableOp)conv3d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3d_10/BiasAdd/ReadVariableOp?
conv3d_10/BiasAddBiasAddconv3d_10/Conv3D:output:0(conv3d_10/BiasAdd/ReadVariableOp:value:0*
T0*6
_output_shapes$
": ???????????? 2
conv3d_10/BiasAdd?
leaky_re_lu_10/LeakyRelu	LeakyReluconv3d_10/BiasAdd:output:0*6
_output_shapes$
": ???????????? 2
leaky_re_lu_10/LeakyRelu?
disp/Conv3D/ReadVariableOpReadVariableOp#disp_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02
disp/Conv3D/ReadVariableOp?
disp/Conv3DConv3D&leaky_re_lu_10/LeakyRelu:activations:0"disp/Conv3D/ReadVariableOp:value:0*
T0*6
_output_shapes$
": ????????????*
paddingSAME*
strides	
2
disp/Conv3D?
disp/BiasAdd/ReadVariableOpReadVariableOp$disp_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
disp/BiasAdd/ReadVariableOp?
disp/BiasAddBiasAdddisp/Conv3D:output:0#disp/BiasAdd/ReadVariableOp:value:0*
T0*6
_output_shapes$
": ????????????2
disp/BiasAdd?
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
transformer/Reshape_1Reshapedisp/BiasAdd:output:0$transformer/Reshape_1/shape:output:0*
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
 *-
body%R#
!transformer_map_while_body_275046*-
cond%R#
!transformer_map_while_cond_275045*!
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

Identity|

Identity_1Identitydisp/BiasAdd:output:0*
T0*6
_output_shapes$
": ????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ????????????: ????????????:::::::::::::::::::::::::` \
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/0:`\
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/1:
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
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?"
e
I__inference_up_sampling3d_layer_call_and_return_conditional_losses_272809

inputs
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0inputs*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2
split\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11concat/axis:output:0*
N*
T0*3
_output_shapes!
:?????????@2
concatT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2	
split_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:?????????@2

concat_1T
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2	
split_2`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis?
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:?????????@2

concat_2q
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
??
?
map_while_body_277294
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
?
~
)__inference_conv3d_2_layer_call_fn_272457

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8????????????????????????????????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_2724472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8????????????????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
Z
.__inference_concatenate_3_layer_call_fn_276907
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*3
_output_shapes!
:?????????````* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_2731562
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????````2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????```@:?????????``` :] Y
3
_output_shapes!
:?????????```@
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:?????????``` 
"
_user_specified_name
inputs/1
?
Z
.__inference_concatenate_4_layer_call_fn_277249
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????B* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_2735202
PartitionedCall{
IdentityIdentityPartitionedCall:output:0*
T0*6
_output_shapes$
": ????????????B2

Identity"
identityIdentity:output:0*W
_input_shapesF
D: ????????????@: ????????????:` \
6
_output_shapes$
": ????????????@
"
_user_specified_name
inputs/0:`\
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/1
?
e
I__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_276528

inputs
identity`
	LeakyRelu	LeakyReluinputs*3
_output_shapes!
:?????????@2
	LeakyReluw
IdentityIdentityLeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_1_layer_call_fn_276503

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_2726952
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????000@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????000@:[ W
3
_output_shapes!
:?????????000@
 
_user_specified_nameinputs
?
~
)__inference_conv3d_7_layer_call_fn_272562

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8????????????????????????????????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_7_layer_call_and_return_conditional_losses_2725522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8????????????????????????????????????`::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????`
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?

?
D__inference_conv3d_3_layer_call_and_return_conditional_losses_272468

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????@*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8????????????????????????????????????@:::v r
N
_output_shapes<
::8????????????????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
??
?
C__inference_model_1_layer_call_and_return_conditional_losses_273995
input_1
input_2
conv3d_273913
conv3d_273915
conv3d_1_273919
conv3d_1_273921
conv3d_2_273925
conv3d_2_273927
conv3d_3_273931
conv3d_3_273933
conv3d_4_273937
conv3d_4_273939
conv3d_5_273945
conv3d_5_273947
conv3d_6_273953
conv3d_6_273955
conv3d_7_273961
conv3d_7_273963
conv3d_8_273967
conv3d_8_273969
conv3d_9_273975
conv3d_9_273977
conv3d_10_273981
conv3d_10_273983
disp_273987
disp_273989
identity

identity_1??conv3d/StatefulPartitionedCall? conv3d_1/StatefulPartitionedCall?!conv3d_10/StatefulPartitionedCall? conv3d_2/StatefulPartitionedCall? conv3d_3/StatefulPartitionedCall? conv3d_4/StatefulPartitionedCall? conv3d_5/StatefulPartitionedCall? conv3d_6/StatefulPartitionedCall? conv3d_7/StatefulPartitionedCall? conv3d_8/StatefulPartitionedCall? conv3d_9/StatefulPartitionedCall?disp/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCallinput_1input_2*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2726582
concatenate/PartitionedCall?
conv3d/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv3d_273913conv3d_273915*
Tin
2*
Tout
2*3
_output_shapes!
:?????????``` *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_2724052 
conv3d/StatefulPartitionedCall?
leaky_re_lu/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????``` * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_2726772
leaky_re_lu/PartitionedCall?
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0conv3d_1_273919conv3d_1_273921*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_2724262"
 conv3d_1/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_2726952
leaky_re_lu_1/PartitionedCall?
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0conv3d_2_273925conv3d_2_273927*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_2724472"
 conv3d_2/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_2727132
leaky_re_lu_2/PartitionedCall?
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0conv3d_3_273931conv3d_3_273933*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_2724682"
 conv3d_3/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_2727312
leaky_re_lu_3/PartitionedCall?
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0conv3d_4_273937conv3d_4_273939*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_2724892"
 conv3d_4/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_2727492
leaky_re_lu_4/PartitionedCall?
up_sampling3d/PartitionedCallPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_up_sampling3d_layer_call_and_return_conditional_losses_2728092
up_sampling3d/PartitionedCall?
concatenate_1/PartitionedCallPartitionedCall&up_sampling3d/PartitionedCall:output:0&leaky_re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*4
_output_shapes"
 :??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_2728242
concatenate_1/PartitionedCall?
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv3d_5_273945conv3d_5_273947*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_2725102"
 conv3d_5/StatefulPartitionedCall?
leaky_re_lu_5/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_2728432
leaky_re_lu_5/PartitionedCall?
up_sampling3d_1/PartitionedCallPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_2729392!
up_sampling3d_1/PartitionedCall?
concatenate_2/PartitionedCallPartitionedCall(up_sampling3d_1/PartitionedCall:output:0&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*4
_output_shapes"
 :?????????000?* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_2729542
concatenate_2/PartitionedCall?
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv3d_6_273953conv3d_6_273955*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_6_layer_call_and_return_conditional_losses_2725312"
 conv3d_6/StatefulPartitionedCall?
leaky_re_lu_6/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_2729732
leaky_re_lu_6/PartitionedCall?
up_sampling3d_2/PartitionedCallPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_2731412!
up_sampling3d_2/PartitionedCall?
concatenate_3/PartitionedCallPartitionedCall(up_sampling3d_2/PartitionedCall:output:0$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????````* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_2731562
concatenate_3/PartitionedCall?
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv3d_7_273961conv3d_7_273963*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_7_layer_call_and_return_conditional_losses_2725522"
 conv3d_7/StatefulPartitionedCall?
leaky_re_lu_7/PartitionedCallPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_2731752
leaky_re_lu_7/PartitionedCall?
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0conv3d_8_273967conv3d_8_273969*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_8_layer_call_and_return_conditional_losses_2725732"
 conv3d_8/StatefulPartitionedCall?
leaky_re_lu_8/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_2731932
leaky_re_lu_8/PartitionedCall?
up_sampling3d_3/PartitionedCallPartitionedCall&leaky_re_lu_8/PartitionedCall:output:0*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_2735052!
up_sampling3d_3/PartitionedCall?
concatenate_4/PartitionedCallPartitionedCall(up_sampling3d_3/PartitionedCall:output:0$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????B* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_2735202
concatenate_4/PartitionedCall?
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv3d_9_273975conv3d_9_273977*
Tin
2*
Tout
2*6
_output_shapes$
": ???????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_9_layer_call_and_return_conditional_losses_2725942"
 conv3d_9/StatefulPartitionedCall?
leaky_re_lu_9/PartitionedCallPartitionedCall)conv3d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*6
_output_shapes$
": ???????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_2735392
leaky_re_lu_9/PartitionedCall?
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_9/PartitionedCall:output:0conv3d_10_273981conv3d_10_273983*
Tin
2*
Tout
2*6
_output_shapes$
": ???????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_conv3d_10_layer_call_and_return_conditional_losses_2726152#
!conv3d_10/StatefulPartitionedCall?
leaky_re_lu_10/PartitionedCallPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*6
_output_shapes$
": ???????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_2735572 
leaky_re_lu_10/PartitionedCall?
disp/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_10/PartitionedCall:output:0disp_273987disp_273989*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_disp_layer_call_and_return_conditional_losses_2726362
disp/StatefulPartitionedCall?
transformer/PartitionedCallPartitionedCallinput_1%disp/StatefulPartitionedCall:output:0*
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
GPU 

CPU2J 8*P
fKRI
G__inference_transformer_layer_call_and_return_conditional_losses_2738972
transformer/PartitionedCall?
IdentityIdentity$transformer/PartitionedCall:output:0^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall^disp/StatefulPartitionedCall*
T0*6
_output_shapes$
": ????????????2

Identity?

Identity_1Identity%disp/StatefulPartitionedCall:output:0^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall^disp/StatefulPartitionedCall*
T0*6
_output_shapes$
": ????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ????????????: ????????????::::::::::::::::::::::::2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall2<
disp/StatefulPartitionedCalldisp/StatefulPartitionedCall:_ [
6
_output_shapes$
": ????????????
!
_user_specified_name	input_1:_[
6
_output_shapes$
": ????????????
!
_user_specified_name	input_2:
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
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
!transformer_map_while_body_275046&
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
?

?
D__inference_conv3d_8_layer_call_and_return_conditional_losses_272573

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????@*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8????????????????????????????????????@:::v r
N
_output_shapes<
::8????????????????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
J
.__inference_leaky_re_lu_4_layer_call_fn_276533

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_2727492
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
X
,__inference_transformer_layer_call_fn_277602
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
GPU 

CPU2J 8*P
fKRI
G__inference_transformer_layer_call_and_return_conditional_losses_2738972
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
?
J
.__inference_leaky_re_lu_2_layer_call_fn_276513

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_2727132
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
u
I__inference_concatenate_2_layer_call_and_return_conditional_losses_276713
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*4
_output_shapes"
 :?????????000?2
concatp
IdentityIdentityconcat:output:0*
T0*4
_output_shapes"
 :?????????000?2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????000@:?????????000@:] Y
3
_output_shapes!
:?????????000@
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:?????????000@
"
_user_specified_name
inputs/1
?
~
)__inference_conv3d_1_layer_call_fn_272436

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8????????????????????????????????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_2724262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8???????????????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
J
.__inference_leaky_re_lu_5_layer_call_fn_276613

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_2728432
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?"
e
I__inference_up_sampling3d_layer_call_and_return_conditional_losses_276585

inputs
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0inputs*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2
split\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11concat/axis:output:0*
N*
T0*3
_output_shapes!
:?????????@2
concatT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2	
split_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:?????????@2

concat_1T
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2	
split_2`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis?
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:?????????@2

concat_2q
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?

?
D__inference_conv3d_2_layer_call_and_return_conditional_losses_272447

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????@*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8????????????????????????????????????@:::v r
N
_output_shapes<
::8????????????????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
e
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_272973

inputs
identity`
	LeakyRelu	LeakyReluinputs*3
_output_shapes!
:?????????000@2
	LeakyReluw
IdentityIdentityLeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????000@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????000@:[ W
3
_output_shapes!
:?????????000@
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_277254

inputs
identityc
	LeakyRelu	LeakyReluinputs*6
_output_shapes$
": ???????????? 2
	LeakyReluz
IdentityIdentityLeakyRelu:activations:0*
T0*6
_output_shapes$
": ???????????? 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ???????????? :^ Z
6
_output_shapes$
": ???????????? 
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_274340
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*X
_output_shapesF
D: ????????????: ????????????*:
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8**
f%R#
!__inference__wrapped_model_2723942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*6
_output_shapes$
": ????????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*6
_output_shapes$
": ????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ????????????: ????????????::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
6
_output_shapes$
": ????????????
!
_user_specified_name	input_1:_[
6
_output_shapes$
": ????????????
!
_user_specified_name	input_2:
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
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
J
.__inference_leaky_re_lu_6_layer_call_fn_276729

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_2729732
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????000@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????000@:[ W
3
_output_shapes!
:?????????000@
 
_user_specified_nameinputs
?
L
0__inference_up_sampling3d_3_layer_call_fn_277236

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_2735052
PartitionedCall{
IdentityIdentityPartitionedCall:output:0*
T0*6
_output_shapes$
": ????????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????```@:[ W
3
_output_shapes!
:?????????```@
 
_user_specified_nameinputs
?
~
)__inference_conv3d_6_layer_call_fn_272541

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8????????????????????????????????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_6_layer_call_and_return_conditional_losses_2725312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:9?????????????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:w s
O
_output_shapes=
;:9?????????????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
e
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_276922

inputs
identity`
	LeakyRelu	LeakyReluinputs*3
_output_shapes!
:?????????```@2
	LeakyReluw
IdentityIdentityLeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????```@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????```@:[ W
3
_output_shapes!
:?????????```@
 
_user_specified_nameinputs
?

?
D__inference_conv3d_1_layer_call_and_return_conditional_losses_272426

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: @*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????@*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8???????????????????????????????????? :::v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
H
,__inference_leaky_re_lu_layer_call_fn_276493

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*3
_output_shapes!
:?????????``` * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_2726772
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????``` 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????``` :[ W
3
_output_shapes!
:?????????``` 
 
_user_specified_nameinputs
?'
q
G__inference_transformer_layer_call_and_return_conditional_losses_273897

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
 *!
bodyR
map_while_body_273595*!
condR
map_while_cond_273594*!
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
?
~
)__inference_conv3d_8_layer_call_fn_272583

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8????????????????????????????????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_8_layer_call_and_return_conditional_losses_2725732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8????????????????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
c
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_276488

inputs
identity`
	LeakyRelu	LeakyReluinputs*3
_output_shapes!
:?????????``` 2
	LeakyReluw
IdentityIdentityLeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????``` 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????``` :[ W
3
_output_shapes!
:?????????``` 
 
_user_specified_nameinputs
?

*__inference_conv3d_10_layer_call_fn_272625

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8???????????????????????????????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_conv3d_10_layer_call_and_return_conditional_losses_2726152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8???????????????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
J
.__inference_leaky_re_lu_9_layer_call_fn_277259

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*6
_output_shapes$
": ???????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_2735392
PartitionedCall{
IdentityIdentityPartitionedCall:output:0*
T0*6
_output_shapes$
": ???????????? 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ???????????? :^ Z
6
_output_shapes$
": ???????????? 
 
_user_specified_nameinputs
?
?
(__inference_model_1_layer_call_fn_276470
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*X
_output_shapesF
D: ????????????: ????????????*:
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2742292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*6
_output_shapes$
": ????????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*6
_output_shapes$
": ????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ????????????: ????????????::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/0:`\
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/1:
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
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
J
.__inference_leaky_re_lu_7_layer_call_fn_276917

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_2731752
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????```@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????```@:[ W
3
_output_shapes!
:?????????```@
 
_user_specified_nameinputs
?

?
E__inference_conv3d_10_layer_call_and_return_conditional_losses_272615

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8???????????????????????????????????? *
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8???????????????????????????????????? :::v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
map_while_cond_277293
map_while_loop_counter
map_strided_slice
placeholder
placeholder_1
less_map_strided_slice2
.map_while_cond_277293___redundant_placeholder02
.map_while_cond_277293___redundant_placeholder1
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
?
J
.__inference_up_sampling3d_layer_call_fn_276590

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_up_sampling3d_layer_call_and_return_conditional_losses_2728092
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
L
0__inference_up_sampling3d_1_layer_call_fn_276706

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_2729392
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????000@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
c
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_272677

inputs
identity`
	LeakyRelu	LeakyReluinputs*3
_output_shapes!
:?????????``` 2
	LeakyReluw
IdentityIdentityLeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????``` 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????``` :[ W
3
_output_shapes!
:?????????``` 
 
_user_specified_nameinputs
?
?
(__inference_model_1_layer_call_fn_276414
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*X
_output_shapesF
D: ????????????: ????????????*:
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2740862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*6
_output_shapes$
": ????????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*6
_output_shapes$
": ????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ????????????: ????????????::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/0:`\
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/1:
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
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
Z
.__inference_concatenate_1_layer_call_fn_276603
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*4
_output_shapes"
 :??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_2728242
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????@:?????????@:] Y
3
_output_shapes!
:?????????@
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:?????????@
"
_user_specified_name
inputs/1
?
?
(__inference_model_1_layer_call_fn_274139
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*X
_output_shapesF
D: ????????????: ????????????*:
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2740862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*6
_output_shapes$
": ????????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*6
_output_shapes$
": ????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ????????????: ????????????::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
6
_output_shapes$
": ????????????
!
_user_specified_name	input_1:_[
6
_output_shapes$
": ????????????
!
_user_specified_name	input_2:
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
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
e
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_273175

inputs
identity`
	LeakyRelu	LeakyReluinputs*3
_output_shapes!
:?????????```@2
	LeakyReluw
IdentityIdentityLeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????```@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????```@:[ W
3
_output_shapes!
:?????????```@
 
_user_specified_nameinputs
?
K
/__inference_leaky_re_lu_10_layer_call_fn_277269

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*6
_output_shapes$
": ???????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_2735572
PartitionedCall{
IdentityIdentityPartitionedCall:output:0*
T0*6
_output_shapes$
": ???????????? 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ???????????? :^ Z
6
_output_shapes$
": ???????????? 
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_276508

inputs
identity`
	LeakyRelu	LeakyReluinputs*3
_output_shapes!
:?????????@2
	LeakyReluw
IdentityIdentityLeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
u
I__inference_concatenate_4_layer_call_and_return_conditional_losses_277243
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*6
_output_shapes$
": ????????????B2
concatr
IdentityIdentityconcat:output:0*
T0*6
_output_shapes$
": ????????????B2

Identity"
identityIdentity:output:0*W
_input_shapesF
D: ????????????@: ????????????:` \
6
_output_shapes$
": ????????????@
"
_user_specified_name
inputs/0:`\
6
_output_shapes$
": ????????????
"
_user_specified_name
inputs/1
?
~
)__inference_conv3d_4_layer_call_fn_272499

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8????????????????????????????????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_2724892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8????????????????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
map_while_cond_273594
map_while_loop_counter
map_strided_slice
placeholder
placeholder_1
less_map_strided_slice2
.map_while_cond_273594___redundant_placeholder02
.map_while_cond_273594___redundant_placeholder1
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
?
|
'__inference_conv3d_layer_call_fn_272415

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8???????????????????????????????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_2724052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8????????????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
??
?

!__inference__wrapped_model_272394
input_1
input_21
-model_1_conv3d_conv3d_readvariableop_resource2
.model_1_conv3d_biasadd_readvariableop_resource3
/model_1_conv3d_1_conv3d_readvariableop_resource4
0model_1_conv3d_1_biasadd_readvariableop_resource3
/model_1_conv3d_2_conv3d_readvariableop_resource4
0model_1_conv3d_2_biasadd_readvariableop_resource3
/model_1_conv3d_3_conv3d_readvariableop_resource4
0model_1_conv3d_3_biasadd_readvariableop_resource3
/model_1_conv3d_4_conv3d_readvariableop_resource4
0model_1_conv3d_4_biasadd_readvariableop_resource3
/model_1_conv3d_5_conv3d_readvariableop_resource4
0model_1_conv3d_5_biasadd_readvariableop_resource3
/model_1_conv3d_6_conv3d_readvariableop_resource4
0model_1_conv3d_6_biasadd_readvariableop_resource3
/model_1_conv3d_7_conv3d_readvariableop_resource4
0model_1_conv3d_7_biasadd_readvariableop_resource3
/model_1_conv3d_8_conv3d_readvariableop_resource4
0model_1_conv3d_8_biasadd_readvariableop_resource3
/model_1_conv3d_9_conv3d_readvariableop_resource4
0model_1_conv3d_9_biasadd_readvariableop_resource4
0model_1_conv3d_10_conv3d_readvariableop_resource5
1model_1_conv3d_10_biasadd_readvariableop_resource/
+model_1_disp_conv3d_readvariableop_resource0
,model_1_disp_biasadd_readvariableop_resource
identity

identity_1??
model_1/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
model_1/concatenate/concat/axis?
model_1/concatenate/concatConcatV2input_1input_2(model_1/concatenate/concat/axis:output:0*
N*
T0*6
_output_shapes$
": ????????????2
model_1/concatenate/concat?
$model_1/conv3d/Conv3D/ReadVariableOpReadVariableOp-model_1_conv3d_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02&
$model_1/conv3d/Conv3D/ReadVariableOp?
model_1/conv3d/Conv3DConv3D#model_1/concatenate/concat:output:0,model_1/conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????``` *
paddingSAME*
strides	
2
model_1/conv3d/Conv3D?
%model_1/conv3d/BiasAdd/ReadVariableOpReadVariableOp.model_1_conv3d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%model_1/conv3d/BiasAdd/ReadVariableOp?
model_1/conv3d/BiasAddBiasAddmodel_1/conv3d/Conv3D:output:0-model_1/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????``` 2
model_1/conv3d/BiasAdd?
model_1/leaky_re_lu/LeakyRelu	LeakyRelumodel_1/conv3d/BiasAdd:output:0*3
_output_shapes!
:?????????``` 2
model_1/leaky_re_lu/LeakyRelu?
&model_1/conv3d_1/Conv3D/ReadVariableOpReadVariableOp/model_1_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype02(
&model_1/conv3d_1/Conv3D/ReadVariableOp?
model_1/conv3d_1/Conv3DConv3D+model_1/leaky_re_lu/LeakyRelu:activations:0.model_1/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????000@*
paddingSAME*
strides	
2
model_1/conv3d_1/Conv3D?
'model_1/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_1/conv3d_1/BiasAdd/ReadVariableOp?
model_1/conv3d_1/BiasAddBiasAdd model_1/conv3d_1/Conv3D:output:0/model_1/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????000@2
model_1/conv3d_1/BiasAdd?
model_1/leaky_re_lu_1/LeakyRelu	LeakyRelu!model_1/conv3d_1/BiasAdd:output:0*3
_output_shapes!
:?????????000@2!
model_1/leaky_re_lu_1/LeakyRelu?
&model_1/conv3d_2/Conv3D/ReadVariableOpReadVariableOp/model_1_conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype02(
&model_1/conv3d_2/Conv3D/ReadVariableOp?
model_1/conv3d_2/Conv3DConv3D-model_1/leaky_re_lu_1/LeakyRelu:activations:0.model_1/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
2
model_1/conv3d_2/Conv3D?
'model_1/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_1/conv3d_2/BiasAdd/ReadVariableOp?
model_1/conv3d_2/BiasAddBiasAdd model_1/conv3d_2/Conv3D:output:0/model_1/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@2
model_1/conv3d_2/BiasAdd?
model_1/leaky_re_lu_2/LeakyRelu	LeakyRelu!model_1/conv3d_2/BiasAdd:output:0*3
_output_shapes!
:?????????@2!
model_1/leaky_re_lu_2/LeakyRelu?
&model_1/conv3d_3/Conv3D/ReadVariableOpReadVariableOp/model_1_conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype02(
&model_1/conv3d_3/Conv3D/ReadVariableOp?
model_1/conv3d_3/Conv3DConv3D-model_1/leaky_re_lu_2/LeakyRelu:activations:0.model_1/conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
2
model_1/conv3d_3/Conv3D?
'model_1/conv3d_3/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_1/conv3d_3/BiasAdd/ReadVariableOp?
model_1/conv3d_3/BiasAddBiasAdd model_1/conv3d_3/Conv3D:output:0/model_1/conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@2
model_1/conv3d_3/BiasAdd?
model_1/leaky_re_lu_3/LeakyRelu	LeakyRelu!model_1/conv3d_3/BiasAdd:output:0*3
_output_shapes!
:?????????@2!
model_1/leaky_re_lu_3/LeakyRelu?
&model_1/conv3d_4/Conv3D/ReadVariableOpReadVariableOp/model_1_conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype02(
&model_1/conv3d_4/Conv3D/ReadVariableOp?
model_1/conv3d_4/Conv3DConv3D-model_1/leaky_re_lu_3/LeakyRelu:activations:0.model_1/conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
2
model_1/conv3d_4/Conv3D?
'model_1/conv3d_4/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_1/conv3d_4/BiasAdd/ReadVariableOp?
model_1/conv3d_4/BiasAddBiasAdd model_1/conv3d_4/Conv3D:output:0/model_1/conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@2
model_1/conv3d_4/BiasAdd?
model_1/leaky_re_lu_4/LeakyRelu	LeakyRelu!model_1/conv3d_4/BiasAdd:output:0*3
_output_shapes!
:?????????@2!
model_1/leaky_re_lu_4/LeakyRelu|
model_1/up_sampling3d/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
model_1/up_sampling3d/Const?
%model_1/up_sampling3d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%model_1/up_sampling3d/split/split_dim?
model_1/up_sampling3d/splitSplit.model_1/up_sampling3d/split/split_dim:output:0-model_1/leaky_re_lu_4/LeakyRelu:activations:0*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2
model_1/up_sampling3d/split?
!model_1/up_sampling3d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/up_sampling3d/concat/axis?
model_1/up_sampling3d/concatConcatV2$model_1/up_sampling3d/split:output:0$model_1/up_sampling3d/split:output:0$model_1/up_sampling3d/split:output:1$model_1/up_sampling3d/split:output:1$model_1/up_sampling3d/split:output:2$model_1/up_sampling3d/split:output:2$model_1/up_sampling3d/split:output:3$model_1/up_sampling3d/split:output:3$model_1/up_sampling3d/split:output:4$model_1/up_sampling3d/split:output:4$model_1/up_sampling3d/split:output:5$model_1/up_sampling3d/split:output:5$model_1/up_sampling3d/split:output:6$model_1/up_sampling3d/split:output:6$model_1/up_sampling3d/split:output:7$model_1/up_sampling3d/split:output:7$model_1/up_sampling3d/split:output:8$model_1/up_sampling3d/split:output:8$model_1/up_sampling3d/split:output:9$model_1/up_sampling3d/split:output:9%model_1/up_sampling3d/split:output:10%model_1/up_sampling3d/split:output:10%model_1/up_sampling3d/split:output:11%model_1/up_sampling3d/split:output:11*model_1/up_sampling3d/concat/axis:output:0*
N*
T0*3
_output_shapes!
:?????????@2
model_1/up_sampling3d/concat?
model_1/up_sampling3d/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
model_1/up_sampling3d/Const_1?
'model_1/up_sampling3d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_1/up_sampling3d/split_1/split_dim?
model_1/up_sampling3d/split_1Split0model_1/up_sampling3d/split_1/split_dim:output:0%model_1/up_sampling3d/concat:output:0*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2
model_1/up_sampling3d/split_1?
#model_1/up_sampling3d/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_1/up_sampling3d/concat_1/axis?	
model_1/up_sampling3d/concat_1ConcatV2&model_1/up_sampling3d/split_1:output:0&model_1/up_sampling3d/split_1:output:0&model_1/up_sampling3d/split_1:output:1&model_1/up_sampling3d/split_1:output:1&model_1/up_sampling3d/split_1:output:2&model_1/up_sampling3d/split_1:output:2&model_1/up_sampling3d/split_1:output:3&model_1/up_sampling3d/split_1:output:3&model_1/up_sampling3d/split_1:output:4&model_1/up_sampling3d/split_1:output:4&model_1/up_sampling3d/split_1:output:5&model_1/up_sampling3d/split_1:output:5&model_1/up_sampling3d/split_1:output:6&model_1/up_sampling3d/split_1:output:6&model_1/up_sampling3d/split_1:output:7&model_1/up_sampling3d/split_1:output:7&model_1/up_sampling3d/split_1:output:8&model_1/up_sampling3d/split_1:output:8&model_1/up_sampling3d/split_1:output:9&model_1/up_sampling3d/split_1:output:9'model_1/up_sampling3d/split_1:output:10'model_1/up_sampling3d/split_1:output:10'model_1/up_sampling3d/split_1:output:11'model_1/up_sampling3d/split_1:output:11,model_1/up_sampling3d/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:?????????@2 
model_1/up_sampling3d/concat_1?
model_1/up_sampling3d/Const_2Const*
_output_shapes
: *
dtype0*
value	B :2
model_1/up_sampling3d/Const_2?
'model_1/up_sampling3d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_1/up_sampling3d/split_2/split_dim?
model_1/up_sampling3d/split_2Split0model_1/up_sampling3d/split_2/split_dim:output:0'model_1/up_sampling3d/concat_1:output:0*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2
model_1/up_sampling3d/split_2?
#model_1/up_sampling3d/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_1/up_sampling3d/concat_2/axis?	
model_1/up_sampling3d/concat_2ConcatV2&model_1/up_sampling3d/split_2:output:0&model_1/up_sampling3d/split_2:output:0&model_1/up_sampling3d/split_2:output:1&model_1/up_sampling3d/split_2:output:1&model_1/up_sampling3d/split_2:output:2&model_1/up_sampling3d/split_2:output:2&model_1/up_sampling3d/split_2:output:3&model_1/up_sampling3d/split_2:output:3&model_1/up_sampling3d/split_2:output:4&model_1/up_sampling3d/split_2:output:4&model_1/up_sampling3d/split_2:output:5&model_1/up_sampling3d/split_2:output:5&model_1/up_sampling3d/split_2:output:6&model_1/up_sampling3d/split_2:output:6&model_1/up_sampling3d/split_2:output:7&model_1/up_sampling3d/split_2:output:7&model_1/up_sampling3d/split_2:output:8&model_1/up_sampling3d/split_2:output:8&model_1/up_sampling3d/split_2:output:9&model_1/up_sampling3d/split_2:output:9'model_1/up_sampling3d/split_2:output:10'model_1/up_sampling3d/split_2:output:10'model_1/up_sampling3d/split_2:output:11'model_1/up_sampling3d/split_2:output:11,model_1/up_sampling3d/concat_2/axis:output:0*
N*
T0*3
_output_shapes!
:?????????@2 
model_1/up_sampling3d/concat_2?
!model_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_1/concat/axis?
model_1/concatenate_1/concatConcatV2'model_1/up_sampling3d/concat_2:output:0-model_1/leaky_re_lu_2/LeakyRelu:activations:0*model_1/concatenate_1/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :??????????2
model_1/concatenate_1/concat?
&model_1/conv3d_5/Conv3D/ReadVariableOpReadVariableOp/model_1_conv3d_5_conv3d_readvariableop_resource*+
_output_shapes
:?@*
dtype02(
&model_1/conv3d_5/Conv3D/ReadVariableOp?
model_1/conv3d_5/Conv3DConv3D%model_1/concatenate_1/concat:output:0.model_1/conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
2
model_1/conv3d_5/Conv3D?
'model_1/conv3d_5/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_1/conv3d_5/BiasAdd/ReadVariableOp?
model_1/conv3d_5/BiasAddBiasAdd model_1/conv3d_5/Conv3D:output:0/model_1/conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@2
model_1/conv3d_5/BiasAdd?
model_1/leaky_re_lu_5/LeakyRelu	LeakyRelu!model_1/conv3d_5/BiasAdd:output:0*3
_output_shapes!
:?????????@2!
model_1/leaky_re_lu_5/LeakyRelu?
model_1/up_sampling3d_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
model_1/up_sampling3d_1/Const?
'model_1/up_sampling3d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_1/up_sampling3d_1/split/split_dim?
model_1/up_sampling3d_1/splitSplit0model_1/up_sampling3d_1/split/split_dim:output:0-model_1/leaky_re_lu_5/LeakyRelu:activations:0*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2
model_1/up_sampling3d_1/split?
#model_1/up_sampling3d_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_1/up_sampling3d_1/concat/axis?
model_1/up_sampling3d_1/concatConcatV2&model_1/up_sampling3d_1/split:output:0&model_1/up_sampling3d_1/split:output:0&model_1/up_sampling3d_1/split:output:1&model_1/up_sampling3d_1/split:output:1&model_1/up_sampling3d_1/split:output:2&model_1/up_sampling3d_1/split:output:2&model_1/up_sampling3d_1/split:output:3&model_1/up_sampling3d_1/split:output:3&model_1/up_sampling3d_1/split:output:4&model_1/up_sampling3d_1/split:output:4&model_1/up_sampling3d_1/split:output:5&model_1/up_sampling3d_1/split:output:5&model_1/up_sampling3d_1/split:output:6&model_1/up_sampling3d_1/split:output:6&model_1/up_sampling3d_1/split:output:7&model_1/up_sampling3d_1/split:output:7&model_1/up_sampling3d_1/split:output:8&model_1/up_sampling3d_1/split:output:8&model_1/up_sampling3d_1/split:output:9&model_1/up_sampling3d_1/split:output:9'model_1/up_sampling3d_1/split:output:10'model_1/up_sampling3d_1/split:output:10'model_1/up_sampling3d_1/split:output:11'model_1/up_sampling3d_1/split:output:11'model_1/up_sampling3d_1/split:output:12'model_1/up_sampling3d_1/split:output:12'model_1/up_sampling3d_1/split:output:13'model_1/up_sampling3d_1/split:output:13'model_1/up_sampling3d_1/split:output:14'model_1/up_sampling3d_1/split:output:14'model_1/up_sampling3d_1/split:output:15'model_1/up_sampling3d_1/split:output:15'model_1/up_sampling3d_1/split:output:16'model_1/up_sampling3d_1/split:output:16'model_1/up_sampling3d_1/split:output:17'model_1/up_sampling3d_1/split:output:17'model_1/up_sampling3d_1/split:output:18'model_1/up_sampling3d_1/split:output:18'model_1/up_sampling3d_1/split:output:19'model_1/up_sampling3d_1/split:output:19'model_1/up_sampling3d_1/split:output:20'model_1/up_sampling3d_1/split:output:20'model_1/up_sampling3d_1/split:output:21'model_1/up_sampling3d_1/split:output:21'model_1/up_sampling3d_1/split:output:22'model_1/up_sampling3d_1/split:output:22'model_1/up_sampling3d_1/split:output:23'model_1/up_sampling3d_1/split:output:23,model_1/up_sampling3d_1/concat/axis:output:0*
N0*
T0*3
_output_shapes!
:?????????0@2 
model_1/up_sampling3d_1/concat?
model_1/up_sampling3d_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2!
model_1/up_sampling3d_1/Const_1?
)model_1/up_sampling3d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)model_1/up_sampling3d_1/split_1/split_dim?
model_1/up_sampling3d_1/split_1Split2model_1/up_sampling3d_1/split_1/split_dim:output:0'model_1/up_sampling3d_1/concat:output:0*
T0*?
_output_shapes?
?:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@*
	num_split2!
model_1/up_sampling3d_1/split_1?
%model_1/up_sampling3d_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%model_1/up_sampling3d_1/concat_1/axis?
 model_1/up_sampling3d_1/concat_1ConcatV2(model_1/up_sampling3d_1/split_1:output:0(model_1/up_sampling3d_1/split_1:output:0(model_1/up_sampling3d_1/split_1:output:1(model_1/up_sampling3d_1/split_1:output:1(model_1/up_sampling3d_1/split_1:output:2(model_1/up_sampling3d_1/split_1:output:2(model_1/up_sampling3d_1/split_1:output:3(model_1/up_sampling3d_1/split_1:output:3(model_1/up_sampling3d_1/split_1:output:4(model_1/up_sampling3d_1/split_1:output:4(model_1/up_sampling3d_1/split_1:output:5(model_1/up_sampling3d_1/split_1:output:5(model_1/up_sampling3d_1/split_1:output:6(model_1/up_sampling3d_1/split_1:output:6(model_1/up_sampling3d_1/split_1:output:7(model_1/up_sampling3d_1/split_1:output:7(model_1/up_sampling3d_1/split_1:output:8(model_1/up_sampling3d_1/split_1:output:8(model_1/up_sampling3d_1/split_1:output:9(model_1/up_sampling3d_1/split_1:output:9)model_1/up_sampling3d_1/split_1:output:10)model_1/up_sampling3d_1/split_1:output:10)model_1/up_sampling3d_1/split_1:output:11)model_1/up_sampling3d_1/split_1:output:11)model_1/up_sampling3d_1/split_1:output:12)model_1/up_sampling3d_1/split_1:output:12)model_1/up_sampling3d_1/split_1:output:13)model_1/up_sampling3d_1/split_1:output:13)model_1/up_sampling3d_1/split_1:output:14)model_1/up_sampling3d_1/split_1:output:14)model_1/up_sampling3d_1/split_1:output:15)model_1/up_sampling3d_1/split_1:output:15)model_1/up_sampling3d_1/split_1:output:16)model_1/up_sampling3d_1/split_1:output:16)model_1/up_sampling3d_1/split_1:output:17)model_1/up_sampling3d_1/split_1:output:17)model_1/up_sampling3d_1/split_1:output:18)model_1/up_sampling3d_1/split_1:output:18)model_1/up_sampling3d_1/split_1:output:19)model_1/up_sampling3d_1/split_1:output:19)model_1/up_sampling3d_1/split_1:output:20)model_1/up_sampling3d_1/split_1:output:20)model_1/up_sampling3d_1/split_1:output:21)model_1/up_sampling3d_1/split_1:output:21)model_1/up_sampling3d_1/split_1:output:22)model_1/up_sampling3d_1/split_1:output:22)model_1/up_sampling3d_1/split_1:output:23)model_1/up_sampling3d_1/split_1:output:23.model_1/up_sampling3d_1/concat_1/axis:output:0*
N0*
T0*3
_output_shapes!
:?????????00@2"
 model_1/up_sampling3d_1/concat_1?
model_1/up_sampling3d_1/Const_2Const*
_output_shapes
: *
dtype0*
value	B :2!
model_1/up_sampling3d_1/Const_2?
)model_1/up_sampling3d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)model_1/up_sampling3d_1/split_2/split_dim?
model_1/up_sampling3d_1/split_2Split2model_1/up_sampling3d_1/split_2/split_dim:output:0)model_1/up_sampling3d_1/concat_1:output:0*
T0*?
_output_shapes?
?:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@*
	num_split2!
model_1/up_sampling3d_1/split_2?
%model_1/up_sampling3d_1/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%model_1/up_sampling3d_1/concat_2/axis?
 model_1/up_sampling3d_1/concat_2ConcatV2(model_1/up_sampling3d_1/split_2:output:0(model_1/up_sampling3d_1/split_2:output:0(model_1/up_sampling3d_1/split_2:output:1(model_1/up_sampling3d_1/split_2:output:1(model_1/up_sampling3d_1/split_2:output:2(model_1/up_sampling3d_1/split_2:output:2(model_1/up_sampling3d_1/split_2:output:3(model_1/up_sampling3d_1/split_2:output:3(model_1/up_sampling3d_1/split_2:output:4(model_1/up_sampling3d_1/split_2:output:4(model_1/up_sampling3d_1/split_2:output:5(model_1/up_sampling3d_1/split_2:output:5(model_1/up_sampling3d_1/split_2:output:6(model_1/up_sampling3d_1/split_2:output:6(model_1/up_sampling3d_1/split_2:output:7(model_1/up_sampling3d_1/split_2:output:7(model_1/up_sampling3d_1/split_2:output:8(model_1/up_sampling3d_1/split_2:output:8(model_1/up_sampling3d_1/split_2:output:9(model_1/up_sampling3d_1/split_2:output:9)model_1/up_sampling3d_1/split_2:output:10)model_1/up_sampling3d_1/split_2:output:10)model_1/up_sampling3d_1/split_2:output:11)model_1/up_sampling3d_1/split_2:output:11)model_1/up_sampling3d_1/split_2:output:12)model_1/up_sampling3d_1/split_2:output:12)model_1/up_sampling3d_1/split_2:output:13)model_1/up_sampling3d_1/split_2:output:13)model_1/up_sampling3d_1/split_2:output:14)model_1/up_sampling3d_1/split_2:output:14)model_1/up_sampling3d_1/split_2:output:15)model_1/up_sampling3d_1/split_2:output:15)model_1/up_sampling3d_1/split_2:output:16)model_1/up_sampling3d_1/split_2:output:16)model_1/up_sampling3d_1/split_2:output:17)model_1/up_sampling3d_1/split_2:output:17)model_1/up_sampling3d_1/split_2:output:18)model_1/up_sampling3d_1/split_2:output:18)model_1/up_sampling3d_1/split_2:output:19)model_1/up_sampling3d_1/split_2:output:19)model_1/up_sampling3d_1/split_2:output:20)model_1/up_sampling3d_1/split_2:output:20)model_1/up_sampling3d_1/split_2:output:21)model_1/up_sampling3d_1/split_2:output:21)model_1/up_sampling3d_1/split_2:output:22)model_1/up_sampling3d_1/split_2:output:22)model_1/up_sampling3d_1/split_2:output:23)model_1/up_sampling3d_1/split_2:output:23.model_1/up_sampling3d_1/concat_2/axis:output:0*
N0*
T0*3
_output_shapes!
:?????????000@2"
 model_1/up_sampling3d_1/concat_2?
!model_1/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_2/concat/axis?
model_1/concatenate_2/concatConcatV2)model_1/up_sampling3d_1/concat_2:output:0-model_1/leaky_re_lu_1/LeakyRelu:activations:0*model_1/concatenate_2/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :?????????000?2
model_1/concatenate_2/concat?
&model_1/conv3d_6/Conv3D/ReadVariableOpReadVariableOp/model_1_conv3d_6_conv3d_readvariableop_resource*+
_output_shapes
:?@*
dtype02(
&model_1/conv3d_6/Conv3D/ReadVariableOp?
model_1/conv3d_6/Conv3DConv3D%model_1/concatenate_2/concat:output:0.model_1/conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????000@*
paddingSAME*
strides	
2
model_1/conv3d_6/Conv3D?
'model_1/conv3d_6/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_1/conv3d_6/BiasAdd/ReadVariableOp?
model_1/conv3d_6/BiasAddBiasAdd model_1/conv3d_6/Conv3D:output:0/model_1/conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????000@2
model_1/conv3d_6/BiasAdd?
model_1/leaky_re_lu_6/LeakyRelu	LeakyRelu!model_1/conv3d_6/BiasAdd:output:0*3
_output_shapes!
:?????????000@2!
model_1/leaky_re_lu_6/LeakyRelu?
model_1/up_sampling3d_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :02
model_1/up_sampling3d_2/Const?
'model_1/up_sampling3d_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_1/up_sampling3d_2/split/split_dim?
model_1/up_sampling3d_2/splitSplit0model_1/up_sampling3d_2/split/split_dim:output:0-model_1/leaky_re_lu_6/LeakyRelu:activations:0*
T0*?
_output_shapes?
?:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@*
	num_split02
model_1/up_sampling3d_2/split?
#model_1/up_sampling3d_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_1/up_sampling3d_2/concat/axis? 
model_1/up_sampling3d_2/concatConcatV2&model_1/up_sampling3d_2/split:output:0&model_1/up_sampling3d_2/split:output:0&model_1/up_sampling3d_2/split:output:1&model_1/up_sampling3d_2/split:output:1&model_1/up_sampling3d_2/split:output:2&model_1/up_sampling3d_2/split:output:2&model_1/up_sampling3d_2/split:output:3&model_1/up_sampling3d_2/split:output:3&model_1/up_sampling3d_2/split:output:4&model_1/up_sampling3d_2/split:output:4&model_1/up_sampling3d_2/split:output:5&model_1/up_sampling3d_2/split:output:5&model_1/up_sampling3d_2/split:output:6&model_1/up_sampling3d_2/split:output:6&model_1/up_sampling3d_2/split:output:7&model_1/up_sampling3d_2/split:output:7&model_1/up_sampling3d_2/split:output:8&model_1/up_sampling3d_2/split:output:8&model_1/up_sampling3d_2/split:output:9&model_1/up_sampling3d_2/split:output:9'model_1/up_sampling3d_2/split:output:10'model_1/up_sampling3d_2/split:output:10'model_1/up_sampling3d_2/split:output:11'model_1/up_sampling3d_2/split:output:11'model_1/up_sampling3d_2/split:output:12'model_1/up_sampling3d_2/split:output:12'model_1/up_sampling3d_2/split:output:13'model_1/up_sampling3d_2/split:output:13'model_1/up_sampling3d_2/split:output:14'model_1/up_sampling3d_2/split:output:14'model_1/up_sampling3d_2/split:output:15'model_1/up_sampling3d_2/split:output:15'model_1/up_sampling3d_2/split:output:16'model_1/up_sampling3d_2/split:output:16'model_1/up_sampling3d_2/split:output:17'model_1/up_sampling3d_2/split:output:17'model_1/up_sampling3d_2/split:output:18'model_1/up_sampling3d_2/split:output:18'model_1/up_sampling3d_2/split:output:19'model_1/up_sampling3d_2/split:output:19'model_1/up_sampling3d_2/split:output:20'model_1/up_sampling3d_2/split:output:20'model_1/up_sampling3d_2/split:output:21'model_1/up_sampling3d_2/split:output:21'model_1/up_sampling3d_2/split:output:22'model_1/up_sampling3d_2/split:output:22'model_1/up_sampling3d_2/split:output:23'model_1/up_sampling3d_2/split:output:23'model_1/up_sampling3d_2/split:output:24'model_1/up_sampling3d_2/split:output:24'model_1/up_sampling3d_2/split:output:25'model_1/up_sampling3d_2/split:output:25'model_1/up_sampling3d_2/split:output:26'model_1/up_sampling3d_2/split:output:26'model_1/up_sampling3d_2/split:output:27'model_1/up_sampling3d_2/split:output:27'model_1/up_sampling3d_2/split:output:28'model_1/up_sampling3d_2/split:output:28'model_1/up_sampling3d_2/split:output:29'model_1/up_sampling3d_2/split:output:29'model_1/up_sampling3d_2/split:output:30'model_1/up_sampling3d_2/split:output:30'model_1/up_sampling3d_2/split:output:31'model_1/up_sampling3d_2/split:output:31'model_1/up_sampling3d_2/split:output:32'model_1/up_sampling3d_2/split:output:32'model_1/up_sampling3d_2/split:output:33'model_1/up_sampling3d_2/split:output:33'model_1/up_sampling3d_2/split:output:34'model_1/up_sampling3d_2/split:output:34'model_1/up_sampling3d_2/split:output:35'model_1/up_sampling3d_2/split:output:35'model_1/up_sampling3d_2/split:output:36'model_1/up_sampling3d_2/split:output:36'model_1/up_sampling3d_2/split:output:37'model_1/up_sampling3d_2/split:output:37'model_1/up_sampling3d_2/split:output:38'model_1/up_sampling3d_2/split:output:38'model_1/up_sampling3d_2/split:output:39'model_1/up_sampling3d_2/split:output:39'model_1/up_sampling3d_2/split:output:40'model_1/up_sampling3d_2/split:output:40'model_1/up_sampling3d_2/split:output:41'model_1/up_sampling3d_2/split:output:41'model_1/up_sampling3d_2/split:output:42'model_1/up_sampling3d_2/split:output:42'model_1/up_sampling3d_2/split:output:43'model_1/up_sampling3d_2/split:output:43'model_1/up_sampling3d_2/split:output:44'model_1/up_sampling3d_2/split:output:44'model_1/up_sampling3d_2/split:output:45'model_1/up_sampling3d_2/split:output:45'model_1/up_sampling3d_2/split:output:46'model_1/up_sampling3d_2/split:output:46'model_1/up_sampling3d_2/split:output:47'model_1/up_sampling3d_2/split:output:47,model_1/up_sampling3d_2/concat/axis:output:0*
N`*
T0*3
_output_shapes!
:?????????`00@2 
model_1/up_sampling3d_2/concat?
model_1/up_sampling3d_2/Const_1Const*
_output_shapes
: *
dtype0*
value	B :02!
model_1/up_sampling3d_2/Const_1?
)model_1/up_sampling3d_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)model_1/up_sampling3d_2/split_1/split_dim?
model_1/up_sampling3d_2/split_1Split2model_1/up_sampling3d_2/split_1/split_dim:output:0'model_1/up_sampling3d_2/concat:output:0*
T0*?
_output_shapes?
?:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@:?????????`0@*
	num_split02!
model_1/up_sampling3d_2/split_1?
%model_1/up_sampling3d_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%model_1/up_sampling3d_2/concat_1/axis?!
 model_1/up_sampling3d_2/concat_1ConcatV2(model_1/up_sampling3d_2/split_1:output:0(model_1/up_sampling3d_2/split_1:output:0(model_1/up_sampling3d_2/split_1:output:1(model_1/up_sampling3d_2/split_1:output:1(model_1/up_sampling3d_2/split_1:output:2(model_1/up_sampling3d_2/split_1:output:2(model_1/up_sampling3d_2/split_1:output:3(model_1/up_sampling3d_2/split_1:output:3(model_1/up_sampling3d_2/split_1:output:4(model_1/up_sampling3d_2/split_1:output:4(model_1/up_sampling3d_2/split_1:output:5(model_1/up_sampling3d_2/split_1:output:5(model_1/up_sampling3d_2/split_1:output:6(model_1/up_sampling3d_2/split_1:output:6(model_1/up_sampling3d_2/split_1:output:7(model_1/up_sampling3d_2/split_1:output:7(model_1/up_sampling3d_2/split_1:output:8(model_1/up_sampling3d_2/split_1:output:8(model_1/up_sampling3d_2/split_1:output:9(model_1/up_sampling3d_2/split_1:output:9)model_1/up_sampling3d_2/split_1:output:10)model_1/up_sampling3d_2/split_1:output:10)model_1/up_sampling3d_2/split_1:output:11)model_1/up_sampling3d_2/split_1:output:11)model_1/up_sampling3d_2/split_1:output:12)model_1/up_sampling3d_2/split_1:output:12)model_1/up_sampling3d_2/split_1:output:13)model_1/up_sampling3d_2/split_1:output:13)model_1/up_sampling3d_2/split_1:output:14)model_1/up_sampling3d_2/split_1:output:14)model_1/up_sampling3d_2/split_1:output:15)model_1/up_sampling3d_2/split_1:output:15)model_1/up_sampling3d_2/split_1:output:16)model_1/up_sampling3d_2/split_1:output:16)model_1/up_sampling3d_2/split_1:output:17)model_1/up_sampling3d_2/split_1:output:17)model_1/up_sampling3d_2/split_1:output:18)model_1/up_sampling3d_2/split_1:output:18)model_1/up_sampling3d_2/split_1:output:19)model_1/up_sampling3d_2/split_1:output:19)model_1/up_sampling3d_2/split_1:output:20)model_1/up_sampling3d_2/split_1:output:20)model_1/up_sampling3d_2/split_1:output:21)model_1/up_sampling3d_2/split_1:output:21)model_1/up_sampling3d_2/split_1:output:22)model_1/up_sampling3d_2/split_1:output:22)model_1/up_sampling3d_2/split_1:output:23)model_1/up_sampling3d_2/split_1:output:23)model_1/up_sampling3d_2/split_1:output:24)model_1/up_sampling3d_2/split_1:output:24)model_1/up_sampling3d_2/split_1:output:25)model_1/up_sampling3d_2/split_1:output:25)model_1/up_sampling3d_2/split_1:output:26)model_1/up_sampling3d_2/split_1:output:26)model_1/up_sampling3d_2/split_1:output:27)model_1/up_sampling3d_2/split_1:output:27)model_1/up_sampling3d_2/split_1:output:28)model_1/up_sampling3d_2/split_1:output:28)model_1/up_sampling3d_2/split_1:output:29)model_1/up_sampling3d_2/split_1:output:29)model_1/up_sampling3d_2/split_1:output:30)model_1/up_sampling3d_2/split_1:output:30)model_1/up_sampling3d_2/split_1:output:31)model_1/up_sampling3d_2/split_1:output:31)model_1/up_sampling3d_2/split_1:output:32)model_1/up_sampling3d_2/split_1:output:32)model_1/up_sampling3d_2/split_1:output:33)model_1/up_sampling3d_2/split_1:output:33)model_1/up_sampling3d_2/split_1:output:34)model_1/up_sampling3d_2/split_1:output:34)model_1/up_sampling3d_2/split_1:output:35)model_1/up_sampling3d_2/split_1:output:35)model_1/up_sampling3d_2/split_1:output:36)model_1/up_sampling3d_2/split_1:output:36)model_1/up_sampling3d_2/split_1:output:37)model_1/up_sampling3d_2/split_1:output:37)model_1/up_sampling3d_2/split_1:output:38)model_1/up_sampling3d_2/split_1:output:38)model_1/up_sampling3d_2/split_1:output:39)model_1/up_sampling3d_2/split_1:output:39)model_1/up_sampling3d_2/split_1:output:40)model_1/up_sampling3d_2/split_1:output:40)model_1/up_sampling3d_2/split_1:output:41)model_1/up_sampling3d_2/split_1:output:41)model_1/up_sampling3d_2/split_1:output:42)model_1/up_sampling3d_2/split_1:output:42)model_1/up_sampling3d_2/split_1:output:43)model_1/up_sampling3d_2/split_1:output:43)model_1/up_sampling3d_2/split_1:output:44)model_1/up_sampling3d_2/split_1:output:44)model_1/up_sampling3d_2/split_1:output:45)model_1/up_sampling3d_2/split_1:output:45)model_1/up_sampling3d_2/split_1:output:46)model_1/up_sampling3d_2/split_1:output:46)model_1/up_sampling3d_2/split_1:output:47)model_1/up_sampling3d_2/split_1:output:47.model_1/up_sampling3d_2/concat_1/axis:output:0*
N`*
T0*3
_output_shapes!
:?????????``0@2"
 model_1/up_sampling3d_2/concat_1?
model_1/up_sampling3d_2/Const_2Const*
_output_shapes
: *
dtype0*
value	B :02!
model_1/up_sampling3d_2/Const_2?
)model_1/up_sampling3d_2/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)model_1/up_sampling3d_2/split_2/split_dim?
model_1/up_sampling3d_2/split_2Split2model_1/up_sampling3d_2/split_2/split_dim:output:0)model_1/up_sampling3d_2/concat_1:output:0*
T0*?
_output_shapes?
?:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@*
	num_split02!
model_1/up_sampling3d_2/split_2?
%model_1/up_sampling3d_2/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%model_1/up_sampling3d_2/concat_2/axis?!
 model_1/up_sampling3d_2/concat_2ConcatV2(model_1/up_sampling3d_2/split_2:output:0(model_1/up_sampling3d_2/split_2:output:0(model_1/up_sampling3d_2/split_2:output:1(model_1/up_sampling3d_2/split_2:output:1(model_1/up_sampling3d_2/split_2:output:2(model_1/up_sampling3d_2/split_2:output:2(model_1/up_sampling3d_2/split_2:output:3(model_1/up_sampling3d_2/split_2:output:3(model_1/up_sampling3d_2/split_2:output:4(model_1/up_sampling3d_2/split_2:output:4(model_1/up_sampling3d_2/split_2:output:5(model_1/up_sampling3d_2/split_2:output:5(model_1/up_sampling3d_2/split_2:output:6(model_1/up_sampling3d_2/split_2:output:6(model_1/up_sampling3d_2/split_2:output:7(model_1/up_sampling3d_2/split_2:output:7(model_1/up_sampling3d_2/split_2:output:8(model_1/up_sampling3d_2/split_2:output:8(model_1/up_sampling3d_2/split_2:output:9(model_1/up_sampling3d_2/split_2:output:9)model_1/up_sampling3d_2/split_2:output:10)model_1/up_sampling3d_2/split_2:output:10)model_1/up_sampling3d_2/split_2:output:11)model_1/up_sampling3d_2/split_2:output:11)model_1/up_sampling3d_2/split_2:output:12)model_1/up_sampling3d_2/split_2:output:12)model_1/up_sampling3d_2/split_2:output:13)model_1/up_sampling3d_2/split_2:output:13)model_1/up_sampling3d_2/split_2:output:14)model_1/up_sampling3d_2/split_2:output:14)model_1/up_sampling3d_2/split_2:output:15)model_1/up_sampling3d_2/split_2:output:15)model_1/up_sampling3d_2/split_2:output:16)model_1/up_sampling3d_2/split_2:output:16)model_1/up_sampling3d_2/split_2:output:17)model_1/up_sampling3d_2/split_2:output:17)model_1/up_sampling3d_2/split_2:output:18)model_1/up_sampling3d_2/split_2:output:18)model_1/up_sampling3d_2/split_2:output:19)model_1/up_sampling3d_2/split_2:output:19)model_1/up_sampling3d_2/split_2:output:20)model_1/up_sampling3d_2/split_2:output:20)model_1/up_sampling3d_2/split_2:output:21)model_1/up_sampling3d_2/split_2:output:21)model_1/up_sampling3d_2/split_2:output:22)model_1/up_sampling3d_2/split_2:output:22)model_1/up_sampling3d_2/split_2:output:23)model_1/up_sampling3d_2/split_2:output:23)model_1/up_sampling3d_2/split_2:output:24)model_1/up_sampling3d_2/split_2:output:24)model_1/up_sampling3d_2/split_2:output:25)model_1/up_sampling3d_2/split_2:output:25)model_1/up_sampling3d_2/split_2:output:26)model_1/up_sampling3d_2/split_2:output:26)model_1/up_sampling3d_2/split_2:output:27)model_1/up_sampling3d_2/split_2:output:27)model_1/up_sampling3d_2/split_2:output:28)model_1/up_sampling3d_2/split_2:output:28)model_1/up_sampling3d_2/split_2:output:29)model_1/up_sampling3d_2/split_2:output:29)model_1/up_sampling3d_2/split_2:output:30)model_1/up_sampling3d_2/split_2:output:30)model_1/up_sampling3d_2/split_2:output:31)model_1/up_sampling3d_2/split_2:output:31)model_1/up_sampling3d_2/split_2:output:32)model_1/up_sampling3d_2/split_2:output:32)model_1/up_sampling3d_2/split_2:output:33)model_1/up_sampling3d_2/split_2:output:33)model_1/up_sampling3d_2/split_2:output:34)model_1/up_sampling3d_2/split_2:output:34)model_1/up_sampling3d_2/split_2:output:35)model_1/up_sampling3d_2/split_2:output:35)model_1/up_sampling3d_2/split_2:output:36)model_1/up_sampling3d_2/split_2:output:36)model_1/up_sampling3d_2/split_2:output:37)model_1/up_sampling3d_2/split_2:output:37)model_1/up_sampling3d_2/split_2:output:38)model_1/up_sampling3d_2/split_2:output:38)model_1/up_sampling3d_2/split_2:output:39)model_1/up_sampling3d_2/split_2:output:39)model_1/up_sampling3d_2/split_2:output:40)model_1/up_sampling3d_2/split_2:output:40)model_1/up_sampling3d_2/split_2:output:41)model_1/up_sampling3d_2/split_2:output:41)model_1/up_sampling3d_2/split_2:output:42)model_1/up_sampling3d_2/split_2:output:42)model_1/up_sampling3d_2/split_2:output:43)model_1/up_sampling3d_2/split_2:output:43)model_1/up_sampling3d_2/split_2:output:44)model_1/up_sampling3d_2/split_2:output:44)model_1/up_sampling3d_2/split_2:output:45)model_1/up_sampling3d_2/split_2:output:45)model_1/up_sampling3d_2/split_2:output:46)model_1/up_sampling3d_2/split_2:output:46)model_1/up_sampling3d_2/split_2:output:47)model_1/up_sampling3d_2/split_2:output:47.model_1/up_sampling3d_2/concat_2/axis:output:0*
N`*
T0*3
_output_shapes!
:?????????```@2"
 model_1/up_sampling3d_2/concat_2?
!model_1/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_3/concat/axis?
model_1/concatenate_3/concatConcatV2)model_1/up_sampling3d_2/concat_2:output:0+model_1/leaky_re_lu/LeakyRelu:activations:0*model_1/concatenate_3/concat/axis:output:0*
N*
T0*3
_output_shapes!
:?????????````2
model_1/concatenate_3/concat?
&model_1/conv3d_7/Conv3D/ReadVariableOpReadVariableOp/model_1_conv3d_7_conv3d_readvariableop_resource**
_output_shapes
:`@*
dtype02(
&model_1/conv3d_7/Conv3D/ReadVariableOp?
model_1/conv3d_7/Conv3DConv3D%model_1/concatenate_3/concat:output:0.model_1/conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????```@*
paddingSAME*
strides	
2
model_1/conv3d_7/Conv3D?
'model_1/conv3d_7/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_1/conv3d_7/BiasAdd/ReadVariableOp?
model_1/conv3d_7/BiasAddBiasAdd model_1/conv3d_7/Conv3D:output:0/model_1/conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????```@2
model_1/conv3d_7/BiasAdd?
model_1/leaky_re_lu_7/LeakyRelu	LeakyRelu!model_1/conv3d_7/BiasAdd:output:0*3
_output_shapes!
:?????????```@2!
model_1/leaky_re_lu_7/LeakyRelu?
&model_1/conv3d_8/Conv3D/ReadVariableOpReadVariableOp/model_1_conv3d_8_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype02(
&model_1/conv3d_8/Conv3D/ReadVariableOp?
model_1/conv3d_8/Conv3DConv3D-model_1/leaky_re_lu_7/LeakyRelu:activations:0.model_1/conv3d_8/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????```@*
paddingSAME*
strides	
2
model_1/conv3d_8/Conv3D?
'model_1/conv3d_8/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv3d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_1/conv3d_8/BiasAdd/ReadVariableOp?
model_1/conv3d_8/BiasAddBiasAdd model_1/conv3d_8/Conv3D:output:0/model_1/conv3d_8/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????```@2
model_1/conv3d_8/BiasAdd?
model_1/leaky_re_lu_8/LeakyRelu	LeakyRelu!model_1/conv3d_8/BiasAdd:output:0*3
_output_shapes!
:?????????```@2!
model_1/leaky_re_lu_8/LeakyRelu?
model_1/up_sampling3d_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :`2
model_1/up_sampling3d_3/Const?
'model_1/up_sampling3d_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_1/up_sampling3d_3/split/split_dim?
model_1/up_sampling3d_3/splitSplit0model_1/up_sampling3d_3/split/split_dim:output:0-model_1/leaky_re_lu_8/LeakyRelu:activations:0*
T0*?
_output_shapes?
?:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@:?????????``@*
	num_split`2
model_1/up_sampling3d_3/split?
#model_1/up_sampling3d_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_1/up_sampling3d_3/concat/axis?>
model_1/up_sampling3d_3/concatConcatV2&model_1/up_sampling3d_3/split:output:0&model_1/up_sampling3d_3/split:output:0&model_1/up_sampling3d_3/split:output:1&model_1/up_sampling3d_3/split:output:1&model_1/up_sampling3d_3/split:output:2&model_1/up_sampling3d_3/split:output:2&model_1/up_sampling3d_3/split:output:3&model_1/up_sampling3d_3/split:output:3&model_1/up_sampling3d_3/split:output:4&model_1/up_sampling3d_3/split:output:4&model_1/up_sampling3d_3/split:output:5&model_1/up_sampling3d_3/split:output:5&model_1/up_sampling3d_3/split:output:6&model_1/up_sampling3d_3/split:output:6&model_1/up_sampling3d_3/split:output:7&model_1/up_sampling3d_3/split:output:7&model_1/up_sampling3d_3/split:output:8&model_1/up_sampling3d_3/split:output:8&model_1/up_sampling3d_3/split:output:9&model_1/up_sampling3d_3/split:output:9'model_1/up_sampling3d_3/split:output:10'model_1/up_sampling3d_3/split:output:10'model_1/up_sampling3d_3/split:output:11'model_1/up_sampling3d_3/split:output:11'model_1/up_sampling3d_3/split:output:12'model_1/up_sampling3d_3/split:output:12'model_1/up_sampling3d_3/split:output:13'model_1/up_sampling3d_3/split:output:13'model_1/up_sampling3d_3/split:output:14'model_1/up_sampling3d_3/split:output:14'model_1/up_sampling3d_3/split:output:15'model_1/up_sampling3d_3/split:output:15'model_1/up_sampling3d_3/split:output:16'model_1/up_sampling3d_3/split:output:16'model_1/up_sampling3d_3/split:output:17'model_1/up_sampling3d_3/split:output:17'model_1/up_sampling3d_3/split:output:18'model_1/up_sampling3d_3/split:output:18'model_1/up_sampling3d_3/split:output:19'model_1/up_sampling3d_3/split:output:19'model_1/up_sampling3d_3/split:output:20'model_1/up_sampling3d_3/split:output:20'model_1/up_sampling3d_3/split:output:21'model_1/up_sampling3d_3/split:output:21'model_1/up_sampling3d_3/split:output:22'model_1/up_sampling3d_3/split:output:22'model_1/up_sampling3d_3/split:output:23'model_1/up_sampling3d_3/split:output:23'model_1/up_sampling3d_3/split:output:24'model_1/up_sampling3d_3/split:output:24'model_1/up_sampling3d_3/split:output:25'model_1/up_sampling3d_3/split:output:25'model_1/up_sampling3d_3/split:output:26'model_1/up_sampling3d_3/split:output:26'model_1/up_sampling3d_3/split:output:27'model_1/up_sampling3d_3/split:output:27'model_1/up_sampling3d_3/split:output:28'model_1/up_sampling3d_3/split:output:28'model_1/up_sampling3d_3/split:output:29'model_1/up_sampling3d_3/split:output:29'model_1/up_sampling3d_3/split:output:30'model_1/up_sampling3d_3/split:output:30'model_1/up_sampling3d_3/split:output:31'model_1/up_sampling3d_3/split:output:31'model_1/up_sampling3d_3/split:output:32'model_1/up_sampling3d_3/split:output:32'model_1/up_sampling3d_3/split:output:33'model_1/up_sampling3d_3/split:output:33'model_1/up_sampling3d_3/split:output:34'model_1/up_sampling3d_3/split:output:34'model_1/up_sampling3d_3/split:output:35'model_1/up_sampling3d_3/split:output:35'model_1/up_sampling3d_3/split:output:36'model_1/up_sampling3d_3/split:output:36'model_1/up_sampling3d_3/split:output:37'model_1/up_sampling3d_3/split:output:37'model_1/up_sampling3d_3/split:output:38'model_1/up_sampling3d_3/split:output:38'model_1/up_sampling3d_3/split:output:39'model_1/up_sampling3d_3/split:output:39'model_1/up_sampling3d_3/split:output:40'model_1/up_sampling3d_3/split:output:40'model_1/up_sampling3d_3/split:output:41'model_1/up_sampling3d_3/split:output:41'model_1/up_sampling3d_3/split:output:42'model_1/up_sampling3d_3/split:output:42'model_1/up_sampling3d_3/split:output:43'model_1/up_sampling3d_3/split:output:43'model_1/up_sampling3d_3/split:output:44'model_1/up_sampling3d_3/split:output:44'model_1/up_sampling3d_3/split:output:45'model_1/up_sampling3d_3/split:output:45'model_1/up_sampling3d_3/split:output:46'model_1/up_sampling3d_3/split:output:46'model_1/up_sampling3d_3/split:output:47'model_1/up_sampling3d_3/split:output:47'model_1/up_sampling3d_3/split:output:48'model_1/up_sampling3d_3/split:output:48'model_1/up_sampling3d_3/split:output:49'model_1/up_sampling3d_3/split:output:49'model_1/up_sampling3d_3/split:output:50'model_1/up_sampling3d_3/split:output:50'model_1/up_sampling3d_3/split:output:51'model_1/up_sampling3d_3/split:output:51'model_1/up_sampling3d_3/split:output:52'model_1/up_sampling3d_3/split:output:52'model_1/up_sampling3d_3/split:output:53'model_1/up_sampling3d_3/split:output:53'model_1/up_sampling3d_3/split:output:54'model_1/up_sampling3d_3/split:output:54'model_1/up_sampling3d_3/split:output:55'model_1/up_sampling3d_3/split:output:55'model_1/up_sampling3d_3/split:output:56'model_1/up_sampling3d_3/split:output:56'model_1/up_sampling3d_3/split:output:57'model_1/up_sampling3d_3/split:output:57'model_1/up_sampling3d_3/split:output:58'model_1/up_sampling3d_3/split:output:58'model_1/up_sampling3d_3/split:output:59'model_1/up_sampling3d_3/split:output:59'model_1/up_sampling3d_3/split:output:60'model_1/up_sampling3d_3/split:output:60'model_1/up_sampling3d_3/split:output:61'model_1/up_sampling3d_3/split:output:61'model_1/up_sampling3d_3/split:output:62'model_1/up_sampling3d_3/split:output:62'model_1/up_sampling3d_3/split:output:63'model_1/up_sampling3d_3/split:output:63'model_1/up_sampling3d_3/split:output:64'model_1/up_sampling3d_3/split:output:64'model_1/up_sampling3d_3/split:output:65'model_1/up_sampling3d_3/split:output:65'model_1/up_sampling3d_3/split:output:66'model_1/up_sampling3d_3/split:output:66'model_1/up_sampling3d_3/split:output:67'model_1/up_sampling3d_3/split:output:67'model_1/up_sampling3d_3/split:output:68'model_1/up_sampling3d_3/split:output:68'model_1/up_sampling3d_3/split:output:69'model_1/up_sampling3d_3/split:output:69'model_1/up_sampling3d_3/split:output:70'model_1/up_sampling3d_3/split:output:70'model_1/up_sampling3d_3/split:output:71'model_1/up_sampling3d_3/split:output:71'model_1/up_sampling3d_3/split:output:72'model_1/up_sampling3d_3/split:output:72'model_1/up_sampling3d_3/split:output:73'model_1/up_sampling3d_3/split:output:73'model_1/up_sampling3d_3/split:output:74'model_1/up_sampling3d_3/split:output:74'model_1/up_sampling3d_3/split:output:75'model_1/up_sampling3d_3/split:output:75'model_1/up_sampling3d_3/split:output:76'model_1/up_sampling3d_3/split:output:76'model_1/up_sampling3d_3/split:output:77'model_1/up_sampling3d_3/split:output:77'model_1/up_sampling3d_3/split:output:78'model_1/up_sampling3d_3/split:output:78'model_1/up_sampling3d_3/split:output:79'model_1/up_sampling3d_3/split:output:79'model_1/up_sampling3d_3/split:output:80'model_1/up_sampling3d_3/split:output:80'model_1/up_sampling3d_3/split:output:81'model_1/up_sampling3d_3/split:output:81'model_1/up_sampling3d_3/split:output:82'model_1/up_sampling3d_3/split:output:82'model_1/up_sampling3d_3/split:output:83'model_1/up_sampling3d_3/split:output:83'model_1/up_sampling3d_3/split:output:84'model_1/up_sampling3d_3/split:output:84'model_1/up_sampling3d_3/split:output:85'model_1/up_sampling3d_3/split:output:85'model_1/up_sampling3d_3/split:output:86'model_1/up_sampling3d_3/split:output:86'model_1/up_sampling3d_3/split:output:87'model_1/up_sampling3d_3/split:output:87'model_1/up_sampling3d_3/split:output:88'model_1/up_sampling3d_3/split:output:88'model_1/up_sampling3d_3/split:output:89'model_1/up_sampling3d_3/split:output:89'model_1/up_sampling3d_3/split:output:90'model_1/up_sampling3d_3/split:output:90'model_1/up_sampling3d_3/split:output:91'model_1/up_sampling3d_3/split:output:91'model_1/up_sampling3d_3/split:output:92'model_1/up_sampling3d_3/split:output:92'model_1/up_sampling3d_3/split:output:93'model_1/up_sampling3d_3/split:output:93'model_1/up_sampling3d_3/split:output:94'model_1/up_sampling3d_3/split:output:94'model_1/up_sampling3d_3/split:output:95'model_1/up_sampling3d_3/split:output:95,model_1/up_sampling3d_3/concat/axis:output:0*
N?*
T0*4
_output_shapes"
 :??????????``@2 
model_1/up_sampling3d_3/concat?
model_1/up_sampling3d_3/Const_1Const*
_output_shapes
: *
dtype0*
value	B :`2!
model_1/up_sampling3d_3/Const_1?
)model_1/up_sampling3d_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)model_1/up_sampling3d_3/split_1/split_dim?
model_1/up_sampling3d_3/split_1Split2model_1/up_sampling3d_3/split_1/split_dim:output:0'model_1/up_sampling3d_3/concat:output:0*
T0*?
_output_shapes?
?:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@:??????????`@*
	num_split`2!
model_1/up_sampling3d_3/split_1?
%model_1/up_sampling3d_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%model_1/up_sampling3d_3/concat_1/axis?A
 model_1/up_sampling3d_3/concat_1ConcatV2(model_1/up_sampling3d_3/split_1:output:0(model_1/up_sampling3d_3/split_1:output:0(model_1/up_sampling3d_3/split_1:output:1(model_1/up_sampling3d_3/split_1:output:1(model_1/up_sampling3d_3/split_1:output:2(model_1/up_sampling3d_3/split_1:output:2(model_1/up_sampling3d_3/split_1:output:3(model_1/up_sampling3d_3/split_1:output:3(model_1/up_sampling3d_3/split_1:output:4(model_1/up_sampling3d_3/split_1:output:4(model_1/up_sampling3d_3/split_1:output:5(model_1/up_sampling3d_3/split_1:output:5(model_1/up_sampling3d_3/split_1:output:6(model_1/up_sampling3d_3/split_1:output:6(model_1/up_sampling3d_3/split_1:output:7(model_1/up_sampling3d_3/split_1:output:7(model_1/up_sampling3d_3/split_1:output:8(model_1/up_sampling3d_3/split_1:output:8(model_1/up_sampling3d_3/split_1:output:9(model_1/up_sampling3d_3/split_1:output:9)model_1/up_sampling3d_3/split_1:output:10)model_1/up_sampling3d_3/split_1:output:10)model_1/up_sampling3d_3/split_1:output:11)model_1/up_sampling3d_3/split_1:output:11)model_1/up_sampling3d_3/split_1:output:12)model_1/up_sampling3d_3/split_1:output:12)model_1/up_sampling3d_3/split_1:output:13)model_1/up_sampling3d_3/split_1:output:13)model_1/up_sampling3d_3/split_1:output:14)model_1/up_sampling3d_3/split_1:output:14)model_1/up_sampling3d_3/split_1:output:15)model_1/up_sampling3d_3/split_1:output:15)model_1/up_sampling3d_3/split_1:output:16)model_1/up_sampling3d_3/split_1:output:16)model_1/up_sampling3d_3/split_1:output:17)model_1/up_sampling3d_3/split_1:output:17)model_1/up_sampling3d_3/split_1:output:18)model_1/up_sampling3d_3/split_1:output:18)model_1/up_sampling3d_3/split_1:output:19)model_1/up_sampling3d_3/split_1:output:19)model_1/up_sampling3d_3/split_1:output:20)model_1/up_sampling3d_3/split_1:output:20)model_1/up_sampling3d_3/split_1:output:21)model_1/up_sampling3d_3/split_1:output:21)model_1/up_sampling3d_3/split_1:output:22)model_1/up_sampling3d_3/split_1:output:22)model_1/up_sampling3d_3/split_1:output:23)model_1/up_sampling3d_3/split_1:output:23)model_1/up_sampling3d_3/split_1:output:24)model_1/up_sampling3d_3/split_1:output:24)model_1/up_sampling3d_3/split_1:output:25)model_1/up_sampling3d_3/split_1:output:25)model_1/up_sampling3d_3/split_1:output:26)model_1/up_sampling3d_3/split_1:output:26)model_1/up_sampling3d_3/split_1:output:27)model_1/up_sampling3d_3/split_1:output:27)model_1/up_sampling3d_3/split_1:output:28)model_1/up_sampling3d_3/split_1:output:28)model_1/up_sampling3d_3/split_1:output:29)model_1/up_sampling3d_3/split_1:output:29)model_1/up_sampling3d_3/split_1:output:30)model_1/up_sampling3d_3/split_1:output:30)model_1/up_sampling3d_3/split_1:output:31)model_1/up_sampling3d_3/split_1:output:31)model_1/up_sampling3d_3/split_1:output:32)model_1/up_sampling3d_3/split_1:output:32)model_1/up_sampling3d_3/split_1:output:33)model_1/up_sampling3d_3/split_1:output:33)model_1/up_sampling3d_3/split_1:output:34)model_1/up_sampling3d_3/split_1:output:34)model_1/up_sampling3d_3/split_1:output:35)model_1/up_sampling3d_3/split_1:output:35)model_1/up_sampling3d_3/split_1:output:36)model_1/up_sampling3d_3/split_1:output:36)model_1/up_sampling3d_3/split_1:output:37)model_1/up_sampling3d_3/split_1:output:37)model_1/up_sampling3d_3/split_1:output:38)model_1/up_sampling3d_3/split_1:output:38)model_1/up_sampling3d_3/split_1:output:39)model_1/up_sampling3d_3/split_1:output:39)model_1/up_sampling3d_3/split_1:output:40)model_1/up_sampling3d_3/split_1:output:40)model_1/up_sampling3d_3/split_1:output:41)model_1/up_sampling3d_3/split_1:output:41)model_1/up_sampling3d_3/split_1:output:42)model_1/up_sampling3d_3/split_1:output:42)model_1/up_sampling3d_3/split_1:output:43)model_1/up_sampling3d_3/split_1:output:43)model_1/up_sampling3d_3/split_1:output:44)model_1/up_sampling3d_3/split_1:output:44)model_1/up_sampling3d_3/split_1:output:45)model_1/up_sampling3d_3/split_1:output:45)model_1/up_sampling3d_3/split_1:output:46)model_1/up_sampling3d_3/split_1:output:46)model_1/up_sampling3d_3/split_1:output:47)model_1/up_sampling3d_3/split_1:output:47)model_1/up_sampling3d_3/split_1:output:48)model_1/up_sampling3d_3/split_1:output:48)model_1/up_sampling3d_3/split_1:output:49)model_1/up_sampling3d_3/split_1:output:49)model_1/up_sampling3d_3/split_1:output:50)model_1/up_sampling3d_3/split_1:output:50)model_1/up_sampling3d_3/split_1:output:51)model_1/up_sampling3d_3/split_1:output:51)model_1/up_sampling3d_3/split_1:output:52)model_1/up_sampling3d_3/split_1:output:52)model_1/up_sampling3d_3/split_1:output:53)model_1/up_sampling3d_3/split_1:output:53)model_1/up_sampling3d_3/split_1:output:54)model_1/up_sampling3d_3/split_1:output:54)model_1/up_sampling3d_3/split_1:output:55)model_1/up_sampling3d_3/split_1:output:55)model_1/up_sampling3d_3/split_1:output:56)model_1/up_sampling3d_3/split_1:output:56)model_1/up_sampling3d_3/split_1:output:57)model_1/up_sampling3d_3/split_1:output:57)model_1/up_sampling3d_3/split_1:output:58)model_1/up_sampling3d_3/split_1:output:58)model_1/up_sampling3d_3/split_1:output:59)model_1/up_sampling3d_3/split_1:output:59)model_1/up_sampling3d_3/split_1:output:60)model_1/up_sampling3d_3/split_1:output:60)model_1/up_sampling3d_3/split_1:output:61)model_1/up_sampling3d_3/split_1:output:61)model_1/up_sampling3d_3/split_1:output:62)model_1/up_sampling3d_3/split_1:output:62)model_1/up_sampling3d_3/split_1:output:63)model_1/up_sampling3d_3/split_1:output:63)model_1/up_sampling3d_3/split_1:output:64)model_1/up_sampling3d_3/split_1:output:64)model_1/up_sampling3d_3/split_1:output:65)model_1/up_sampling3d_3/split_1:output:65)model_1/up_sampling3d_3/split_1:output:66)model_1/up_sampling3d_3/split_1:output:66)model_1/up_sampling3d_3/split_1:output:67)model_1/up_sampling3d_3/split_1:output:67)model_1/up_sampling3d_3/split_1:output:68)model_1/up_sampling3d_3/split_1:output:68)model_1/up_sampling3d_3/split_1:output:69)model_1/up_sampling3d_3/split_1:output:69)model_1/up_sampling3d_3/split_1:output:70)model_1/up_sampling3d_3/split_1:output:70)model_1/up_sampling3d_3/split_1:output:71)model_1/up_sampling3d_3/split_1:output:71)model_1/up_sampling3d_3/split_1:output:72)model_1/up_sampling3d_3/split_1:output:72)model_1/up_sampling3d_3/split_1:output:73)model_1/up_sampling3d_3/split_1:output:73)model_1/up_sampling3d_3/split_1:output:74)model_1/up_sampling3d_3/split_1:output:74)model_1/up_sampling3d_3/split_1:output:75)model_1/up_sampling3d_3/split_1:output:75)model_1/up_sampling3d_3/split_1:output:76)model_1/up_sampling3d_3/split_1:output:76)model_1/up_sampling3d_3/split_1:output:77)model_1/up_sampling3d_3/split_1:output:77)model_1/up_sampling3d_3/split_1:output:78)model_1/up_sampling3d_3/split_1:output:78)model_1/up_sampling3d_3/split_1:output:79)model_1/up_sampling3d_3/split_1:output:79)model_1/up_sampling3d_3/split_1:output:80)model_1/up_sampling3d_3/split_1:output:80)model_1/up_sampling3d_3/split_1:output:81)model_1/up_sampling3d_3/split_1:output:81)model_1/up_sampling3d_3/split_1:output:82)model_1/up_sampling3d_3/split_1:output:82)model_1/up_sampling3d_3/split_1:output:83)model_1/up_sampling3d_3/split_1:output:83)model_1/up_sampling3d_3/split_1:output:84)model_1/up_sampling3d_3/split_1:output:84)model_1/up_sampling3d_3/split_1:output:85)model_1/up_sampling3d_3/split_1:output:85)model_1/up_sampling3d_3/split_1:output:86)model_1/up_sampling3d_3/split_1:output:86)model_1/up_sampling3d_3/split_1:output:87)model_1/up_sampling3d_3/split_1:output:87)model_1/up_sampling3d_3/split_1:output:88)model_1/up_sampling3d_3/split_1:output:88)model_1/up_sampling3d_3/split_1:output:89)model_1/up_sampling3d_3/split_1:output:89)model_1/up_sampling3d_3/split_1:output:90)model_1/up_sampling3d_3/split_1:output:90)model_1/up_sampling3d_3/split_1:output:91)model_1/up_sampling3d_3/split_1:output:91)model_1/up_sampling3d_3/split_1:output:92)model_1/up_sampling3d_3/split_1:output:92)model_1/up_sampling3d_3/split_1:output:93)model_1/up_sampling3d_3/split_1:output:93)model_1/up_sampling3d_3/split_1:output:94)model_1/up_sampling3d_3/split_1:output:94)model_1/up_sampling3d_3/split_1:output:95)model_1/up_sampling3d_3/split_1:output:95.model_1/up_sampling3d_3/concat_1/axis:output:0*
N?*
T0*5
_output_shapes#
!:???????????`@2"
 model_1/up_sampling3d_3/concat_1?
model_1/up_sampling3d_3/Const_2Const*
_output_shapes
: *
dtype0*
value	B :`2!
model_1/up_sampling3d_3/Const_2?
)model_1/up_sampling3d_3/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)model_1/up_sampling3d_3/split_2/split_dim?
model_1/up_sampling3d_3/split_2Split2model_1/up_sampling3d_3/split_2/split_dim:output:0)model_1/up_sampling3d_3/concat_1:output:0*
T0*?
_output_shapes?
?:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@:???????????@*
	num_split`2!
model_1/up_sampling3d_3/split_2?
%model_1/up_sampling3d_3/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%model_1/up_sampling3d_3/concat_2/axis?A
 model_1/up_sampling3d_3/concat_2ConcatV2(model_1/up_sampling3d_3/split_2:output:0(model_1/up_sampling3d_3/split_2:output:0(model_1/up_sampling3d_3/split_2:output:1(model_1/up_sampling3d_3/split_2:output:1(model_1/up_sampling3d_3/split_2:output:2(model_1/up_sampling3d_3/split_2:output:2(model_1/up_sampling3d_3/split_2:output:3(model_1/up_sampling3d_3/split_2:output:3(model_1/up_sampling3d_3/split_2:output:4(model_1/up_sampling3d_3/split_2:output:4(model_1/up_sampling3d_3/split_2:output:5(model_1/up_sampling3d_3/split_2:output:5(model_1/up_sampling3d_3/split_2:output:6(model_1/up_sampling3d_3/split_2:output:6(model_1/up_sampling3d_3/split_2:output:7(model_1/up_sampling3d_3/split_2:output:7(model_1/up_sampling3d_3/split_2:output:8(model_1/up_sampling3d_3/split_2:output:8(model_1/up_sampling3d_3/split_2:output:9(model_1/up_sampling3d_3/split_2:output:9)model_1/up_sampling3d_3/split_2:output:10)model_1/up_sampling3d_3/split_2:output:10)model_1/up_sampling3d_3/split_2:output:11)model_1/up_sampling3d_3/split_2:output:11)model_1/up_sampling3d_3/split_2:output:12)model_1/up_sampling3d_3/split_2:output:12)model_1/up_sampling3d_3/split_2:output:13)model_1/up_sampling3d_3/split_2:output:13)model_1/up_sampling3d_3/split_2:output:14)model_1/up_sampling3d_3/split_2:output:14)model_1/up_sampling3d_3/split_2:output:15)model_1/up_sampling3d_3/split_2:output:15)model_1/up_sampling3d_3/split_2:output:16)model_1/up_sampling3d_3/split_2:output:16)model_1/up_sampling3d_3/split_2:output:17)model_1/up_sampling3d_3/split_2:output:17)model_1/up_sampling3d_3/split_2:output:18)model_1/up_sampling3d_3/split_2:output:18)model_1/up_sampling3d_3/split_2:output:19)model_1/up_sampling3d_3/split_2:output:19)model_1/up_sampling3d_3/split_2:output:20)model_1/up_sampling3d_3/split_2:output:20)model_1/up_sampling3d_3/split_2:output:21)model_1/up_sampling3d_3/split_2:output:21)model_1/up_sampling3d_3/split_2:output:22)model_1/up_sampling3d_3/split_2:output:22)model_1/up_sampling3d_3/split_2:output:23)model_1/up_sampling3d_3/split_2:output:23)model_1/up_sampling3d_3/split_2:output:24)model_1/up_sampling3d_3/split_2:output:24)model_1/up_sampling3d_3/split_2:output:25)model_1/up_sampling3d_3/split_2:output:25)model_1/up_sampling3d_3/split_2:output:26)model_1/up_sampling3d_3/split_2:output:26)model_1/up_sampling3d_3/split_2:output:27)model_1/up_sampling3d_3/split_2:output:27)model_1/up_sampling3d_3/split_2:output:28)model_1/up_sampling3d_3/split_2:output:28)model_1/up_sampling3d_3/split_2:output:29)model_1/up_sampling3d_3/split_2:output:29)model_1/up_sampling3d_3/split_2:output:30)model_1/up_sampling3d_3/split_2:output:30)model_1/up_sampling3d_3/split_2:output:31)model_1/up_sampling3d_3/split_2:output:31)model_1/up_sampling3d_3/split_2:output:32)model_1/up_sampling3d_3/split_2:output:32)model_1/up_sampling3d_3/split_2:output:33)model_1/up_sampling3d_3/split_2:output:33)model_1/up_sampling3d_3/split_2:output:34)model_1/up_sampling3d_3/split_2:output:34)model_1/up_sampling3d_3/split_2:output:35)model_1/up_sampling3d_3/split_2:output:35)model_1/up_sampling3d_3/split_2:output:36)model_1/up_sampling3d_3/split_2:output:36)model_1/up_sampling3d_3/split_2:output:37)model_1/up_sampling3d_3/split_2:output:37)model_1/up_sampling3d_3/split_2:output:38)model_1/up_sampling3d_3/split_2:output:38)model_1/up_sampling3d_3/split_2:output:39)model_1/up_sampling3d_3/split_2:output:39)model_1/up_sampling3d_3/split_2:output:40)model_1/up_sampling3d_3/split_2:output:40)model_1/up_sampling3d_3/split_2:output:41)model_1/up_sampling3d_3/split_2:output:41)model_1/up_sampling3d_3/split_2:output:42)model_1/up_sampling3d_3/split_2:output:42)model_1/up_sampling3d_3/split_2:output:43)model_1/up_sampling3d_3/split_2:output:43)model_1/up_sampling3d_3/split_2:output:44)model_1/up_sampling3d_3/split_2:output:44)model_1/up_sampling3d_3/split_2:output:45)model_1/up_sampling3d_3/split_2:output:45)model_1/up_sampling3d_3/split_2:output:46)model_1/up_sampling3d_3/split_2:output:46)model_1/up_sampling3d_3/split_2:output:47)model_1/up_sampling3d_3/split_2:output:47)model_1/up_sampling3d_3/split_2:output:48)model_1/up_sampling3d_3/split_2:output:48)model_1/up_sampling3d_3/split_2:output:49)model_1/up_sampling3d_3/split_2:output:49)model_1/up_sampling3d_3/split_2:output:50)model_1/up_sampling3d_3/split_2:output:50)model_1/up_sampling3d_3/split_2:output:51)model_1/up_sampling3d_3/split_2:output:51)model_1/up_sampling3d_3/split_2:output:52)model_1/up_sampling3d_3/split_2:output:52)model_1/up_sampling3d_3/split_2:output:53)model_1/up_sampling3d_3/split_2:output:53)model_1/up_sampling3d_3/split_2:output:54)model_1/up_sampling3d_3/split_2:output:54)model_1/up_sampling3d_3/split_2:output:55)model_1/up_sampling3d_3/split_2:output:55)model_1/up_sampling3d_3/split_2:output:56)model_1/up_sampling3d_3/split_2:output:56)model_1/up_sampling3d_3/split_2:output:57)model_1/up_sampling3d_3/split_2:output:57)model_1/up_sampling3d_3/split_2:output:58)model_1/up_sampling3d_3/split_2:output:58)model_1/up_sampling3d_3/split_2:output:59)model_1/up_sampling3d_3/split_2:output:59)model_1/up_sampling3d_3/split_2:output:60)model_1/up_sampling3d_3/split_2:output:60)model_1/up_sampling3d_3/split_2:output:61)model_1/up_sampling3d_3/split_2:output:61)model_1/up_sampling3d_3/split_2:output:62)model_1/up_sampling3d_3/split_2:output:62)model_1/up_sampling3d_3/split_2:output:63)model_1/up_sampling3d_3/split_2:output:63)model_1/up_sampling3d_3/split_2:output:64)model_1/up_sampling3d_3/split_2:output:64)model_1/up_sampling3d_3/split_2:output:65)model_1/up_sampling3d_3/split_2:output:65)model_1/up_sampling3d_3/split_2:output:66)model_1/up_sampling3d_3/split_2:output:66)model_1/up_sampling3d_3/split_2:output:67)model_1/up_sampling3d_3/split_2:output:67)model_1/up_sampling3d_3/split_2:output:68)model_1/up_sampling3d_3/split_2:output:68)model_1/up_sampling3d_3/split_2:output:69)model_1/up_sampling3d_3/split_2:output:69)model_1/up_sampling3d_3/split_2:output:70)model_1/up_sampling3d_3/split_2:output:70)model_1/up_sampling3d_3/split_2:output:71)model_1/up_sampling3d_3/split_2:output:71)model_1/up_sampling3d_3/split_2:output:72)model_1/up_sampling3d_3/split_2:output:72)model_1/up_sampling3d_3/split_2:output:73)model_1/up_sampling3d_3/split_2:output:73)model_1/up_sampling3d_3/split_2:output:74)model_1/up_sampling3d_3/split_2:output:74)model_1/up_sampling3d_3/split_2:output:75)model_1/up_sampling3d_3/split_2:output:75)model_1/up_sampling3d_3/split_2:output:76)model_1/up_sampling3d_3/split_2:output:76)model_1/up_sampling3d_3/split_2:output:77)model_1/up_sampling3d_3/split_2:output:77)model_1/up_sampling3d_3/split_2:output:78)model_1/up_sampling3d_3/split_2:output:78)model_1/up_sampling3d_3/split_2:output:79)model_1/up_sampling3d_3/split_2:output:79)model_1/up_sampling3d_3/split_2:output:80)model_1/up_sampling3d_3/split_2:output:80)model_1/up_sampling3d_3/split_2:output:81)model_1/up_sampling3d_3/split_2:output:81)model_1/up_sampling3d_3/split_2:output:82)model_1/up_sampling3d_3/split_2:output:82)model_1/up_sampling3d_3/split_2:output:83)model_1/up_sampling3d_3/split_2:output:83)model_1/up_sampling3d_3/split_2:output:84)model_1/up_sampling3d_3/split_2:output:84)model_1/up_sampling3d_3/split_2:output:85)model_1/up_sampling3d_3/split_2:output:85)model_1/up_sampling3d_3/split_2:output:86)model_1/up_sampling3d_3/split_2:output:86)model_1/up_sampling3d_3/split_2:output:87)model_1/up_sampling3d_3/split_2:output:87)model_1/up_sampling3d_3/split_2:output:88)model_1/up_sampling3d_3/split_2:output:88)model_1/up_sampling3d_3/split_2:output:89)model_1/up_sampling3d_3/split_2:output:89)model_1/up_sampling3d_3/split_2:output:90)model_1/up_sampling3d_3/split_2:output:90)model_1/up_sampling3d_3/split_2:output:91)model_1/up_sampling3d_3/split_2:output:91)model_1/up_sampling3d_3/split_2:output:92)model_1/up_sampling3d_3/split_2:output:92)model_1/up_sampling3d_3/split_2:output:93)model_1/up_sampling3d_3/split_2:output:93)model_1/up_sampling3d_3/split_2:output:94)model_1/up_sampling3d_3/split_2:output:94)model_1/up_sampling3d_3/split_2:output:95)model_1/up_sampling3d_3/split_2:output:95.model_1/up_sampling3d_3/concat_2/axis:output:0*
N?*
T0*6
_output_shapes$
": ????????????@2"
 model_1/up_sampling3d_3/concat_2?
!model_1/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_4/concat/axis?
model_1/concatenate_4/concatConcatV2)model_1/up_sampling3d_3/concat_2:output:0#model_1/concatenate/concat:output:0*model_1/concatenate_4/concat/axis:output:0*
N*
T0*6
_output_shapes$
": ????????????B2
model_1/concatenate_4/concat?
&model_1/conv3d_9/Conv3D/ReadVariableOpReadVariableOp/model_1_conv3d_9_conv3d_readvariableop_resource**
_output_shapes
:B *
dtype02(
&model_1/conv3d_9/Conv3D/ReadVariableOp?
model_1/conv3d_9/Conv3DConv3D%model_1/concatenate_4/concat:output:0.model_1/conv3d_9/Conv3D/ReadVariableOp:value:0*
T0*6
_output_shapes$
": ???????????? *
paddingSAME*
strides	
2
model_1/conv3d_9/Conv3D?
'model_1/conv3d_9/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv3d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_1/conv3d_9/BiasAdd/ReadVariableOp?
model_1/conv3d_9/BiasAddBiasAdd model_1/conv3d_9/Conv3D:output:0/model_1/conv3d_9/BiasAdd/ReadVariableOp:value:0*
T0*6
_output_shapes$
": ???????????? 2
model_1/conv3d_9/BiasAdd?
model_1/leaky_re_lu_9/LeakyRelu	LeakyRelu!model_1/conv3d_9/BiasAdd:output:0*6
_output_shapes$
": ???????????? 2!
model_1/leaky_re_lu_9/LeakyRelu?
'model_1/conv3d_10/Conv3D/ReadVariableOpReadVariableOp0model_1_conv3d_10_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype02)
'model_1/conv3d_10/Conv3D/ReadVariableOp?
model_1/conv3d_10/Conv3DConv3D-model_1/leaky_re_lu_9/LeakyRelu:activations:0/model_1/conv3d_10/Conv3D/ReadVariableOp:value:0*
T0*6
_output_shapes$
": ???????????? *
paddingSAME*
strides	
2
model_1/conv3d_10/Conv3D?
(model_1/conv3d_10/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv3d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_1/conv3d_10/BiasAdd/ReadVariableOp?
model_1/conv3d_10/BiasAddBiasAdd!model_1/conv3d_10/Conv3D:output:00model_1/conv3d_10/BiasAdd/ReadVariableOp:value:0*
T0*6
_output_shapes$
": ???????????? 2
model_1/conv3d_10/BiasAdd?
 model_1/leaky_re_lu_10/LeakyRelu	LeakyRelu"model_1/conv3d_10/BiasAdd:output:0*6
_output_shapes$
": ???????????? 2"
 model_1/leaky_re_lu_10/LeakyRelu?
"model_1/disp/Conv3D/ReadVariableOpReadVariableOp+model_1_disp_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02$
"model_1/disp/Conv3D/ReadVariableOp?
model_1/disp/Conv3DConv3D.model_1/leaky_re_lu_10/LeakyRelu:activations:0*model_1/disp/Conv3D/ReadVariableOp:value:0*
T0*6
_output_shapes$
": ????????????*
paddingSAME*
strides	
2
model_1/disp/Conv3D?
#model_1/disp/BiasAdd/ReadVariableOpReadVariableOp,model_1_disp_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model_1/disp/BiasAdd/ReadVariableOp?
model_1/disp/BiasAddBiasAddmodel_1/disp/Conv3D:output:0+model_1/disp/BiasAdd/ReadVariableOp:value:0*
T0*6
_output_shapes$
": ????????????2
model_1/disp/BiasAdd?
!model_1/transformer/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"?????   ?   ?      2#
!model_1/transformer/Reshape/shape?
model_1/transformer/ReshapeReshapeinput_1*model_1/transformer/Reshape/shape:output:0*
T0*6
_output_shapes$
": ????????????2
model_1/transformer/Reshape?
#model_1/transformer/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"?????   ?   ?      2%
#model_1/transformer/Reshape_1/shape?
model_1/transformer/Reshape_1Reshapemodel_1/disp/BiasAdd:output:0,model_1/transformer/Reshape_1/shape:output:0*
T0*6
_output_shapes$
": ????????????2
model_1/transformer/Reshape_1?
model_1/transformer/map/ShapeShape$model_1/transformer/Reshape:output:0*
T0*
_output_shapes
:2
model_1/transformer/map/Shape?
+model_1/transformer/map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+model_1/transformer/map/strided_slice/stack?
-model_1/transformer/map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_1/transformer/map/strided_slice/stack_1?
-model_1/transformer/map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_1/transformer/map/strided_slice/stack_2?
%model_1/transformer/map/strided_sliceStridedSlice&model_1/transformer/map/Shape:output:04model_1/transformer/map/strided_slice/stack:output:06model_1/transformer/map/strided_slice/stack_1:output:06model_1/transformer/map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%model_1/transformer/map/strided_slice?
3model_1/transformer/map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????25
3model_1/transformer/map/TensorArrayV2/element_shape?
%model_1/transformer/map/TensorArrayV2TensorListReserve<model_1/transformer/map/TensorArrayV2/element_shape:output:0.model_1/transformer/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%model_1/transformer/map/TensorArrayV2?
5model_1/transformer/map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????27
5model_1/transformer/map/TensorArrayV2_1/element_shape?
'model_1/transformer/map/TensorArrayV2_1TensorListReserve>model_1/transformer/map/TensorArrayV2_1/element_shape:output:0.model_1/transformer/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'model_1/transformer/map/TensorArrayV2_1?
Mmodel_1/transformer/map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      2O
Mmodel_1/transformer/map/TensorArrayUnstack/TensorListFromTensor/element_shape?
?model_1/transformer/map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$model_1/transformer/Reshape:output:0Vmodel_1/transformer/map/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?model_1/transformer/map/TensorArrayUnstack/TensorListFromTensor?
Omodel_1/transformer/map/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      2Q
Omodel_1/transformer/map/TensorArrayUnstack_1/TensorListFromTensor/element_shape?
Amodel_1/transformer/map/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor&model_1/transformer/Reshape_1:output:0Xmodel_1/transformer/map/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02C
Amodel_1/transformer/map/TensorArrayUnstack_1/TensorListFromTensor?
model_1/transformer/map/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
model_1/transformer/map/Const?
5model_1/transformer/map/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????27
5model_1/transformer/map/TensorArrayV2_2/element_shape?
'model_1/transformer/map/TensorArrayV2_2TensorListReserve>model_1/transformer/map/TensorArrayV2_2/element_shape:output:0.model_1/transformer/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'model_1/transformer/map/TensorArrayV2_2?
*model_1/transformer/map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model_1/transformer/map/while/loop_counter?
model_1/transformer/map/whileStatelessWhile3model_1/transformer/map/while/loop_counter:output:0.model_1/transformer/map/strided_slice:output:0&model_1/transformer/map/Const:output:00model_1/transformer/map/TensorArrayV2_2:handle:0.model_1/transformer/map/strided_slice:output:0Omodel_1/transformer/map/TensorArrayUnstack/TensorListFromTensor:output_handle:0Qmodel_1/transformer/map/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : * 
_read_only_resource_inputs
 *5
body-R+
)model_1_transformer_map_while_body_272091*5
cond-R+
)model_1_transformer_map_while_cond_272090*!
output_shapes
: : : : : : : 2
model_1/transformer/map/while?
Hmodel_1/transformer/map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"?   ?   ?      2J
Hmodel_1/transformer/map/TensorArrayV2Stack/TensorListStack/element_shape?
:model_1/transformer/map/TensorArrayV2Stack/TensorListStackTensorListStack&model_1/transformer/map/while:output:3Qmodel_1/transformer/map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*6
_output_shapes$
": ????????????*
element_dtype02<
:model_1/transformer/map/TensorArrayV2Stack/TensorListStack?
IdentityIdentitymodel_1/disp/BiasAdd:output:0*
T0*6
_output_shapes$
": ????????????2

Identity?

Identity_1IdentityCmodel_1/transformer/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*6
_output_shapes$
": ????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ????????????: ????????????:::::::::::::::::::::::::_ [
6
_output_shapes$
": ????????????
!
_user_specified_name	input_1:_[
6
_output_shapes$
": ????????????
!
_user_specified_name	input_2:
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
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
q
G__inference_concatenate_layer_call_and_return_conditional_losses_272658

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*6
_output_shapes$
": ????????????2
concatr
IdentityIdentityconcat:output:0*
T0*6
_output_shapes$
": ????????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D: ????????????: ????????????:^ Z
6
_output_shapes$
": ????????????
 
_user_specified_nameinputs:^Z
6
_output_shapes$
": ????????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_272713

inputs
identity`
	LeakyRelu	LeakyReluinputs*3
_output_shapes!
:?????????@2
	LeakyReluw
IdentityIdentityLeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
??
?
C__inference_model_1_layer_call_and_return_conditional_losses_274229

inputs
inputs_1
conv3d_274147
conv3d_274149
conv3d_1_274153
conv3d_1_274155
conv3d_2_274159
conv3d_2_274161
conv3d_3_274165
conv3d_3_274167
conv3d_4_274171
conv3d_4_274173
conv3d_5_274179
conv3d_5_274181
conv3d_6_274187
conv3d_6_274189
conv3d_7_274195
conv3d_7_274197
conv3d_8_274201
conv3d_8_274203
conv3d_9_274209
conv3d_9_274211
conv3d_10_274215
conv3d_10_274217
disp_274221
disp_274223
identity

identity_1??conv3d/StatefulPartitionedCall? conv3d_1/StatefulPartitionedCall?!conv3d_10/StatefulPartitionedCall? conv3d_2/StatefulPartitionedCall? conv3d_3/StatefulPartitionedCall? conv3d_4/StatefulPartitionedCall? conv3d_5/StatefulPartitionedCall? conv3d_6/StatefulPartitionedCall? conv3d_7/StatefulPartitionedCall? conv3d_8/StatefulPartitionedCall? conv3d_9/StatefulPartitionedCall?disp/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2726582
concatenate/PartitionedCall?
conv3d/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv3d_274147conv3d_274149*
Tin
2*
Tout
2*3
_output_shapes!
:?????????``` *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_2724052 
conv3d/StatefulPartitionedCall?
leaky_re_lu/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????``` * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_2726772
leaky_re_lu/PartitionedCall?
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0conv3d_1_274153conv3d_1_274155*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_2724262"
 conv3d_1/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_2726952
leaky_re_lu_1/PartitionedCall?
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0conv3d_2_274159conv3d_2_274161*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_2724472"
 conv3d_2/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_2727132
leaky_re_lu_2/PartitionedCall?
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0conv3d_3_274165conv3d_3_274167*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_2724682"
 conv3d_3/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_2727312
leaky_re_lu_3/PartitionedCall?
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0conv3d_4_274171conv3d_4_274173*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_2724892"
 conv3d_4/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_2727492
leaky_re_lu_4/PartitionedCall?
up_sampling3d/PartitionedCallPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_up_sampling3d_layer_call_and_return_conditional_losses_2728092
up_sampling3d/PartitionedCall?
concatenate_1/PartitionedCallPartitionedCall&up_sampling3d/PartitionedCall:output:0&leaky_re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*4
_output_shapes"
 :??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_2728242
concatenate_1/PartitionedCall?
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv3d_5_274179conv3d_5_274181*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_2725102"
 conv3d_5/StatefulPartitionedCall?
leaky_re_lu_5/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_2728432
leaky_re_lu_5/PartitionedCall?
up_sampling3d_1/PartitionedCallPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_2729392!
up_sampling3d_1/PartitionedCall?
concatenate_2/PartitionedCallPartitionedCall(up_sampling3d_1/PartitionedCall:output:0&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*4
_output_shapes"
 :?????????000?* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_2729542
concatenate_2/PartitionedCall?
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv3d_6_274187conv3d_6_274189*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_6_layer_call_and_return_conditional_losses_2725312"
 conv3d_6/StatefulPartitionedCall?
leaky_re_lu_6/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????000@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_2729732
leaky_re_lu_6/PartitionedCall?
up_sampling3d_2/PartitionedCallPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_2731412!
up_sampling3d_2/PartitionedCall?
concatenate_3/PartitionedCallPartitionedCall(up_sampling3d_2/PartitionedCall:output:0$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????````* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_2731562
concatenate_3/PartitionedCall?
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv3d_7_274195conv3d_7_274197*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_7_layer_call_and_return_conditional_losses_2725522"
 conv3d_7/StatefulPartitionedCall?
leaky_re_lu_7/PartitionedCallPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_2731752
leaky_re_lu_7/PartitionedCall?
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0conv3d_8_274201conv3d_8_274203*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_8_layer_call_and_return_conditional_losses_2725732"
 conv3d_8/StatefulPartitionedCall?
leaky_re_lu_8/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????```@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_2731932
leaky_re_lu_8/PartitionedCall?
up_sampling3d_3/PartitionedCallPartitionedCall&leaky_re_lu_8/PartitionedCall:output:0*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_2735052!
up_sampling3d_3/PartitionedCall?
concatenate_4/PartitionedCallPartitionedCall(up_sampling3d_3/PartitionedCall:output:0$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????B* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_2735202
concatenate_4/PartitionedCall?
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv3d_9_274209conv3d_9_274211*
Tin
2*
Tout
2*6
_output_shapes$
": ???????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv3d_9_layer_call_and_return_conditional_losses_2725942"
 conv3d_9/StatefulPartitionedCall?
leaky_re_lu_9/PartitionedCallPartitionedCall)conv3d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*6
_output_shapes$
": ???????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_2735392
leaky_re_lu_9/PartitionedCall?
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_9/PartitionedCall:output:0conv3d_10_274215conv3d_10_274217*
Tin
2*
Tout
2*6
_output_shapes$
": ???????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_conv3d_10_layer_call_and_return_conditional_losses_2726152#
!conv3d_10/StatefulPartitionedCall?
leaky_re_lu_10/PartitionedCallPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*6
_output_shapes$
": ???????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_2735572 
leaky_re_lu_10/PartitionedCall?
disp/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_10/PartitionedCall:output:0disp_274221disp_274223*
Tin
2*
Tout
2*6
_output_shapes$
": ????????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_disp_layer_call_and_return_conditional_losses_2726362
disp/StatefulPartitionedCall?
transformer/PartitionedCallPartitionedCallinputs%disp/StatefulPartitionedCall:output:0*
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
GPU 

CPU2J 8*P
fKRI
G__inference_transformer_layer_call_and_return_conditional_losses_2738972
transformer/PartitionedCall?
IdentityIdentity$transformer/PartitionedCall:output:0^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall^disp/StatefulPartitionedCall*
T0*6
_output_shapes$
": ????????????2

Identity?

Identity_1Identity%disp/StatefulPartitionedCall:output:0^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall^disp/StatefulPartitionedCall*
T0*6
_output_shapes$
": ????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ????????????: ????????????::::::::::::::::::::::::2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall2<
disp/StatefulPartitionedCalldisp/StatefulPartitionedCall:^ Z
6
_output_shapes$
": ????????????
 
_user_specified_nameinputs:^Z
6
_output_shapes$
": ????????????
 
_user_specified_nameinputs:
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
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
!transformer_map_while_body_276055&
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
?5
g
K__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_272939

inputs
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0inputs*
T0*?
_output_shapes?
?:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@:?????????@*
	num_split2
split\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2split:output:0split:output:0split:output:1split:output:1split:output:2split:output:2split:output:3split:output:3split:output:4split:output:4split:output:5split:output:5split:output:6split:output:6split:output:7split:output:7split:output:8split:output:8split:output:9split:output:9split:output:10split:output:10split:output:11split:output:11split:output:12split:output:12split:output:13split:output:13split:output:14split:output:14split:output:15split:output:15split:output:16split:output:16split:output:17split:output:17split:output:18split:output:18split:output:19split:output:19split:output:20split:output:20split:output:21split:output:21split:output:22split:output:22split:output:23split:output:23concat/axis:output:0*
N0*
T0*3
_output_shapes!
:?????????0@2
concatT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*?
_output_shapes?
?:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@:?????????0@*
	num_split2	
split_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?
concat_1ConcatV2split_1:output:0split_1:output:0split_1:output:1split_1:output:1split_1:output:2split_1:output:2split_1:output:3split_1:output:3split_1:output:4split_1:output:4split_1:output:5split_1:output:5split_1:output:6split_1:output:6split_1:output:7split_1:output:7split_1:output:8split_1:output:8split_1:output:9split_1:output:9split_1:output:10split_1:output:10split_1:output:11split_1:output:11split_1:output:12split_1:output:12split_1:output:13split_1:output:13split_1:output:14split_1:output:14split_1:output:15split_1:output:15split_1:output:16split_1:output:16split_1:output:17split_1:output:17split_1:output:18split_1:output:18split_1:output:19split_1:output:19split_1:output:20split_1:output:20split_1:output:21split_1:output:21split_1:output:22split_1:output:22split_1:output:23split_1:output:23concat_1/axis:output:0*
N0*
T0*3
_output_shapes!
:?????????00@2

concat_1T
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*?
_output_shapes?
?:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@:?????????00@*
	num_split2	
split_2`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis?
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15split_2:output:16split_2:output:16split_2:output:17split_2:output:17split_2:output:18split_2:output:18split_2:output:19split_2:output:19split_2:output:20split_2:output:20split_2:output:21split_2:output:21split_2:output:22split_2:output:22split_2:output:23split_2:output:23concat_2/axis:output:0*
N0*
T0*3
_output_shapes!
:?????????000@2

concat_2q
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:?????????000@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????@:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_276498

inputs
identity`
	LeakyRelu	LeakyReluinputs*3
_output_shapes!
:?????????000@2
	LeakyReluw
IdentityIdentityLeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????000@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????000@:[ W
3
_output_shapes!
:?????????000@
 
_user_specified_nameinputs
?

?
B__inference_conv3d_layer_call_and_return_conditional_losses_272405

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8???????????????????????????????????? *
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8????????????????????????????????????:::v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?

?
D__inference_conv3d_6_layer_call_and_return_conditional_losses_272531

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:?@*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????@*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:9?????????????????????????????????????:::w s
O
_output_shapes=
;:9?????????????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?'
s
G__inference_transformer_layer_call_and_return_conditional_losses_277596
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
 *!
bodyR
map_while_body_277294*!
condR
map_while_cond_277293*!
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
?

?
D__inference_conv3d_5_layer_call_and_return_conditional_losses_272510

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:?@*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????@*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:9?????????????????????????????????????:::w s
O
_output_shapes=
;:9?????????????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
J
input_1?
serving_default_input_1:0 ????????????
J
input_2?
serving_default_input_2:0 ????????????G
disp?
StatefulPartitionedCall:0 ????????????N
transformer?
StatefulPartitionedCall:1 ????????????tensorflow/serving/predict:??
??
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer_with_weights-4
layer-11
layer-12
layer-13
layer-14
layer_with_weights-5
layer-15
layer-16
layer-17
layer-18
layer_with_weights-6
layer-19
layer-20
layer-21
layer-22
layer_with_weights-7
layer-23
layer-24
layer_with_weights-8
layer-25
layer-26
layer-27
layer-28
layer_with_weights-9
layer-29
layer-30
 layer_with_weights-10
 layer-31
!layer-32
"layer_with_weights-11
"layer-33
#layer-34
$trainable_variables
%	variables
&regularization_losses
'	keras_api
(
signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"??
_tf_keras_model??{"class_name": "Model", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 192, 192, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 192, 192, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["input_1", 0, 0, {}], ["input_2", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu", "inbound_nodes": [[["conv3d", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_1", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_1", "inbound_nodes": [[["conv3d_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_2", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_2", "inbound_nodes": [[["conv3d_2", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_3", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_3", "inbound_nodes": [[["conv3d_3", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_4", "inbound_nodes": [[["leaky_re_lu_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_4", "inbound_nodes": [[["conv3d_4", 0, 0, {}]]]}, {"class_name": "UpSampling3D", "config": {"name": "up_sampling3d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "name": "up_sampling3d", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["up_sampling3d", 0, 0, {}], ["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_5", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_5", "inbound_nodes": [[["conv3d_5", 0, 0, {}]]]}, {"class_name": "UpSampling3D", "config": {"name": "up_sampling3d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "name": "up_sampling3d_1", "inbound_nodes": [[["leaky_re_lu_5", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_2", "inbound_nodes": [[["up_sampling3d_1", 0, 0, {}], ["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_6", "inbound_nodes": [[["concatenate_2", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_6", "inbound_nodes": [[["conv3d_6", 0, 0, {}]]]}, {"class_name": "UpSampling3D", "config": {"name": "up_sampling3d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "name": "up_sampling3d_2", "inbound_nodes": [[["leaky_re_lu_6", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_3", "inbound_nodes": [[["up_sampling3d_2", 0, 0, {}], ["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_7", "inbound_nodes": [[["concatenate_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_7", "inbound_nodes": [[["conv3d_7", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_8", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_8", "inbound_nodes": [[["leaky_re_lu_7", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_8", "inbound_nodes": [[["conv3d_8", 0, 0, {}]]]}, {"class_name": "UpSampling3D", "config": {"name": "up_sampling3d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "name": "up_sampling3d_3", "inbound_nodes": [[["leaky_re_lu_8", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_4", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_4", "inbound_nodes": [[["up_sampling3d_3", 0, 0, {}], ["concatenate", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_9", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_9", "inbound_nodes": [[["concatenate_4", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_9", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_9", "inbound_nodes": [[["conv3d_9", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_10", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_10", "inbound_nodes": [[["leaky_re_lu_9", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_10", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_10", "inbound_nodes": [[["conv3d_10", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "disp", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "disp", "inbound_nodes": [[["leaky_re_lu_10", 0, 0, {}]]]}, {"class_name": "SpatialTransformer", "config": {"name": "transformer", "trainable": true, "dtype": "float32", "interp_method": "linear", "ndims": 3, "inshape": [{"class_name": "TensorShape", "items": [null, 192, 192, 192, 1]}, {"class_name": "TensorShape", "items": [null, 192, 192, 192, 3]}], "single_transform": false, "indexing": "ij"}, "name": "transformer", "inbound_nodes": [[["input_1", 0, 0, {}], ["disp", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["transformer", 0, 0], ["disp", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 192, 192, 192, 1]}, {"class_name": "TensorShape", "items": [null, 192, 192, 192, 1]}], "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 192, 192, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 192, 192, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["input_1", 0, 0, {}], ["input_2", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu", "inbound_nodes": [[["conv3d", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_1", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_1", "inbound_nodes": [[["conv3d_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_2", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_2", "inbound_nodes": [[["conv3d_2", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_3", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_3", "inbound_nodes": [[["conv3d_3", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_4", "inbound_nodes": [[["leaky_re_lu_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_4", "inbound_nodes": [[["conv3d_4", 0, 0, {}]]]}, {"class_name": "UpSampling3D", "config": {"name": "up_sampling3d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "name": "up_sampling3d", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["up_sampling3d", 0, 0, {}], ["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_5", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_5", "inbound_nodes": [[["conv3d_5", 0, 0, {}]]]}, {"class_name": "UpSampling3D", "config": {"name": "up_sampling3d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "name": "up_sampling3d_1", "inbound_nodes": [[["leaky_re_lu_5", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_2", "inbound_nodes": [[["up_sampling3d_1", 0, 0, {}], ["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_6", "inbound_nodes": [[["concatenate_2", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_6", "inbound_nodes": [[["conv3d_6", 0, 0, {}]]]}, {"class_name": "UpSampling3D", "config": {"name": "up_sampling3d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "name": "up_sampling3d_2", "inbound_nodes": [[["leaky_re_lu_6", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_3", "inbound_nodes": [[["up_sampling3d_2", 0, 0, {}], ["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_7", "inbound_nodes": [[["concatenate_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_7", "inbound_nodes": [[["conv3d_7", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_8", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_8", "inbound_nodes": [[["leaky_re_lu_7", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_8", "inbound_nodes": [[["conv3d_8", 0, 0, {}]]]}, {"class_name": "UpSampling3D", "config": {"name": "up_sampling3d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "name": "up_sampling3d_3", "inbound_nodes": [[["leaky_re_lu_8", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_4", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_4", "inbound_nodes": [[["up_sampling3d_3", 0, 0, {}], ["concatenate", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_9", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_9", "inbound_nodes": [[["concatenate_4", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_9", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_9", "inbound_nodes": [[["conv3d_9", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_10", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_10", "inbound_nodes": [[["leaky_re_lu_9", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_10", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_10", "inbound_nodes": [[["conv3d_10", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "disp", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "disp", "inbound_nodes": [[["leaky_re_lu_10", 0, 0, {}]]]}, {"class_name": "SpatialTransformer", "config": {"name": "transformer", "trainable": true, "dtype": "float32", "interp_method": "linear", "ndims": 3, "inshape": [{"class_name": "TensorShape", "items": [null, 192, 192, 192, 1]}, {"class_name": "TensorShape", "items": [null, 192, 192, 192, 3]}], "single_transform": false, "indexing": "ij"}, "name": "transformer", "inbound_nodes": [[["input_1", 0, 0, {}], ["disp", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["transformer", 0, 0], ["disp", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 192, 192, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 192, 192, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 192, 192, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 192, 192, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?
)trainable_variables
*	variables
+regularization_losses
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 192, 192, 192, 1]}, {"class_name": "TensorShape", "items": [null, 192, 192, 192, 1]}]}
?


-kernel
.bias
/trainable_variables
0	variables
1regularization_losses
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv3D", "name": "conv3d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv3d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 192, 192, 192, 2]}}
?
3trainable_variables
4	variables
5regularization_losses
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?


7kernel
8bias
9trainable_variables
:	variables
;regularization_losses
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv3D", "name": "conv3d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv3d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 96, 96, 32]}}
?
=trainable_variables
>	variables
?regularization_losses
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?


Akernel
Bbias
Ctrainable_variables
D	variables
Eregularization_losses
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv3D", "name": "conv3d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv3d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 48, 64]}}
?
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?


Kkernel
Lbias
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv3D", "name": "conv3d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv3d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 24, 64]}}
?
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?


Ukernel
Vbias
Wtrainable_variables
X	variables
Yregularization_losses
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv3D", "name": "conv3d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv3d_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 12, 64]}}
?
[trainable_variables
\	variables
]regularization_losses
^	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?
_trainable_variables
`	variables
aregularization_losses
b	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling3D", "name": "up_sampling3d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "up_sampling3d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}}
?
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 24, 24, 24, 64]}, {"class_name": "TensorShape", "items": [null, 24, 24, 24, 64]}]}
?


gkernel
hbias
itrainable_variables
j	variables
kregularization_losses
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv3D", "name": "conv3d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv3d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 24, 128]}}
?
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?
qtrainable_variables
r	variables
sregularization_losses
t	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling3D", "name": "up_sampling3d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "up_sampling3d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}}
?
utrainable_variables
v	variables
wregularization_losses
x	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 48, 48, 48, 64]}, {"class_name": "TensorShape", "items": [null, 48, 48, 48, 64]}]}
?


ykernel
zbias
{trainable_variables
|	variables
}regularization_losses
~	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv3D", "name": "conv3d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv3d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 48, 128]}}
?
trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling3D", "name": "up_sampling3d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "up_sampling3d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 96, 96, 96, 64]}, {"class_name": "TensorShape", "items": [null, 96, 96, 96, 32]}]}
?

?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv3D", "name": "conv3d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv3d_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 96}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 96, 96, 96]}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?

?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv3D", "name": "conv3d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv3d_8", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 96, 96, 64]}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling3D", "name": "up_sampling3d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "up_sampling3d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "concatenate_4", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 192, 192, 192, 64]}, {"class_name": "TensorShape", "items": [null, 192, 192, 192, 2]}]}
?

?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv3D", "name": "conv3d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv3d_9", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 66}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 192, 192, 192, 66]}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "leaky_re_lu_9", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?

?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv3D", "name": "conv3d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv3d_10", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 192, 192, 192, 32]}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "leaky_re_lu_10", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?	
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv3D", "name": "disp", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "disp", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 192, 192, 192, 32]}}
?
?inshape
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "SpatialTransformer", "name": "transformer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "transformer", "trainable": true, "dtype": "float32", "interp_method": "linear", "ndims": 3, "inshape": [{"class_name": "TensorShape", "items": [null, 192, 192, 192, 1]}, {"class_name": "TensorShape", "items": [null, 192, 192, 192, 3]}], "single_transform": false, "indexing": "ij"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 192, 192, 192, 1]}, {"class_name": "TensorShape", "items": [null, 192, 192, 192, 3]}]}
?
-0
.1
72
83
A4
B5
K6
L7
U8
V9
g10
h11
y12
z13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23"
trackable_list_wrapper
?
-0
.1
72
83
A4
B5
K6
L7
U8
V9
g10
h11
y12
z13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
$trainable_variables
?layer_metrics
?metrics
?layers
%	variables
?non_trainable_variables
&regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
)trainable_variables
?layer_metrics
?metrics
?layers
*	variables
?non_trainable_variables
+regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:) 2conv3d/kernel
: 2conv3d/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
/trainable_variables
?layer_metrics
?metrics
?layers
0	variables
?non_trainable_variables
1regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
3trainable_variables
?layer_metrics
?metrics
?layers
4	variables
?non_trainable_variables
5regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:. @2conv3d_1_13/kernel
:@2conv3d_1_13/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
9trainable_variables
?layer_metrics
?metrics
?layers
:	variables
?non_trainable_variables
;regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
=trainable_variables
?layer_metrics
?metrics
?layers
>	variables
?non_trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.@@2conv3d_2_13/kernel
:@2conv3d_2_13/bias
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
Ctrainable_variables
?layer_metrics
?metrics
?layers
D	variables
?non_trainable_variables
Eregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
Gtrainable_variables
?layer_metrics
?metrics
?layers
H	variables
?non_trainable_variables
Iregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.@@2conv3d_3_13/kernel
:@2conv3d_3_13/bias
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
Mtrainable_variables
?layer_metrics
?metrics
?layers
N	variables
?non_trainable_variables
Oregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
Qtrainable_variables
?layer_metrics
?metrics
?layers
R	variables
?non_trainable_variables
Sregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.@@2conv3d_4_13/kernel
:@2conv3d_4_13/bias
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
Wtrainable_variables
?layer_metrics
?metrics
?layers
X	variables
?non_trainable_variables
Yregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
[trainable_variables
?layer_metrics
?metrics
?layers
\	variables
?non_trainable_variables
]regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
_trainable_variables
?layer_metrics
?metrics
?layers
`	variables
?non_trainable_variables
aregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
ctrainable_variables
?layer_metrics
?metrics
?layers
d	variables
?non_trainable_variables
eregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/?@2conv3d_5_13/kernel
:@2conv3d_5_13/bias
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
itrainable_variables
?layer_metrics
?metrics
?layers
j	variables
?non_trainable_variables
kregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
mtrainable_variables
?layer_metrics
?metrics
?layers
n	variables
?non_trainable_variables
oregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
qtrainable_variables
?layer_metrics
?metrics
?layers
r	variables
?non_trainable_variables
sregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
utrainable_variables
?layer_metrics
?metrics
?layers
v	variables
?non_trainable_variables
wregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/?@2conv3d_6_13/kernel
:@2conv3d_6_13/bias
.
y0
z1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
{trainable_variables
?layer_metrics
?metrics
?layers
|	variables
?non_trainable_variables
}regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.`@2conv3d_7_13/kernel
:@2conv3d_7_13/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.@@2conv3d_8_13/kernel
:@2conv3d_8_13/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.B 2conv3d_9_13/kernel
: 2conv3d_9_13/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/  2conv3d_10_13/kernel
: 2conv3d_10_13/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:* 2disp_10/kernel
:2disp_10/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?metrics
?layers
?	variables
?non_trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34"
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
(__inference_model_1_layer_call_fn_274139
(__inference_model_1_layer_call_fn_276470
(__inference_model_1_layer_call_fn_276414
(__inference_model_1_layer_call_fn_274282?
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
!__inference__wrapped_model_272394?
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
input_1 ????????????
0?-
input_2 ????????????
?2?
C__inference_model_1_layer_call_and_return_conditional_losses_276358
C__inference_model_1_layer_call_and_return_conditional_losses_275349
C__inference_model_1_layer_call_and_return_conditional_losses_273908
C__inference_model_1_layer_call_and_return_conditional_losses_273995?
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
,__inference_concatenate_layer_call_fn_276483?
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
G__inference_concatenate_layer_call_and_return_conditional_losses_276477?
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
?2?
'__inference_conv3d_layer_call_fn_272415?
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
annotations? *D?A
??<8????????????????????????????????????
?2?
B__inference_conv3d_layer_call_and_return_conditional_losses_272405?
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
annotations? *D?A
??<8????????????????????????????????????
?2?
,__inference_leaky_re_lu_layer_call_fn_276493?
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
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_276488?
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
?2?
)__inference_conv3d_1_layer_call_fn_272436?
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
annotations? *D?A
??<8???????????????????????????????????? 
?2?
D__inference_conv3d_1_layer_call_and_return_conditional_losses_272426?
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
annotations? *D?A
??<8???????????????????????????????????? 
?2?
.__inference_leaky_re_lu_1_layer_call_fn_276503?
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
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_276498?
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
?2?
)__inference_conv3d_2_layer_call_fn_272457?
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
annotations? *D?A
??<8????????????????????????????????????@
?2?
D__inference_conv3d_2_layer_call_and_return_conditional_losses_272447?
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
annotations? *D?A
??<8????????????????????????????????????@
?2?
.__inference_leaky_re_lu_2_layer_call_fn_276513?
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
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_276508?
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
?2?
)__inference_conv3d_3_layer_call_fn_272478?
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
annotations? *D?A
??<8????????????????????????????????????@
?2?
D__inference_conv3d_3_layer_call_and_return_conditional_losses_272468?
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
annotations? *D?A
??<8????????????????????????????????????@
?2?
.__inference_leaky_re_lu_3_layer_call_fn_276523?
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
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_276518?
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
?2?
)__inference_conv3d_4_layer_call_fn_272499?
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
annotations? *D?A
??<8????????????????????????????????????@
?2?
D__inference_conv3d_4_layer_call_and_return_conditional_losses_272489?
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
annotations? *D?A
??<8????????????????????????????????????@
?2?
.__inference_leaky_re_lu_4_layer_call_fn_276533?
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
I__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_276528?
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
.__inference_up_sampling3d_layer_call_fn_276590?
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
I__inference_up_sampling3d_layer_call_and_return_conditional_losses_276585?
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
.__inference_concatenate_1_layer_call_fn_276603?
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
I__inference_concatenate_1_layer_call_and_return_conditional_losses_276597?
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
?2?
)__inference_conv3d_5_layer_call_fn_272520?
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
annotations? *E?B
@?=9?????????????????????????????????????
?2?
D__inference_conv3d_5_layer_call_and_return_conditional_losses_272510?
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
annotations? *E?B
@?=9?????????????????????????????????????
?2?
.__inference_leaky_re_lu_5_layer_call_fn_276613?
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
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_276608?
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
0__inference_up_sampling3d_1_layer_call_fn_276706?
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
K__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_276701?
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
.__inference_concatenate_2_layer_call_fn_276719?
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
I__inference_concatenate_2_layer_call_and_return_conditional_losses_276713?
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
?2?
)__inference_conv3d_6_layer_call_fn_272541?
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
annotations? *E?B
@?=9?????????????????????????????????????
?2?
D__inference_conv3d_6_layer_call_and_return_conditional_losses_272531?
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
annotations? *E?B
@?=9?????????????????????????????????????
?2?
.__inference_leaky_re_lu_6_layer_call_fn_276729?
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
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_276724?
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
0__inference_up_sampling3d_2_layer_call_fn_276894?
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
K__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_276889?
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
.__inference_concatenate_3_layer_call_fn_276907?
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
I__inference_concatenate_3_layer_call_and_return_conditional_losses_276901?
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
?2?
)__inference_conv3d_7_layer_call_fn_272562?
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
annotations? *D?A
??<8????????????????????????????????????`
?2?
D__inference_conv3d_7_layer_call_and_return_conditional_losses_272552?
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
annotations? *D?A
??<8????????????????????????????????????`
?2?
.__inference_leaky_re_lu_7_layer_call_fn_276917?
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
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_276912?
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
?2?
)__inference_conv3d_8_layer_call_fn_272583?
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
annotations? *D?A
??<8????????????????????????????????????@
?2?
D__inference_conv3d_8_layer_call_and_return_conditional_losses_272573?
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
annotations? *D?A
??<8????????????????????????????????????@
?2?
.__inference_leaky_re_lu_8_layer_call_fn_276927?
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
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_276922?
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
0__inference_up_sampling3d_3_layer_call_fn_277236?
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
K__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_277231?
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
.__inference_concatenate_4_layer_call_fn_277249?
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
I__inference_concatenate_4_layer_call_and_return_conditional_losses_277243?
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
?2?
)__inference_conv3d_9_layer_call_fn_272604?
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
annotations? *D?A
??<8????????????????????????????????????B
?2?
D__inference_conv3d_9_layer_call_and_return_conditional_losses_272594?
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
annotations? *D?A
??<8????????????????????????????????????B
?2?
.__inference_leaky_re_lu_9_layer_call_fn_277259?
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
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_277254?
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
?2?
*__inference_conv3d_10_layer_call_fn_272625?
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
annotations? *D?A
??<8???????????????????????????????????? 
?2?
E__inference_conv3d_10_layer_call_and_return_conditional_losses_272615?
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
annotations? *D?A
??<8???????????????????????????????????? 
?2?
/__inference_leaky_re_lu_10_layer_call_fn_277269?
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
J__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_277264?
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
?2?
%__inference_disp_layer_call_fn_272646?
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
annotations? *D?A
??<8???????????????????????????????????? 
?2?
@__inference_disp_layer_call_and_return_conditional_losses_272636?
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
annotations? *D?A
??<8???????????????????????????????????? 
?2?
,__inference_transformer_layer_call_fn_277602?
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
G__inference_transformer_layer_call_and_return_conditional_losses_277596?
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
:B8
$__inference_signature_wrapper_274340input_1input_2?
!__inference__wrapped_model_272394?"-.78ABKLUVghyz??????????v?s
l?i
g?d
0?-
input_1 ????????????
0?-
input_2 ????????????
? "?|
5
disp-?*
disp ????????????
C
transformer4?1
transformer ?????????????
I__inference_concatenate_1_layer_call_and_return_conditional_losses_276597?r?o
h?e
c?`
.?+
inputs/0?????????@
.?+
inputs/1?????????@
? "2?/
(?%
0??????????
? ?
.__inference_concatenate_1_layer_call_fn_276603?r?o
h?e
c?`
.?+
inputs/0?????????@
.?+
inputs/1?????????@
? "%?"???????????
I__inference_concatenate_2_layer_call_and_return_conditional_losses_276713?r?o
h?e
c?`
.?+
inputs/0?????????000@
.?+
inputs/1?????????000@
? "2?/
(?%
0?????????000?
? ?
.__inference_concatenate_2_layer_call_fn_276719?r?o
h?e
c?`
.?+
inputs/0?????????000@
.?+
inputs/1?????????000@
? "%?"?????????000??
I__inference_concatenate_3_layer_call_and_return_conditional_losses_276901?r?o
h?e
c?`
.?+
inputs/0?????????```@
.?+
inputs/1?????????``` 
? "1?.
'?$
0?????????````
? ?
.__inference_concatenate_3_layer_call_fn_276907?r?o
h?e
c?`
.?+
inputs/0?????????```@
.?+
inputs/1?????????``` 
? "$?!?????????````?
I__inference_concatenate_4_layer_call_and_return_conditional_losses_277243?x?u
n?k
i?f
1?.
inputs/0 ????????????@
1?.
inputs/1 ????????????
? "4?1
*?'
0 ????????????B
? ?
.__inference_concatenate_4_layer_call_fn_277249?x?u
n?k
i?f
1?.
inputs/0 ????????????@
1?.
inputs/1 ????????????
? "'?$ ????????????B?
G__inference_concatenate_layer_call_and_return_conditional_losses_276477?x?u
n?k
i?f
1?.
inputs/0 ????????????
1?.
inputs/1 ????????????
? "4?1
*?'
0 ????????????
? ?
,__inference_concatenate_layer_call_fn_276483?x?u
n?k
i?f
1?.
inputs/0 ????????????
1?.
inputs/1 ????????????
? "'?$ ?????????????
E__inference_conv3d_10_layer_call_and_return_conditional_losses_272615???V?S
L?I
G?D
inputs8???????????????????????????????????? 
? "L?I
B??
08???????????????????????????????????? 
? ?
*__inference_conv3d_10_layer_call_fn_272625???V?S
L?I
G?D
inputs8???????????????????????????????????? 
? "??<8???????????????????????????????????? ?
D__inference_conv3d_1_layer_call_and_return_conditional_losses_272426?78V?S
L?I
G?D
inputs8???????????????????????????????????? 
? "L?I
B??
08????????????????????????????????????@
? ?
)__inference_conv3d_1_layer_call_fn_272436?78V?S
L?I
G?D
inputs8???????????????????????????????????? 
? "??<8????????????????????????????????????@?
D__inference_conv3d_2_layer_call_and_return_conditional_losses_272447?ABV?S
L?I
G?D
inputs8????????????????????????????????????@
? "L?I
B??
08????????????????????????????????????@
? ?
)__inference_conv3d_2_layer_call_fn_272457?ABV?S
L?I
G?D
inputs8????????????????????????????????????@
? "??<8????????????????????????????????????@?
D__inference_conv3d_3_layer_call_and_return_conditional_losses_272468?KLV?S
L?I
G?D
inputs8????????????????????????????????????@
? "L?I
B??
08????????????????????????????????????@
? ?
)__inference_conv3d_3_layer_call_fn_272478?KLV?S
L?I
G?D
inputs8????????????????????????????????????@
? "??<8????????????????????????????????????@?
D__inference_conv3d_4_layer_call_and_return_conditional_losses_272489?UVV?S
L?I
G?D
inputs8????????????????????????????????????@
? "L?I
B??
08????????????????????????????????????@
? ?
)__inference_conv3d_4_layer_call_fn_272499?UVV?S
L?I
G?D
inputs8????????????????????????????????????@
? "??<8????????????????????????????????????@?
D__inference_conv3d_5_layer_call_and_return_conditional_losses_272510?ghW?T
M?J
H?E
inputs9?????????????????????????????????????
? "L?I
B??
08????????????????????????????????????@
? ?
)__inference_conv3d_5_layer_call_fn_272520?ghW?T
M?J
H?E
inputs9?????????????????????????????????????
? "??<8????????????????????????????????????@?
D__inference_conv3d_6_layer_call_and_return_conditional_losses_272531?yzW?T
M?J
H?E
inputs9?????????????????????????????????????
? "L?I
B??
08????????????????????????????????????@
? ?
)__inference_conv3d_6_layer_call_fn_272541?yzW?T
M?J
H?E
inputs9?????????????????????????????????????
? "??<8????????????????????????????????????@?
D__inference_conv3d_7_layer_call_and_return_conditional_losses_272552???V?S
L?I
G?D
inputs8????????????????????????????????????`
? "L?I
B??
08????????????????????????????????????@
? ?
)__inference_conv3d_7_layer_call_fn_272562???V?S
L?I
G?D
inputs8????????????????????????????????????`
? "??<8????????????????????????????????????@?
D__inference_conv3d_8_layer_call_and_return_conditional_losses_272573???V?S
L?I
G?D
inputs8????????????????????????????????????@
? "L?I
B??
08????????????????????????????????????@
? ?
)__inference_conv3d_8_layer_call_fn_272583???V?S
L?I
G?D
inputs8????????????????????????????????????@
? "??<8????????????????????????????????????@?
D__inference_conv3d_9_layer_call_and_return_conditional_losses_272594???V?S
L?I
G?D
inputs8????????????????????????????????????B
? "L?I
B??
08???????????????????????????????????? 
? ?
)__inference_conv3d_9_layer_call_fn_272604???V?S
L?I
G?D
inputs8????????????????????????????????????B
? "??<8???????????????????????????????????? ?
B__inference_conv3d_layer_call_and_return_conditional_losses_272405?-.V?S
L?I
G?D
inputs8????????????????????????????????????
? "L?I
B??
08???????????????????????????????????? 
? ?
'__inference_conv3d_layer_call_fn_272415?-.V?S
L?I
G?D
inputs8????????????????????????????????????
? "??<8???????????????????????????????????? ?
@__inference_disp_layer_call_and_return_conditional_losses_272636???V?S
L?I
G?D
inputs8???????????????????????????????????? 
? "L?I
B??
08????????????????????????????????????
? ?
%__inference_disp_layer_call_fn_272646???V?S
L?I
G?D
inputs8???????????????????????????????????? 
? "??<8?????????????????????????????????????
J__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_277264v>?;
4?1
/?,
inputs ???????????? 
? "4?1
*?'
0 ???????????? 
? ?
/__inference_leaky_re_lu_10_layer_call_fn_277269i>?;
4?1
/?,
inputs ???????????? 
? "'?$ ???????????? ?
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_276498p;?8
1?.
,?)
inputs?????????000@
? "1?.
'?$
0?????????000@
? ?
.__inference_leaky_re_lu_1_layer_call_fn_276503c;?8
1?.
,?)
inputs?????????000@
? "$?!?????????000@?
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_276508p;?8
1?.
,?)
inputs?????????@
? "1?.
'?$
0?????????@
? ?
.__inference_leaky_re_lu_2_layer_call_fn_276513c;?8
1?.
,?)
inputs?????????@
? "$?!?????????@?
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_276518p;?8
1?.
,?)
inputs?????????@
? "1?.
'?$
0?????????@
? ?
.__inference_leaky_re_lu_3_layer_call_fn_276523c;?8
1?.
,?)
inputs?????????@
? "$?!?????????@?
I__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_276528p;?8
1?.
,?)
inputs?????????@
? "1?.
'?$
0?????????@
? ?
.__inference_leaky_re_lu_4_layer_call_fn_276533c;?8
1?.
,?)
inputs?????????@
? "$?!?????????@?
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_276608p;?8
1?.
,?)
inputs?????????@
? "1?.
'?$
0?????????@
? ?
.__inference_leaky_re_lu_5_layer_call_fn_276613c;?8
1?.
,?)
inputs?????????@
? "$?!?????????@?
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_276724p;?8
1?.
,?)
inputs?????????000@
? "1?.
'?$
0?????????000@
? ?
.__inference_leaky_re_lu_6_layer_call_fn_276729c;?8
1?.
,?)
inputs?????????000@
? "$?!?????????000@?
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_276912p;?8
1?.
,?)
inputs?????????```@
? "1?.
'?$
0?????????```@
? ?
.__inference_leaky_re_lu_7_layer_call_fn_276917c;?8
1?.
,?)
inputs?????????```@
? "$?!?????????```@?
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_276922p;?8
1?.
,?)
inputs?????????```@
? "1?.
'?$
0?????????```@
? ?
.__inference_leaky_re_lu_8_layer_call_fn_276927c;?8
1?.
,?)
inputs?????????```@
? "$?!?????????```@?
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_277254v>?;
4?1
/?,
inputs ???????????? 
? "4?1
*?'
0 ???????????? 
? ?
.__inference_leaky_re_lu_9_layer_call_fn_277259i>?;
4?1
/?,
inputs ???????????? 
? "'?$ ???????????? ?
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_276488p;?8
1?.
,?)
inputs?????????``` 
? "1?.
'?$
0?????????``` 
? ?
,__inference_leaky_re_lu_layer_call_fn_276493c;?8
1?.
,?)
inputs?????????``` 
? "$?!?????????``` ?
C__inference_model_1_layer_call_and_return_conditional_losses_273908?"-.78ABKLUVghyz??????????~?{
t?q
g?d
0?-
input_1 ????????????
0?-
input_2 ????????????
p

 
? "i?f
_?\
,?)
0/0 ????????????
,?)
0/1 ????????????
? ?
C__inference_model_1_layer_call_and_return_conditional_losses_273995?"-.78ABKLUVghyz??????????~?{
t?q
g?d
0?-
input_1 ????????????
0?-
input_2 ????????????
p 

 
? "i?f
_?\
,?)
0/0 ????????????
,?)
0/1 ????????????
? ?
C__inference_model_1_layer_call_and_return_conditional_losses_275349?"-.78ABKLUVghyz????????????}
v?s
i?f
1?.
inputs/0 ????????????
1?.
inputs/1 ????????????
p

 
? "i?f
_?\
,?)
0/0 ????????????
,?)
0/1 ????????????
? ?
C__inference_model_1_layer_call_and_return_conditional_losses_276358?"-.78ABKLUVghyz????????????}
v?s
i?f
1?.
inputs/0 ????????????
1?.
inputs/1 ????????????
p 

 
? "i?f
_?\
,?)
0/0 ????????????
,?)
0/1 ????????????
? ?
(__inference_model_1_layer_call_fn_274139?"-.78ABKLUVghyz??????????~?{
t?q
g?d
0?-
input_1 ????????????
0?-
input_2 ????????????
p

 
? "[?X
*?'
0 ????????????
*?'
1 ?????????????
(__inference_model_1_layer_call_fn_274282?"-.78ABKLUVghyz??????????~?{
t?q
g?d
0?-
input_1 ????????????
0?-
input_2 ????????????
p 

 
? "[?X
*?'
0 ????????????
*?'
1 ?????????????
(__inference_model_1_layer_call_fn_276414?"-.78ABKLUVghyz????????????}
v?s
i?f
1?.
inputs/0 ????????????
1?.
inputs/1 ????????????
p

 
? "[?X
*?'
0 ????????????
*?'
1 ?????????????
(__inference_model_1_layer_call_fn_276470?"-.78ABKLUVghyz????????????}
v?s
i?f
1?.
inputs/0 ????????????
1?.
inputs/1 ????????????
p 

 
? "[?X
*?'
0 ????????????
*?'
1 ?????????????
$__inference_signature_wrapper_274340?"-.78ABKLUVghyz?????????????
? 
}?z
;
input_10?-
input_1 ????????????
;
input_20?-
input_2 ????????????"?|
5
disp-?*
disp ????????????
C
transformer4?1
transformer ?????????????
G__inference_transformer_layer_call_and_return_conditional_losses_277596?x?u
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
,__inference_transformer_layer_call_fn_277602?x?u
n?k
i?f
1?.
inputs/0 ????????????
1?.
inputs/1 ????????????
? "'?$ ?????????????
K__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_276701p;?8
1?.
,?)
inputs?????????@
? "1?.
'?$
0?????????000@
? ?
0__inference_up_sampling3d_1_layer_call_fn_276706c;?8
1?.
,?)
inputs?????????@
? "$?!?????????000@?
K__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_276889p;?8
1?.
,?)
inputs?????????000@
? "1?.
'?$
0?????????```@
? ?
0__inference_up_sampling3d_2_layer_call_fn_276894c;?8
1?.
,?)
inputs?????????000@
? "$?!?????????```@?
K__inference_up_sampling3d_3_layer_call_and_return_conditional_losses_277231s;?8
1?.
,?)
inputs?????????```@
? "4?1
*?'
0 ????????????@
? ?
0__inference_up_sampling3d_3_layer_call_fn_277236f;?8
1?.
,?)
inputs?????????```@
? "'?$ ????????????@?
I__inference_up_sampling3d_layer_call_and_return_conditional_losses_276585p;?8
1?.
,?)
inputs?????????@
? "1?.
'?$
0?????????@
? ?
.__inference_up_sampling3d_layer_call_fn_276590c;?8
1?.
,?)
inputs?????????@
? "$?!?????????@