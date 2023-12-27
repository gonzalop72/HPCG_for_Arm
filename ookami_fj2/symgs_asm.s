.L2413:					// :entr:term
	.loc 38 114 0 is_stmt 1
..LDL81:
sxtw	x20, w2
	.loc 38 116 0
..LDL82:
sxtw	x3, w2
	.loc 38 114 0
..LDL83:
whilelo	p1.d, x20, x7
..LDL84:
whilelo	p0.d, x20, x30
	.loc 38 128 0
..LDL85:
add	x2, x20, x5
	.loc 38 117 0
..LDL86:
add	x19, x13, x3, lsl #2
	.loc 38 128 0
..LDL87:
cmp	w2, w29
	.loc 38 124 0
..LDL88:
add	x20, x17, x3, lsl #2
	.loc 38 117 0
..LDL89:
ld1sw	{z5.d}, p1/z, [x19, 0, mul vl]	//  (*)
	.loc 38 116 0
..LDL90:
add	x19, x14, x3, lsl #3
	.loc 38 124 0
..LDL91:
ld1sw	{z4.d}, p0/z, [x20, 0, mul vl]	//  (*)
	.loc 38 116 0
..LDL92:
ld1d	{z1.d}, p1/z, [x19, 0, mul vl]	//  (*)
	.loc 38 123 0
..LDL93:
add	x3, x15, x3, lsl #3
ld1d	{z3.d}, p0/z, [x3, 0, mul vl]	//  (*)
	.loc 38 118 0
..LDL94:
ld1d	{z5.d}, p1/z, [x18, z5.d, lsl #3]	//  (*)
	.loc 38 125 0
..LDL95:
ld1d	{z4.d}, p0/z, [x18, z4.d, lsl #3]	//  (*)
	.loc 38 120 0
..LDL96:
fmla	z2.d, p1/m, z5.d, z1.d
	.loc 38 127 0
..LDL97:
fmla	z0.d, p0/m, z4.d, z3.d
	.loc 38 128 0 is_stmt 0
..LDL98:
blt	.L2413
.L2415:		
