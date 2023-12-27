.L2239:					// :entr:term
	.loc 38 103 0 is_stmt 1
..LDL55:
//*    103 */	
sxtw	x16, w2
	.loc 38 105 0
..LDL56:
//*    105 */	
sxtw	x3, w2
	.loc 38 103 0
..LDL57:
//*    103 */	
whilelo	p0.d, x16, x11
	.loc 38 106 0
..LDL58:
//*    106 */	
add	x15, x8, x3, lsl #2
	.loc 38 110 0
..LDL59:
//*    110 */	
add	x2, x16, x5
	.loc 38 105 0
..LDL60:
//*    105 */	
add	x3, x10, x3, lsl #3
	.loc 38 110 0
..LDL61:
//*    110 */	
cmp	w2, w14
	.loc 38 106 0
..LDL62:
//*    106 */	
ld1sw	{z3.d}, p0/z, [x15, 0, mul vl]	//  (*)
	.loc 38 105 0
..LDL63:
//*    105 */	
ld1d	{z2.d}, p0/z, [x3, 0, mul vl]	//  (*)
	.loc 38 107 0
..LDL64:
//*    107 */	
ld1d	{z3.d}, p0/z, [x12, z3.d, lsl #3]	//  (*)
	.loc 38 109 0
..LDL65:
//*    109 */	
fmla	z1.d, p0/m, z3.d, z2.d
	.loc 38 110 0 is_stmt 0
..LDL66:
//*    110 */	
blt	.L2239
