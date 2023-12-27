#! /usr/bin/awk -f

BEGIN { OFS = "|"; ORS = " "; FS="|" }
{ if(FNR==1) { PRINTFILE=1; PRINTFILE2=1;} }
#/Region/ && /Group/ { FS = " "; if (PRINTFILE==1) print "\n", FILENAME; print $2; PRINTFILE=0; }  
#/FP_ARITH_INST_RETIRED_SCALAR_DOUBLE STAT/ { FS="|"; if (PRINTFILE==1) print "\n", FILENAME, $2; print OFS; print $4; PRINTFILE=0; }
/FP_ARITH_INST_RETIRED_512B_PACKED_DOUBLE STAT/ { FS="|"; if (PRINTFILE2==1) print "\n", FILENAME, $2; print OFS; print $4; PRINTFILE2=0; }
#/Memory data volume \[GBytes\] STAT/ { FS="|"; if (PRINTFILE==1) print "\n", FILENAME, $2; print OFS; print $3; PRINTFILE=0; }
#/L3 data volume \[GBytes\] STAT/ { FS="|"; if (PRINTFILE==1) print "\n", FILENAME, $2; print OFS; print $3; PRINTFILE=0; }
ENDFILE { }
END { print "\n" }
