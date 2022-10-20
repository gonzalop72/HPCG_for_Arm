/*
 *
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  Copyright (C) 2019, Arm Limited and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeMG.cpp

 HPCG routine
 */

#include "ComputeMG.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeSYMGS.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeRestriction.hpp"
#include "ComputeProlongation.hpp"
#include "mytimer.hpp"

#ifdef LIKWID_PERFMON
#include "likwid.h"
#endif

#ifdef ENABLE_MG_COUNTERS
MGTimers MGGlobalTimers[10];

//t1 : symgs 
//T2:  spmv
//t3 : restriction
//t4 : prolongation
//t5 : symgs

//#define TICK_MG()  t0 = mytimer() //!< record current time in 't0'
//#define TOCK_MG(t) t += mytimer() - t0 //!< store time difference in 't' using time in 't0'
#define TICK_MG(f)  if (f) t0 = mytimer()  //!< record current time in 't0'
#define TOCK_MG(f,t) if (f) t += mytimer() - t0 //!< store time difference in 't' using time in 't0'
#else
#define TICK_MG(f)	
#define TOCK_MG(f, t) 
#endif

#ifdef LIKWID_INSTRUMENTATION
#define LIKWID_START(t, s) if(t) { _Pragma("omp parallel") { LIKWID_MARKER_START(s); } }
#define LIKWID_STOP(t, s) if(t) { _Pragma("omp parallel") { LIKWID_MARKER_STOP(s); } }
#else
#define LIKWID_START(t, s)
#define LIKWID_STOP(t, s)
#endif //LIKWID_INSTRUMENTATION

int ComputeMG_TDG(const SparseMatrix  & A, const Vector & r, Vector & x, TraceData& trace) {
	int ierr = 0;

#ifdef ENABLE_MG_COUNTERS
    double t0=0,t1=0,t2=0,t3=0,t4=0,t5=0;
#endif

	if ( A.mgData != 0 ) {
		int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
		for (int i = 0; i < numberOfPresmootherSteps-1; i++ ) {
			ierr += ComputeSYMGS(A, r, x, trace);
		}
#ifdef HPCG_USE_FUSED_SYMGS_SPMV

		// Fuse the last SYMGS iteration with the following SPMV
		// HPCG rules forbid that, so the result will be invalid
		// and therefore not submiteable

#ifdef HPCG_USE_SVE
		ierr += ComputeFusedSYMGS_SPMV_SVE(A, r, x, *A.mgData->Axf);
#elif defined HPCG_USE_NEON
		ierr += ComputeFusedSYMGS_SPMV_NEON(A, r, x, *A.mgData->Axf);
#else
		ierr += ComputeFusedSYMGS_SPMV(A, r, x, *A.mgData->Axf);
#endif
		if ( ierr != 0 ) return ierr;

#else // if !HPCG_USE_FUSED_SYMGS_SPMV

#ifdef ENABLE_MG_COUNTERS
	bool doTrace = trace.enabled && A.mgData != 0 &&  A.mgData->levelMG == trace.level;
#endif

    TICK_MG(trace.enabled);
	LIKWID_START(doTrace, "symgs_tdg");
		ierr += ComputeSYMGS(A, r, x, trace);
		if ( ierr != 0 ) return ierr;
	LIKWID_STOP(trace.enabled && A.mgData != 0 &&  A.mgData->levelMG == trace.level, "symgs_tdg");
    TOCK_MG(trace.enabled, t1);

    TICK_MG(trace.enabled);
	LIKWID_START(trace.enabled && A.mgData != 0 &&  A.mgData->levelMG == trace.level, "spmv_tdg");
		ierr = ComputeSPMV(A, x, *A.mgData->Axf);
		if ( ierr != 0 ) return ierr;
	LIKWID_STOP(trace.enabled && A.mgData != 0 &&  A.mgData->levelMG == trace.level, "spmv_tdg");
    TOCK_MG(trace.enabled, t2);

#endif // HPCG_USE_FUSED_SYMGS_SPMV

    TICK_MG(trace.enabled);
	LIKWID_START(trace.enabled && A.mgData != 0 &&  A.mgData->levelMG == trace.level, "rest_tdg");
		ierr = ComputeRestriction(A, r);
		if ( ierr != 0 ) return ierr;
	LIKWID_STOP(trace.enabled && A.mgData != 0 &&  A.mgData->levelMG == trace.level, "rest_tdg");
    TOCK_MG(trace.enabled, t3);

		ierr = ComputeMG(*A.Ac, *A.mgData->rc, *A.mgData->xc, trace);
		if ( ierr != 0 ) return ierr;

    TICK_MG(trace.enabled);
	LIKWID_START(trace.enabled && A.mgData != 0 &&  A.mgData->levelMG == trace.level, "prol_tdg");
		ierr = ComputeProlongation(A, x);
		if ( ierr != 0 ) return ierr;
	LIKWID_STOP(trace.enabled && A.mgData != 0 &&  A.mgData->levelMG == trace.level, "prol_tdg");
    TOCK_MG(trace.enabled,t4);

    TICK_MG(trace.enabled);
	LIKWID_START(trace.enabled && A.mgData != 0 &&  A.mgData->levelMG == trace.level, "symgs2_tdg");
		int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
		for ( int i = 0; i < numberOfPostsmootherSteps; i++ ) {
			ierr += ComputeSYMGS(A, r, x, trace);
		}
	LIKWID_STOP(trace.enabled && A.mgData != 0 &&  A.mgData->levelMG == trace.level, "symgs2_tdg");
    TOCK_MG(trace.enabled, t5);

#ifdef ENABLE_MG_COUNTERS
    if (trace.enabled) {    
	int lvl = A.mgData->levelMG;
    MGGlobalTimers[lvl].t1 +=t1;
    MGGlobalTimers[lvl].t2 +=t2;
    MGGlobalTimers[lvl].t3 +=t3;
    MGGlobalTimers[lvl].t4 +=t4;
    MGGlobalTimers[lvl].t5 +=t5;
    }
#endif

		if ( ierr != 0 ) return ierr;

	} else {
    TICK_MG(trace.enabled);
	LIKWID_START(trace.enabled && trace.level==4, "symgs_tdg");
		ierr = ComputeSYMGS(A, r, x, trace);
		if ( ierr != 0 ) return ierr;
	LIKWID_STOP(trace.enabled && trace.level==4, "symgs_tdg");
    TOCK_MG(trace.enabled, t5);

#ifdef ENABLE_MG_COUNTERS
    if (trace.enabled) {
    MGGlobalTimers[4].t5 +=t5;
    }
#endif
	}
	return 0;
}

int ComputeMG_BLOCK(const SparseMatrix  & A, const Vector & r, Vector & x, TraceData& trace) {
	int ierr = 0;

#ifdef ENABLE_MG_COUNTERS
    double t0=0,t1=0,t2=0,t3=0,t4=0,t5=0;
#endif

	if ( A.mgData != 0 ) {

    TICK_MG(trace.enabled);
	LIKWID_START(trace.enabled && A.mgData != 0 &&  A.mgData->levelMG == trace.level, "symgs_bl");

		int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
		for (int i = 0; i < numberOfPresmootherSteps; i++ ) {
			ierr += ComputeSYMGS(A, r, x, trace);
		}
		if ( ierr != 0 ) return ierr;
	LIKWID_STOP(trace.enabled && A.mgData != 0 &&  A.mgData->levelMG == trace.level, "symgs_bl");
    TOCK_MG(trace.enabled,t1);

    TICK_MG(trace.enabled);
	LIKWID_START(trace.enabled && A.mgData != 0 &&  A.mgData->levelMG == trace.level, "spmv_bl");
		ierr = ComputeSPMV(A, x, *A.mgData->Axf);
		if ( ierr != 0 ) return ierr;
	LIKWID_STOP(trace.enabled && A.mgData != 0 &&  A.mgData->levelMG == trace.level, "spmv_bl");
    TOCK_MG(trace.enabled,t2);

    TICK_MG(trace.enabled);
	LIKWID_START(trace.enabled && A.mgData != 0 &&  A.mgData->levelMG == trace.level, "rest_bl");
		ierr = ComputeRestriction(A, r);
		if ( ierr != 0 ) return ierr;
	LIKWID_STOP(trace.enabled && A.mgData != 0 &&  A.mgData->levelMG == trace.level, "rest_bl");
    TOCK_MG(trace.enabled,t3);
		
		ierr = ComputeMG(*A.Ac, *A.mgData->rc, *A.mgData->xc, trace);
		if ( ierr != 0 ) return ierr;

    TICK_MG(trace.enabled);
	LIKWID_START(trace.enabled && A.mgData != 0 &&  A.mgData->levelMG == trace.level, "prol_bl");
		ierr = ComputeProlongation(A, x);
		if ( ierr != 0 ) return ierr;
	LIKWID_STOP(trace.enabled && A.mgData != 0 &&  A.mgData->levelMG == trace.level, "prol_bl");
    TOCK_MG(trace.enabled, t4);

    TICK_MG(trace.enabled);
	LIKWID_START(trace.enabled && A.mgData != 0 &&  A.mgData->levelMG == trace.level, "symgs2_bl");
		int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
		for ( int i = 0; i < numberOfPostsmootherSteps; i++ ) {
			ierr += ComputeSYMGS(A, r, x, trace);
		}
		if ( ierr != 0 ) return ierr;
	LIKWID_STOP(trace.enabled && A.mgData != 0 &&  A.mgData->levelMG == trace.level, "symgs2_bl");
    TOCK_MG(trace.enabled, t5);

#ifdef ENABLE_MG_COUNTERS
    if (trace.enabled) {
    int lvl = A.mgData->levelMG;
    MGGlobalTimers[5+lvl].t1 +=t1;
    MGGlobalTimers[5+lvl].t2 +=t2;
    MGGlobalTimers[5+lvl].t3 +=t3;
    MGGlobalTimers[5+lvl].t4 +=t4;
    MGGlobalTimers[5+lvl].t5 +=t5;
    }
#endif

	} else {
    TICK_MG(trace.enabled);
	LIKWID_START(trace.enabled && trace.level == 4, "symgs_bl");
		ierr = ComputeSYMGS(A, r, x, trace);
		if ( ierr != 0 ) return ierr;
	LIKWID_STOP(trace.enabled && trace.level == 4, "symgs_bl");
    TOCK_MG(trace.enabled, t5);

#ifdef ENABLE_MG_COUNTERS
    if (trace.enabled) {
    MGGlobalTimers[5+4].t5 +=t5;
    }
#endif
	}
	return 0;
}


/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/
int ComputeMG(const SparseMatrix  & A, const Vector & r, Vector & x, TraceData& trace) {
	assert(x.localLength == A.localNumberOfColumns);

	ZeroVector(x);

	if ( A.TDG ) {
		return ComputeMG_TDG(A, r, x, trace);
	}
	return ComputeMG_BLOCK(A, r, x, trace);

}
