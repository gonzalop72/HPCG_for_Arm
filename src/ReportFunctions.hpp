#pragma once


#include "ComputeMG.hpp"
#include "OutputFile.hpp"
#include <sstream>
#include <string>
#include "omp.h"

std::ostream& operator<<(std::ostream& o, omp_sched_t c) {
#ifdef __FCC_version__
  switch (c) {
#else
  if (c & omp_sched_t::omp_sched_monotonic)
    o << "(monotonic)";
  switch (c & ~omp_sched_t::omp_sched_monotonic) {
#endif
	  case omp_sched_t::omp_sched_static:
      o << "static";
      break;
	  case omp_sched_t::omp_sched_dynamic:
      o << "dynamic";
      break;
	  case omp_sched_t::omp_sched_guided:
      o << "guided";
      break;
	  case omp_sched_t::omp_sched_auto:
      o << "auto";
      break;
    default:
      o << "unknown";
  }
  return o;
}

std::string GetCompiler() {
#ifdef __FCC_version__
    std::string version = __FCC_version__;
    return "Fujitsu " + version;
#elif defined __armclang_version__
    std::string version = __armclang_version__;
    return "Arm " + version;
#else
    std::string version = __VERSION__;
    return "GCC " + version;
#endif
}

#ifdef HPCG_MAN_OPT_DDOT
std::string DdotSVEManualOptimizations(int& unroll) {
#ifdef DDOT_INTRINSICS

	#ifdef DDOT_2_UNROLL
    unroll=2; return "Manual SVE Unroll 2";
	#elif defined DDOT_4_UNROLL
    unroll=4; return "Manual SVE Unroll 4";
	#elif defined DDOT_6_UNROLL
    unroll=6; return "Manual SVE Unroll 6"; 
  #else
    return "Error(1)";
	#endif
#else
	#ifdef DDOT_2_UNROLL
    unroll=2; return "Manual Unroll 2";
	#elif defined DDOT_4_UNROLL
    unroll=4; return "Manual Unroll 4";
	#elif defined DDOT_6_UNROLL
    unroll=6; return "Manual Unroll 6";
  #else
    return "Error(2)";
	#endif
#endif
}
#endif

#ifdef HPCG_MAN_OPT_SPMV_UNROLL
std::string spmvManualOptimizations(int& unroll) {

	#ifdef SPMV_2_UNROLL
    unroll=2; return "Manual Unroll 2";
	#elif defined SPMV_4_UNROLL
    unroll=4; return "Manual Unroll 4";
    #else
    return "Error(3)";
	#endif

}
#endif

void ReportEnvironment(const SparseMatrix& A, OutputFile& doc) {

    std::string ArmGroup = "ARM Configuration";

    //////// ARM CONFIGURATION
    doc.add(ArmGroup, "");
    doc.get(ArmGroup)->add("Compiler", GetCompiler());
    doc.get("ARM Configuration")->add("Mode", 	A.TDG ? "TDG": "BC");
    //char* env_val_1 = getenv("OMP_SCHEDULE");
    omp_sched_t schedule; int chunkSize;
    omp_get_schedule(&schedule, &chunkSize);
    std::stringstream schedule_info("");
    schedule_info << "(" << schedule << "," << chunkSize << ")";
    doc.get("ARM Configuration")->add("Schedule", schedule_info.str());

    std::string optimizations;

#ifdef HPCG_USE_SVE
    optimizations += "SVE,";
    doc.get(ArmGroup)->add("SVE ON", true);
#else
    doc.get(ArmGroup)->add("SVE ON", false);
#endif

#ifdef HPCG_USE_ARMPL_SPMV
    optimizations += "ARMPL_SPMV,";
    doc.get("ARM Configuration")->add("ARMPL SPMV",true);
#endif

//DDOT Optimizations
#ifdef HPCG_USE_SVE
    //no real optimization -- global
#elif defined HPCG_USE_DDOT_ARMPL
    optimizations += "ARMPL_DDOT,";
    doc.get(ArmGroup)->add("ARMPL DDOT",true);
#elif defined HPCG_MAN_OPT_DDOT    
    int unrolling = 0;

    optimizations += "DDOT,";
    doc.get(ArmGroup)->add("DDOT optimized",true);
    doc.get(ArmGroup)->add("DDOT Unrolling", DdotSVEManualOptimizations(unrolling));
    doc.get(ArmGroup)->add("DDOT Unroll-level", unrolling);
#endif
#ifdef HPCG_MAN_OPT_SPMV_UNROLL    //2-WAY UNROLLING ON SPMV
    int unrolling = 0;

    optimizations += "SPMV,";
    doc.get("ARM Configuration")->add("SPMV optimized",true);
    doc.get(ArmGroup)->add("SPMV Unrolling", spmvManualOptimizations(unrolling));
    doc.get(ArmGroup)->add("SPMV Unroll-level", unrolling);
#endif
#ifdef HPCG_MAN_OPT_SCHEDULE_ON    //schedule(runtime)
    optimizations += "SCH,";
    doc.get("ARM Configuration")->add("Runtime Scheduling",true);
#endif

    doc.get("ARM Configuration")->add("Optimizations", optimizations);

    //////// environment values
    std::vector<std::string> env_vars = {"SLURM_JOB_ID",
        "SLURM_JOB_NODELIST", "SLURM_JOB_NUM_NODES", "SLURM_NTASKS","SLURM_NPROCS",
        "SLURM_TASKS_PER_NODE","SLURM_JOB_CPUS_PER_NODE"};
    doc.add("SLURM VARIABLES","");    
    for(auto env_key : env_vars) {
        try {
          std::string env_val = std::string(getenv(env_key.c_str()));
          doc.get("SLURM VARIABLES")->add(env_key, env_val);
        }
        catch(std::exception&) {}
    }
    ////////
}

void ReportInstrumentation(const SparseMatrix& A, OutputFile& doc, int numberOfMgLevels, int numberOfCgSets, int optMaxIters) {
#ifdef ENABLE_MG_COUNTERS
/*    Af = &A;
    doc.get("Multigrid Information")->add("Coarse Grids","");
    for (int i=1; i<numberOfMgLevels; ++i) {
        doc.get("Multigrid Information")->get("Coarse Grids")->add("Grid Level",i);
        doc.get("Multigrid Information")->get("Coarse Grids")->add("Number of Equations",Af->Ac->totalNumberOfRows);
        doc.get("Multigrid Information")->get("Coarse Grids")->add("Number of Nonzero Terms",Af->Ac->totalNumberOfNonzeros);
        doc.get("Multigrid Information")->get("Coarse Grids")->add("Number of Presmoother Steps",Af->mgData->numberOfPresmootherSteps);
        doc.get("Multigrid Information")->get("Coarse Grids")->add("Number of Postsmoother Steps",Af->mgData->numberOfPostsmootherSteps);
    	Af = Af->Ac;
    }*/
/*
    double fnops_symgs_pre[numberOfMgLevels]; //0 is always empty
    double fnops_symgs_post[numberOfMgLevels]; //0 is always empty

    // Op counts from the multigrid preconditioners
    double fnops_precond = 0.0;
    const SparseMatrix * Af = &A;
    for (int i=1; i<numberOfMgLevels; ++i) {
      double fnnz_Af = Af->totalNumberOfNonzeros;
      double fnumberOfPresmootherSteps = Af->mgData->numberOfPresmootherSteps;
      double fnumberOfPostsmootherSteps = Af->mgData->numberOfPostsmootherSteps;
      fnops_precond += fnumberOfPresmootherSteps*fniters*4.0*fnnz_Af; // number of presmoother flops
      fnops_precond += fniters*2.0*fnnz_Af; // cost of fine grid residual calculation (SPMV)
      fnops_precond += fnumberOfPostsmootherSteps*fniters*4.0*fnnz_Af;  // number of postsmoother flops
      fnops_symgs_pre[i] = fnumberOfPresmootherSteps*fniters*4.0*fnnz_Af;
      fnops_symgs_post[i] = fnumberOfPostsmootherSteps*fniters*4.0*fnnz_Af; 
      Af = Af->Ac; // Go to next coarse level
    }

    fnops_precond += fniters*4.0*((double) Af->totalNumberOfNonzeros); // One symmetric GS sweep at the coarsest level
    double fnops = fnops_ddot+fnops_waxpby+fnops_sparsemv+fnops_precond;
    double frefnops = fnops * ((double) refMaxIters)/((double) optMaxIters);
*/
    //FROM:  ======================== FLOP count model =======================================
    double fNumberOfCgSets = numberOfCgSets;
    double fniters = fNumberOfCgSets * (double) optMaxIters;
    //double fnrow = A.totalNumberOfRows;
    //double fnnz = A.totalNumberOfNonzeros;
    double fnops_symgs_pre[numberOfMgLevels]; //0 is always empty
    double fnops_symgs_post[numberOfMgLevels]; //0 is always empty
    double fnops_tdg_spmv[numberOfMgLevels]; //0 is always empty

    // Op counts from the multigrid preconditioners
    const SparseMatrix * Af = &A;
    for (int i=1; i<numberOfMgLevels; ++i) {
        double fnnz_Af = Af->totalNumberOfNonzeros;
        double fnumberOfPresmootherSteps = Af->mgData->numberOfPresmootherSteps;
        double fnumberOfPostsmootherSteps = Af->mgData->numberOfPostsmootherSteps;
        fnops_symgs_pre[i] = fnumberOfPresmootherSteps*fniters*4.0*fnnz_Af; // number of presmoother flops
        fnops_tdg_spmv[i] = fniters*2.0*fnnz_Af; // cost of fine grid residual calculation (SPMV)
        fnops_symgs_post[i] =  fnumberOfPostsmootherSteps*fniters*4.0*fnnz_Af;  // number of postsmoother flops
        
        Af = Af->Ac; // Go to next coarse level
    }

    fnops_symgs_pre[numberOfMgLevels] = fniters*4.0*((double) Af->totalNumberOfNonzeros); // One symmetric GS sweep at the coarsest level

    doc.add(" MG Counters ","");
    doc.add(" MG Performance", "");
    
    auto mgPerf = doc.get(" MG Performance");
    for(int lvl=1; lvl<numberOfMgLevels; ++lvl) {
        doc.get(" MG Counters ")->add("MG Counter level", lvl);
        doc.get(" MG Counters ")->add("MG Counter TDG t1 (symgs)",MGGlobalTimers[lvl].t1);
        doc.get(" MG Counters ")->add("MG Counter TDG t2 (spmv)",MGGlobalTimers[lvl].t2);
        doc.get(" MG Counters ")->add("MG Counter TDG t3 (restriction)",MGGlobalTimers[lvl].t3);     
        doc.get(" MG Counters ")->add("MG Counter TDG t4 (prolongation)",MGGlobalTimers[lvl].t4);
        doc.get(" MG Counters ")->add("MG Counter TDG t5 (symgs)",MGGlobalTimers[lvl].t5);
        doc.get(" MG Counters ")->add("MG Counter BC t1 (symgs)",MGGlobalTimers[5+lvl].t1);
        doc.get(" MG Counters ")->add("MG Counter BC t2 (spmv)",MGGlobalTimers[5+lvl].t2);
        doc.get(" MG Counters ")->add("MG Counter BC t3 (restriction)",MGGlobalTimers[5+lvl].t3);     
        doc.get(" MG Counters ")->add("MG Counter BC t4 (prolongation)",MGGlobalTimers[5+lvl].t4);
        doc.get(" MG Counters ")->add("MG Counter BC t5 (symgs)",MGGlobalTimers[5+lvl].t5);

        if (lvl==1) {
            mgPerf->add("TDG Perf Pre (symgs)",fnops_symgs_pre[lvl]/MGGlobalTimers[lvl].t1/1e9);
            mgPerf->add("TDG Perf SPMV (spmv)",fnops_tdg_spmv[lvl]/MGGlobalTimers[lvl].t2/1e9);
            mgPerf->add("TDG Perf Post (symgs)",fnops_symgs_post[lvl]/MGGlobalTimers[lvl].t5/1e9);
        }
        else {
            mgPerf->add("BC Perf Pre (symgs) LVL_"+std::to_string(lvl),fnops_symgs_pre[lvl]/MGGlobalTimers[5+lvl].t1/1e9);
            mgPerf->add("BC Perf SPMV (spmv) LVL_"+std::to_string(lvl),fnops_tdg_spmv[lvl]/(MGGlobalTimers[lvl].t2+MGGlobalTimers[5+lvl].t2)/1e9);
            mgPerf->add("BC Perf Post (symgs) LVL_"+std::to_string(lvl),fnops_symgs_post[lvl]/MGGlobalTimers[5+lvl].t5/1e9);
        }
    }
#endif
}
