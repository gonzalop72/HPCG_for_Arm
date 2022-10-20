#pragma once


#include "ComputeMG.hpp"
#include "OutputFile.hpp"
#include <sstream>
#include "omp.h"

std::ostream& operator<<(std::ostream& o, omp_sched_t c) {
  if (c & omp_sched_t::omp_sched_monotonic)
    o << "(monotonic)";
  switch (c & ~omp_sched_t::omp_sched_monotonic) {
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

void ReportEnvironment(const SparseMatrix& A, OutputFile& doc) {

    //////// ARM CONFIGURATION
    doc.add("ARM Configuration", "");
    doc.get("ARM Configuration")->add("Mode", 	A.TDG ? "TDG": "BC");
    //char* env_val_1 = getenv("OMP_SCHEDULE");
    omp_sched_t schedule; int chunkSize;
    omp_get_schedule(&schedule, &chunkSize);
    std::stringstream schedule_info("");
    schedule_info << "(" << schedule << "," << chunkSize << ")";
    doc.get("ARM Configuration")->add("Schedule", schedule_info.str());

    std::string optimizations;
#ifdef HPCG_MAN_OPT_DDOT    
    optimizations += "DDOT,";
    doc.get("ARM Configuration")->add("DDOT optimized",true);
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

void ReportInstrumentation(const SparseMatrix& A, OutputFile& doc) {
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

  doc.add(" MG Counters ","");
  for(int lvl=0; lvl<5; ++lvl) {
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
  }
#endif
}