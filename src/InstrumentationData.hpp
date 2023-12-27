#pragma once

struct TraceData {
    int level=1;
    bool enabled=false;
#ifdef CONVERGENCE_TEST
    std::vector<double> convergence_list;
#endif    
};

#ifdef ENABLE_MG_COUNTERS
struct MGTimers {
    double t1=0,t2=0,t3=0,t4=0,t5=0;
};
extern MGTimers MGGlobalTimers[];
#endif
