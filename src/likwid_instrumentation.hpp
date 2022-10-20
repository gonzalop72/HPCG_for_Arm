#pragma once

#ifdef LIKWID_PERFMON
#include "likwid.h"
#endif

#ifdef LIKWID_INSTRUMENTATION
#define LIKWID_START(t, s) if(t) { _Pragma("omp parallel") { LIKWID_MARKER_START(s); } }
#define LIKWID_STOP(t, s) if(t) { _Pragma("omp parallel") { LIKWID_MARKER_STOP(s); } }
#else
#define LIKWID_START(t, s)
#define LIKWID_STOP(t, s)
#endif //LIKWID_INSTRUMENTATION
