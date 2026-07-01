---
title: Trace Requirements
nav_order: 4
parent: Home
---

# Paraver Trace Requirements

To enable CARM analysis, your Paraver trace must include hardware performance counters for floating-point and memory operations. Configure [Extrae](https://tools.bsc.es/doc/html/extrae/xml.html#xml-section-performance-counters) to include the appropriate counters from the tables below.

## Choosing Which Counters to Include

Include only the necessary counters for your analysis so they fit in a single counter set. Activating too many counters simultaneously can reduce measurement accuracy.

### Example Scenarios

- **App 1:** Only uses double precision; unknown vector ISA usage.
- **App 2:** Vectorized with AVX2, using both single and double precision.

**Recommendation:** If unsure, include all counters and prune them later as you learn more about the application. Using separate load and store counters is recommended for a more detailed analysis.

## Counter Tables

### Intel CPUs

| FP/Mem Operation | Intel Counter | App 1 | App 2 |
|-|-|:-:|:-:|
| Scalar DP Insts | `FP_ARITH_INST_RETIRED:SCALAR_DOUBLE` | ✓ | ✓ |
| Scalar SP Insts | `FP_ARITH_INST_RETIRED:SCALAR_SINGLE` | | ✓ |
| SSE DP Insts | `FP_ARITH_INST_RETIRED:128B_PACKED_DOUBLE` | ✓ | |
| SSE SP Insts | `FP_ARITH_INST_RETIRED:128B_PACKED_SINGLE` | | |
| AVX2 DP Insts | `FP_ARITH_INST_RETIRED:256B_PACKED_DOUBLE` | ✓ | ✓ |
| AVX2 SP Insts | `FP_ARITH_INST_RETIRED:256B_PACKED_SINGLE` | | ✓ |
| AVX512 DP Insts | `FP_ARITH_INST_RETIRED:512B_PACKED_DOUBLE` | ✓ | |
| AVX512 SP Insts | `FP_ARITH_INST_RETIRED:512B_PACKED_SINGLE` | | |
| Loads | `MEM_INST_RETIRED:ALL_LOADS` | ✓ | ✓ |
| Stores | `MEM_INST_RETIRED:ALL_STORES` | ✓ | ✓ |
| Loads and Stores | `MEM_INST_RETIRED:ALL` | | |

### AMD CPUs

| FP/Mem Operation | AMD Counter | App 1 | App 2 |
|-|-|:-:|:-:|
| Mul/Add DP Flops | `retired_sse_avx_operations:dp_mult_add_flops` | ✓ | ✓ |
| Mul/Add SP Flops | `retired_sse_avx_operations:sp_mult_add_flops` | | ✓ |
| Add/Sub DP Flops | `retired_sse_avx_operations:dp_add_sub_flops` | ✓ | ✓ |
| Add/Sub SP Flops | `retired_sse_avx_operations:sp_add_sub_flops` | | ✓ |
| Mul DP Flops | `retired_sse_avx_operations:dp_mult_flops` | ✓ | ✓ |
| Mul SP Flops | `retired_sse_avx_operations:sp_mult_flops` | | ✓ |
| Div DP Flops | `retired_sse_avx_operations:dp_div_flops` | ✓ | ✓ |
| Div SP Flops | `retired_sse_avx_operations:sp_div_flops` | | ✓ |
| Loads | `ls_dispatch:ld_dispatch` | ✓ | ✓ |
| Stores | `ls_dispatch:store_dispatch` | ✓ | ✓ |

## Additional Recommendations

For best results when labeling your code with [Extrae events](https://tools.bsc.es/doc/html/extrae/api.html) (e.g. `Extrae_eventandcounters` calls):

- **Avoid labeling regions that include MPI calls.** Focus on labeling regions of pure computation. MPI calls cause hardware counter timestamps to diverge from region timestamps, preventing those regions from being displayed correctly in the CARM GUI.
