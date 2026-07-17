# Scientific run: hemm-coupling-repair-performance-20260717

**Question:** Why is the current HEMM pair gate inactive, can the registered coupling/router repair pass frozen Stage-0, and does M=1 lgl improve the registered QM9 gap validation/resource tradeoff over ggg?
**Review:** passed

## Plan

- **completed:** scope
- **completed:** counterfactual-diagnosis
- **completed:** repair-and-tests
- **completed:** stage0-matrix
- **completed:** cuda-resource-gate
- **completed:** paired-qm9-validation
- **completed:** provenance-analysis

## Claims

- **claim-hemm-stage0-block:** Neither the learned invariant router nor any registered radial/identity/residual coupling admits M=4/M=8 across all aligned Stage-0 lanes. — evidence: stage0-summary.json, stage0-report.md, stage0-suite.json.gz, stage0-rerun-provenance.json
- **claim-local-vectorization:** The exact-semantics vectorized same-graph candidate implementation passes the registered final-source CUDA resource gate; the earlier unpinned failure is diagnostic rather than causal evidence. — evidence: implementation-evidence.md, cuda-benchmarks-vectorized.json
- **claim-lgl-registered-probe:** M=1 lgl passes the registered three-seed QM9 gap validation and CUDA resource rules against matched M=1 ggg. — evidence: performance-summary.json, performance-report.md, qm9-runs-public.jsonl, cuda-benchmarks-vectorized.json
- **claim-boundary:** The performance result is validation-only on a random-row warm split; no test, cold-molecule, EGNN, or default claim is made. — evidence: performance-summary.json, qm9-runs-public.jsonl, qm9-runs-public-provenance.json

## Visual results

No raster image artifacts recorded.

## Files

- [scope.md](scope.md) — approved-plan; SHA-256 `92c1bcc1f8e04ab41c41163496628987a0620976bc401b1937542699bef6e166`
- [compute-environment.json](compute-environment.json) — environment-record; SHA-256 `4a813904319a0d0b23a78784223593b36eb46741818cc30fe0d0219270b67a35`
- [tdd-red.md](tdd-red.md) — test-record; SHA-256 `41872fc9aa36cfe0e75f1b9759a3d018784486561f328ab35684dbe33c20f9e2`
- [incumbent\_counterfactual.py](incumbent_counterfactual.py) — analysis-code; SHA-256 `eb801edcdad63942a4567c84720b690e500c524c20a8f808174eef6083da3f31`
- [incumbent-counterfactual.json](incumbent-counterfactual.json) — counterfactual-output; SHA-256 `b42b35ff5f66dbb59fcde8015dfcc5cf7c61e9fea1466d164fa51eb00078e2af`
- [analyze\_stage0.py](analyze_stage0.py) — analysis-code; SHA-256 `c98d6a15237d30cbbeeb67e654fbce22185a609b9ba05694cc6a839de7605a44`
- [stage0-summary.json](stage0-summary.json) — decision-table; SHA-256 `1729d3ea0c9c00b8848c9dc267224dedf728d4cb311e46b25b5331c375b556ed`
- [stage0-report.md](stage0-report.md) — scientific-report; SHA-256 `1e551915794496fa3fdb6547b910ffd5b35e3465df2680393059e9ef4c2d1135`
- [stage0-suite.json.gz](stage0-suite.json.gz) — compressed-raw-output; SHA-256 `2897c7443451d7e2e95b9a645c68379840fadb9e1f7853491b7f481b2d9b2bf2`
- [stage0-rerun-provenance.json](stage0-rerun-provenance.json) — execution-receipt; SHA-256 `e83d7cbadea3d03e04e3b69047e1061090e3d3860bb6f49d0e543cca1f5cae0e`
- [implementation-evidence.md](implementation-evidence.md) — verification-record; SHA-256 `00073de350490fdfd34b66a39725708af48a39ea9fed5ea09f05324b00ff2ff5`
- [run\_cuda\_benchmarks.py](run_cuda_benchmarks.py) — benchmark-code; SHA-256 `404ee8a8c19f942fb34a977cd6ade4f82f512dba77633eca79cd20aae0dd62ca`
- [cuda-benchmarks.json](cuda-benchmarks.json) — unpinned-preliminary-benchmark-output; SHA-256 `4d9fb9db9b8c6d10a3c3b41a58f04e5688e0182660c0ed6e872e97d0165ec1a2`
- [cuda-benchmarks-vectorized-smoke.json](cuda-benchmarks-vectorized-smoke.json) — smoke-benchmark-output; SHA-256 `37fe545d19aa404278a324762f73ec36dffee663344bbfc1fd96fd1a5529cba5`
- [cuda-benchmarks-vectorized.json](cuda-benchmarks-vectorized.json) — benchmark-output; SHA-256 `2d6b16ae8546038f22d799d87db64e3726629f702504b257e3722dfb00f8f9c7`
- [qm9-smoke-lgl.json](qm9-smoke-lgl.json) — training-smoke-output; SHA-256 `9a00f442c095507de054063006a997c6c77135c0345e878b15845ece133752f7`
- [run\_qm9\_paired.sh](run_qm9_paired.sh) — execution-code; SHA-256 `961c488ca9e23a4372a190e0828ffb08a824c8dc00e8e1c5e972668906bc2348`
- [publish\_qm9\_ledger.py](publish_qm9_ledger.py) — publication-code; SHA-256 `96e0d46d0245b3059116196f6848f5e9a2a5daa5443a88d5670b62bce53d58f8`
- [qm9-runs-public.jsonl](qm9-runs-public.jsonl) — execution-ledger; SHA-256 `c349985162e1fa21e06d1c22c664245ad2c3a9bd820fe03cf459b52bba211e5b`
- [qm9-runs-public-provenance.json](qm9-runs-public-provenance.json) — publication-transform-receipt; SHA-256 `add2c92fbe9ac2a3562ec4d9b028f9eb79aad9a90d3bd9452c5ce3b2abbfdb26`
- [qm9/ggg-m1-seed41.json](qm9/ggg-m1-seed41.json) — training-output; SHA-256 `6acdd808bd20e033ff54c510a599756d09bfbcf52d7d2a17d72239a5a5f33355`
- [qm9/lgl-m1-seed41.json](qm9/lgl-m1-seed41.json) — training-output; SHA-256 `4a48672522ff83710b31080041479962166af1c65bb2dc1ada86d81661dad3ac`
- [qm9/ggg-m1-seed42.json](qm9/ggg-m1-seed42.json) — training-output; SHA-256 `dcded76a8b6b11a534be87afd8ef8b9ea5472daee76f931fda89461ef2e7d583`
- [qm9/lgl-m1-seed42.json](qm9/lgl-m1-seed42.json) — training-output; SHA-256 `22a558c994e24502da1cba7b0e7dbd3b761d3726be3d72a92a75b3e98338dd59`
- [qm9/ggg-m1-seed43.json](qm9/ggg-m1-seed43.json) — training-output; SHA-256 `f776d74663f5a640683eb4ba120961c660eabeebafb5ea5588fb444a52c30a94`
- [qm9/lgl-m1-seed43.json](qm9/lgl-m1-seed43.json) — training-output; SHA-256 `8c4bfca8026a0c932f0148d31fa47248ab567b8e610b52f615582b72917b3117`
- [analyze\_performance.py](analyze_performance.py) — analysis-code; SHA-256 `5b8b4a53a3fec201af0e6d3ad992135f71c9c22071ad3c58fecbcd5c5d28228c`
- [build\_manifest.py](build_manifest.py) — provenance-code; SHA-256 `4d179e7ff4e73ac6f0fb13308e5c1c271b75ec61519f41d91ae63d3832a7c140`
- [performance-summary.json](performance-summary.json) — decision-table; SHA-256 `bebafe258fa633d894eb975033228317fc4bdddecfc9147b89e86c8a61a26d7a`
- [performance-report.md](performance-report.md) — scientific-report; SHA-256 `bfa6fe4963b5243706df9dc8e9d0f6759a11314bfe0f39a11311752190fb6acd`
- [performance-progress.md](performance-progress.md) — execution-narrative; SHA-256 `23877e267523c33b670d8a286f1dea28b41c1aa3b327ff38f529f230e1af584c`
- [structural-review.json](structural-review.json) — deterministic-review-output; SHA-256 `901c63d7d00aa05be77f85b488662475ee888b1ce79414f4d60f515898faa0df`
- [independent-review.json](independent-review.json) — independent-review-receipt; SHA-256 `2fd47635a38df7f6d46b3142765b7efa6eb27f7758fbca083883ebf9108a80ba`

_Generated from `manifest.json`; this index is a derived view, not evidence._
