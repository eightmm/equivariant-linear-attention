# Scientific run: architecture-v2-positive-tensor-20260723

**Question:** Can bounded positive scalar magnitude and a shifted persistent-2e Frobenius feature improve real-data selectivity while preserving exact fixed-width O(N), O(3), positivity, permutation, batch-isolation, and disabled-path compatibility contracts?
**Review:** passed

## Plan

- **completed:** P1
- **completed:** P2
- **completed:** P3
- **completed:** P4
- **completed:** P5
- **completed:** P6
- **completed:** P7

## Claims

- **C1:** Disabling bounded scalar content and the shifted persistent-2e kernel preserves the incumbent public behavior, state schema, and common initialization. — status=supported; inference=software compatibility for the tested construction path and public configuration; depends_on=1; supports=3
  - Uncertainty: common-initialization identity is established for CPU construction followed by device transfer, not torch.set\_default\_device(cuda)
  - Next action: keep both options disabled by default
- **C2:** The bounded positive feature and shifted persistent-2e kernel are finite, nonnegative, and agree with their explicit dense algebra while retaining exact fixed-width O(N) factorization. — status=supported; inference=implemented fixed-width mathematical kernel and tested numerical ranges; depends_on=1; supports=2
  - Uncertainty: fixed-degree moment factorization is not a softmax approximation or universality result
  - Next action: retain as an experimental opt-in mechanism
- **C3:** The opt-in architecture preserves the tested O(3), reflection, translation, permutation, and batch-isolation contracts. — status=supported; inference=tested public scalar output and registered equivariant intermediate/output paths; depends_on=2; supports=2
  - Uncertainty: these tests do not establish chirality sensitivity, force consistency, or physical dynamics
  - Next action: preserve the symmetry tests on future kernel changes
- **C4:** At least one architecture-v2 candidate improves the strict seed-42 500-update QM9 validation screen by at least 0.010 eV without exceeding the 0.020 eV regression guard. — status=unsupported; inference=single-seed candidate-admission decision only; contradicts=3; depends_on=1
  - Uncertainty: the original tensor arms had an initialization confound; the separate v2.1 follow-up repaired it and still admitted no candidate
  - Next action: do not promote either option or change defaults
- **C5:** An admitted candidate improves five-seed 2,000-update QM9 mean validation MAE by at least 0.020 eV, wins at least four pairs, and has worst regression no larger than 0.020 eV; EGNN competitiveness requires a separate lower-mean and three-win gate. — status=unavailable; inference=five-seed QM9 confirmation if and only if C4 admits a candidate; contradicts=1
  - Uncertainty: confirmation and private EGNN comparison were correctly not run after the screen failed
  - Next action: no confirmation budget should be spent on this package
- **C6:** The combined candidate reaches train MAE at most 0.10 pK within 3,000 updates on frozen ATOM3D-LBA train rows 0-15. — status=unsupported; inference=train-only capacity on the exact cached 16-complex subset; contradicts=3
  - Uncertainty: sixteen train rows cannot support a PDBBind generalization, ranking, docking, or external-affinity claim
  - Next action: do not promote the candidate; redesign before another expensive confirmation

## Evidence graph

- Nodes: 15; edges: 20; contradicts=7, depends\_on=5, derived\_from=1, supports=7
- Graph file: [evidence-graph.json](evidence-graph.json)

## Visual results

No raster image artifacts recorded.

## Files

- [scope.md](scope.md) — protocol; SHA-256 `cc7df2912809ad39509f00a9665d29faa727572c29b714fde20a21e586dbf09f`
- [initialization-preserving-followup.md](initialization-preserving-followup.md) — protocol-amendment; SHA-256 `9953250580a70924dbde4fd5b089541631863707276206e532315be45cbfffa0`
- [exploratory-followup.md](exploratory-followup.md) — exploratory-record; SHA-256 `34852c49963c5f61ca5bf2bae77d9f5caffef1fe30f33390144ef5a8bcf97386`
- [cuda-smoke.json](cuda-smoke.json) — verification-result; SHA-256 `ffe694f0231fcceee0f8e8aa7c1d3923cfcf6131922a8598b88e2a884f1dda14`
- [cuda-smoke-v2.1.json](cuda-smoke-v2.1.json) — verification-result; SHA-256 `9174f609fb221e9eb2c546077aeca4aa9ed42d167590e5341d1c7e79fb9d45f9`
- [qm9-study/plan.json](qm9-study/plan.json) — execution-plan; SHA-256 `ce66b3316a38fef7de59479e880128847113a8a86a620967ae142e3d7f41d5ba`
- [qm9-study/progress.json](qm9-study/progress.json) — execution-progress; SHA-256 `a2f5a0ec537241922adb0c42bf3bcd9afc68eed79da89cd7a38a4cd766fdbed9`
- [qm9-study/registered-runs/screen/screen-incumbent.json](qm9-study/registered-runs/screen/screen-incumbent.json) — metrics; SHA-256 `7f7243b4c5fd0b8f3dc8898c95b4d0492dddbf1366b5d778dbe80a9352af301b`
- [qm9-study/registered-runs/screen/screen-bounded.json](qm9-study/registered-runs/screen/screen-bounded.json) — metrics; SHA-256 `8c7f104ae817c4195dcd7bdbd9f9e16112edf491475295daa3b1c8db616d23a0`
- [qm9-study/registered-runs/screen/screen-tensor.json](qm9-study/registered-runs/screen/screen-tensor.json) — metrics; SHA-256 `78dd5abbba6674eed1c4616ce24a8348222b37059ff964121c1d9e982f50520a`
- [qm9-study/registered-runs/screen/screen-combined.json](qm9-study/registered-runs/screen/screen-combined.json) — metrics; SHA-256 `a2856ab03a6281905dc6d8e6a11f6ca4a7ce1afadc6a87c0345af6e2df7265d5`
- [qm9-study/summary.json](qm9-study/summary.json) — metrics-summary; SHA-256 `025f2ea30ea5f27523585ed152b0b02f2b6da450e4713eade37fdab796add470`
- [qm9-persistent-only-exploratory.json](qm9-persistent-only-exploratory.json) — exploratory-metrics; SHA-256 `590f3107662e064782f8c8386b5b653007cfdb8ee2601bcd2152276d514b685c`
- [qm9-tensor-eta0001-exploratory.json](qm9-tensor-eta0001-exploratory.json) — exploratory-metrics; SHA-256 `91936dd75c206d90c23c9d2e49ec72674acde73b324436447e3931d34ef2e650`
- [pdbbind-study.json](pdbbind-study.json) — metrics; SHA-256 `07e00e27c4093ef8e8e510df3ef821455fb3b962694839a052fedf11c49d0436`
- [qm9-study-v2.1/plan.json](qm9-study-v2.1/plan.json) — execution-plan; SHA-256 `4ac7922993043e07eed366e8f63cbc9feab700a5a40baf5da4c5fb3f1579e614`
- [qm9-study-v2.1/progress.json](qm9-study-v2.1/progress.json) — execution-progress; SHA-256 `a1072d257355d56c935065239c7dc4ce274cf197b4ee0aea1a57bc36c88bd1a0`
- [qm9-study-v2.1/registered-runs/screen/screen-incumbent.json](qm9-study-v2.1/registered-runs/screen/screen-incumbent.json) — metrics; SHA-256 `166cfbaf790ec433202d35dfdd13c4b465f0f201029e55d8ec301f2c625ef40c`
- [qm9-study-v2.1/registered-runs/screen/screen-bounded.json](qm9-study-v2.1/registered-runs/screen/screen-bounded.json) — metrics; SHA-256 `18a3a6355b37aa49f93cf461314438d9d4db18307cefd5c0298435af50ca236e`
- [qm9-study-v2.1/registered-runs/screen/screen-tensor.json](qm9-study-v2.1/registered-runs/screen/screen-tensor.json) — metrics; SHA-256 `5695c5b08d0aac3625cb106f15f4f6159b04f2e7d631fb43d9ba38a1894c2486`
- [qm9-study-v2.1/registered-runs/screen/screen-combined.json](qm9-study-v2.1/registered-runs/screen/screen-combined.json) — metrics; SHA-256 `e49014689c62f14046d4683eedc2db0b03fcefee6e1f944abd4c6835f65aa327`
- [qm9-study-v2.1/summary.json](qm9-study-v2.1/summary.json) — metrics-summary; SHA-256 `14455aacfb7156b1b62ad70de644463b24dab97f52ce3117da455c047dbfad4a`
- [pdbbind-study-v2.1.json](pdbbind-study-v2.1.json) — metrics; SHA-256 `6b8fe62f49245a3c0728957ba61ca64de2638ae6ee1da5fe6e6248b42cf2008a`
- [results-summary.json](results-summary.json) — analysis; SHA-256 `29a566a2ddce653a0901be0c1d0790944d22efc295fa55671c6dc951d808c683`
- [claim-register.json](claim-register.json) — claim-register; SHA-256 `efa38d3443742b41701abad681fc5fe7de404b0e8a5a4b71964faf6cb3471a02`
- [evidence-graph.json](evidence-graph.json) — evidence-graph; SHA-256 `4639644825409da4b6cf415d1e51c9a3a4fb69424566d65b5951c194d2ca7fa0`
- [compute-environment.json](compute-environment.json) — environment; SHA-256 `7f672cbee61ce285c2453f8d1c136a1e58c33a5d304679df658e1bf9861b735b`
- [execution-log.json](execution-log.json) — execution-log; SHA-256 `17f591333c52b2542f4cd93bdc9f1f4043875d29d77c0d4ee48da53fdb101f13`
- [reproduction-commands.json](reproduction-commands.json) — reproduction-command-record; SHA-256 `952f1e3a06ed7f95c9482b8b307532d45bf83b95a13488947ba8f8c0755a31e7`
- [reference-use-ledger.json](reference-use-ledger.json) — reference-use-ledger; SHA-256 `7b319204ba39168eaca9185854a98904ba6ecef39129f80587389b1a2b6dd5de`
- [verification-summary.json](verification-summary.json) — verification-summary; SHA-256 `f7fdebd529d3322184f77397733b97735a707cfba2a969adb81d629702b365aa`
- [report.md](report.md) — report; SHA-256 `7d1ce14c9c9d04167786f4d331954d85f8ff1fe0afbb98f4f154f8de6fa758fd`
- [review-task.json](review-task.json) — independent-review-task; SHA-256 `d38012ed6363b0cc5b52c08a1fc396da6d3804faa97e0246a1a727482a5dfc97`
- [reviewer-response.json](reviewer-response.json) — independent-review-response; SHA-256 `3fa6f28b65d45c1bf1d4fadc44513c69bb9ee2cf6d014b661a98de998e742d63`
- [review-receipt-initial.json](review-receipt-initial.json) — independent-review-receipt; SHA-256 `3f0b8c0bd071f7d798e456252129e217a971693c7d6460f01b3e56f59909e94b`
- [artifact-validation-finding.json](artifact-validation-finding.json) — review-finding; SHA-256 `f305e34a49dace8b735b3c81b3609d57f1d4d735ae0a4cea42d24bdcfadeaac6`
- [review-task-v2.json](review-task-v2.json) — independent-review-task; SHA-256 `5446b5d9eaa74a4b5d2a079df5008692f1f1426d45c8068f1d1fde83058bcfc9`
- [reviewer-response-v2.json](reviewer-response-v2.json) — independent-review-response; SHA-256 `62fe122fc9ba2b5d5ce83c8ece68d8c90a88c2e1511fc5fa1275cea32479d4f2`
- [review-receipt-v2.json](review-receipt-v2.json) — independent-review-receipt; SHA-256 `72f5396d499baa906288b8c0d66d4a9eb12191ac7ecc099219092771ee5531d5`
- [reviewer-response-final.json](reviewer-response-final.json) — independent-review-response; SHA-256 `aa42b5bc7a0fea3a231e0159d5ad3f80f652f9171b9a112cb56d80f62885ec23`
- [review-receipt-final.json](review-receipt-final.json) — independent-review-receipt; SHA-256 `45a29811595ed51a4b1cdb446cb3a5dcd1a9c8f6a98bc4e107a9bf9a40fb5890`
- [review-task-v3.json](review-task-v3.json) — independent-review-task; SHA-256 `dc10798459bb146f6d757020debde3e059f28daa4fe00a5adb05d8a077f38c70`
- [reviewer-response-v3.json](reviewer-response-v3.json) — independent-review-response; SHA-256 `e10d0951de8853b70fc0d88b7e8fee4e2a373fdc7546ce8701567afd8f0c2bcf`
- [review-receipt-v3.json](review-receipt-v3.json) — independent-review-receipt; SHA-256 `d073249d2ae135c0170da90a6f8dccf7d1351f8db949819464bbe637af1da359`

_Generated from `manifest.json` and validated hashed sidecars; this index is a derived view, not evidence._
