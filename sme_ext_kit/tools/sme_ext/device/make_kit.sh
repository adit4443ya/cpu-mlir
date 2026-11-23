#!/usr/bin/env bash
# make_kit.sh: pack everything the dev-compute agent needs into runs/sme_ext/sme_ext_kit.tar.gz
#   sme_ext_kit/RUNBOOK.md, patches/*.patch, tools/sme_ext/**, tools/softmax/run_one.py
set -euo pipefail
R=$(cd "$(dirname "$0")/../../.." && pwd); K=$R/runs/sme_ext/kit/sme_ext_kit
rm -rf $R/runs/sme_ext/kit; mkdir -p $K/patches $K/tools/softmax
cp $R/docs/SME_EXT_DEVICE_RUNBOOK.md $K/RUNBOOK.md; mkdir -p $K/docs; cp $R/docs/{MATMUL_SME_LOWERING,SME_OUTLINE_EXT}.md $K/docs/
cp $R/runs/sme_ext/{sme-outline-ext,combined_with_pr14,pr14}.patch $K/patches/
cp -r $R/tools/sme_ext $K/tools/; rm -rf $K/tools/sme_ext/__pycache__
cp $R/tools/softmax/run_one.py $K/tools/softmax/
tar -C $R/runs/sme_ext/kit -czf $R/runs/sme_ext/sme_ext_kit.tar.gz sme_ext_kit
echo "kit: $R/runs/sme_ext/sme_ext_kit.tar.gz ($(du -h $R/runs/sme_ext/sme_ext_kit.tar.gz | cut -f1))"
