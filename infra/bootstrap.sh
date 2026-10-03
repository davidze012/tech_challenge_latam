#!/usr/bin/env bash
# gcloud-only provisioning of the whole challenge, equivalent to infra/terraform.
# Implemented in the bootstrap-script phase; until then use the Terraform path
# (make tf-state-bucket, tf-bootstrap-plan/apply, tf-plan/apply — see `make help`).
set -Eeuo pipefail

echo "infra/bootstrap.sh is not implemented yet: use the Terraform targets (make help)." >&2
exit 1
