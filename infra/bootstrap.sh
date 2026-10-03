#!/usr/bin/env bash
# gcloud-only provisioning of the whole challenge, equivalent to infra/terraform
# (bootstrap root + one environment). Same resources, names and BigQuery schemas.
#
# Usage:
#   PROJECT_ID=<project> [ENV=staging|prod] [REGION=us-central1] infra/bootstrap.sh [command]
#
# Commands (default: all):
#   foundation  APIs, service account + IAM, Workload Identity Federation, Artifact Registry
#   data        buckets, CSV uploads, BigQuery dataset, tables and load jobs
#   images      build and push the 3 images with Cloud Build, running as the service account
#   deploy      Cloud Run jobs (training, serving) and the public API service
#   seed        run training then serving, and smoke-test the API
#   outputs     print the GitHub Actions variables and the API URL
#   all         foundation + data + images + deploy + seed + outputs

set -Eeuo pipefail
trap 'echo "✗ bootstrap.sh failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- Configuration (mirrors infra/terraform variables and envs/*.tfvars) ------------------------

PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-us-central1}"
ENV="${ENV:-staging}"
GITHUB_REPOSITORY="${GITHUB_REPOSITORY:-davidze012/tech_challenge_latam}"
SA_ID="${SA_ID:-mle-challenge-sa}"
AR_REPO="${AR_REPO:-mle-challenge}"
OPERATOR_MEMBERS="${OPERATOR_MEMBERS:-}"
IMAGE_TAG="${IMAGE_TAG:-$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || date -u +%Y%m%d%H%M%S)}"

SA_EMAIL="${SA_ID}@${PROJECT_ID}.iam.gserviceaccount.com"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}"
INPUT_BUCKET="${PROJECT_ID}-mle-input-${ENV}"
ARTIFACTS_BUCKET="${PROJECT_ID}-mle-artifacts-${ENV}"
DATASET="mle_challenge_${ENV}"
TRAINING_JOB="mle-training-${ENV}"
SERVING_JOB="mle-serving-${ENV}"
API_SERVICE="mle-api-${ENV}"
LABELS="app=mle-challenge,env=${ENV},managed-by=bootstrap-sh"

APIS=(
  artifactregistry.googleapis.com
  bigquery.googleapis.com
  bigquerystorage.googleapis.com
  cloudbuild.googleapis.com
  cloudresourcemanager.googleapis.com
  iam.googleapis.com
  iamcredentials.googleapis.com
  logging.googleapis.com
  monitoring.googleapis.com
  run.googleapis.com
  serviceusage.googleapis.com
  storage.googleapis.com
  sts.googleapis.com
)

SA_ROLES=(
  roles/artifactregistry.writer
  roles/bigquery.dataOwner
  roles/bigquery.jobUser
  roles/bigquery.resourceViewer
  roles/logging.logWriter
  roles/run.admin
  roles/storage.admin
)

# --- Helpers ---------------------------------------------------------------------------------------

log() { printf '\n\033[1m▶ %s\033[0m\n' "$*"; }
ok() { printf '  ✓ %s\n' "$*"; }
die() {
  echo "✗ $*" >&2
  exit 1
}
usage() { sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

gc() { gcloud --project="${PROJECT_ID}" --quiet "$@"; }
bqp() { bq --project_id="${PROJECT_ID}" --location="${REGION}" --quiet "$@"; }

retry() {
  local attempts=$1 n=1
  shift
  until "$@"; do
    if ((n >= attempts)); then
      return 1
    fi
    echo "  … retrying in $((n * 5))s (${n}/${attempts})" >&2
    sleep $((n * 5))
    n=$((n + 1))
  done
}

file_md5_hex() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | cut -d' ' -f1
  else
    md5 -q "$1"
  fi
}

project_number() { gc projects describe "${PROJECT_ID}" --format='value(projectNumber)'; }

image_ref() {
  local name=$1 digest
  digest=$(gc artifacts docker images describe "${REGISTRY}/${name}:${IMAGE_TAG}" \
    --format='value(image_summary.digest)') || die "image ${name}:${IMAGE_TAG} not found (run: images)"
  echo "${REGISTRY}/${name}@${digest}"
}

preflight() {
  [[ -n "${PROJECT_ID}" ]] || {
    usage
    die "PROJECT_ID is required"
  }
  case "${ENV}" in
  staging)
    API_MIN_INSTANCES=0
    API_MAX_INSTANCES=2
    ;;
  prod)
    API_MIN_INSTANCES=1
    API_MAX_INSTANCES=5
    ;;
  *) die "ENV must be staging or prod (got '${ENV}')" ;;
  esac
  local tool
  for tool in gcloud bq curl; do
    command -v "${tool}" >/dev/null 2>&1 || die "${tool} is required"
  done
}

# --- foundation: equivalent to infra/terraform/bootstrap -----------------------------------------

foundation() {
  log "APIs"
  gc services enable "${APIS[@]}"
  ok "enabled: ${APIS[*]}"

  log "Service account ${SA_EMAIL}"
  if gc iam service-accounts describe "${SA_EMAIL}" >/dev/null 2>&1; then
    ok "exists"
  else
    gc iam service-accounts create "${SA_ID}" \
      --display-name="MLE challenge (API, pipelines, CI/CD)" \
      --description="Identity for the API service, the training/serving jobs and GitHub Actions."
    ok "created"
  fi

  log "Project roles of the service account"
  local role member
  for role in "${SA_ROLES[@]}"; do
    retry 6 gc projects add-iam-policy-binding "${PROJECT_ID}" --condition=None \
      --member="serviceAccount:${SA_EMAIL}" --role="${role}" >/dev/null
    ok "${role}"
  done

  log "Service account IAM"
  retry 6 gc iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
    --member="serviceAccount:${SA_EMAIL}" --role=roles/iam.serviceAccountUser >/dev/null
  ok "roles/iam.serviceAccountUser for itself (deploys resources that run as it)"
  for member in ${OPERATOR_MEMBERS//,/ }; do
    retry 6 gc iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
      --member="${member}" --role=roles/iam.serviceAccountTokenCreator >/dev/null
    ok "roles/iam.serviceAccountTokenCreator for ${member}"
  done

  log "Workload Identity Federation (GitHub Actions → ${SA_ID})"
  local pool_state
  pool_state=$(gc iam workload-identity-pools describe github --location=global \
    --format='value(state)' 2>/dev/null || true)
  if [[ -z "${pool_state}" ]]; then
    gc iam workload-identity-pools create github --location=global \
      --display-name="GitHub Actions" --description="Identities issued by GitHub Actions OIDC."
    ok "pool created"
  elif [[ "${pool_state}" == "DELETED" ]]; then
    gc iam workload-identity-pools undelete github --location=global
    ok "pool undeleted (deleted pools stay reserved for 30 days)"
  else
    ok "pool exists"
  fi
  if gc iam workload-identity-pools providers describe github-actions \
    --workload-identity-pool=github --location=global >/dev/null 2>&1; then
    ok "provider exists"
  else
    gc iam workload-identity-pools providers create-oidc github-actions \
      --workload-identity-pool=github --location=global \
      --display-name="GitHub Actions OIDC" \
      --issuer-uri="https://token.actions.githubusercontent.com" \
      --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner,attribute.ref=assertion.ref" \
      --attribute-condition="assertion.repository == '${GITHUB_REPOSITORY}'"
    ok "provider created (only ${GITHUB_REPOSITORY} can exchange tokens)"
  fi
  retry 6 gc iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
    --role=roles/iam.workloadIdentityUser \
    --member="principalSet://iam.googleapis.com/projects/$(project_number)/locations/global/workloadIdentityPools/github/attribute.repository/${GITHUB_REPOSITORY}" \
    >/dev/null
  ok "roles/iam.workloadIdentityUser for the repository principalSet"

  log "Artifact Registry ${REGISTRY}"
  if gc artifacts repositories describe "${AR_REPO}" --location="${REGION}" >/dev/null 2>&1; then
    ok "exists"
  else
    gc artifacts repositories create "${AR_REPO}" --location="${REGION}" \
      --repository-format=docker --description="Images of the API, training and serving components."
    ok "created"
  fi
  local policy
  policy=$(mktemp)
  cat >"${policy}" <<'JSON'
[
  {"name": "keep-recent-versions", "action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": 10}},
  {"name": "delete-old-untagged", "action": {"type": "Delete"}, "condition": {"tagState": "untagged", "olderThan": "7d"}}
]
JSON
  gc artifacts repositories set-cleanup-policies "${AR_REPO}" --location="${REGION}" \
    --policy="${policy}" --no-dry-run >/dev/null
  rm -f "${policy}"
  ok "cleanup policies: keep 10 most recent, delete untagged older than 7 days"
}

# --- data: equivalent to infra/terraform/modules/data --------------------------------------------

upload_if_changed() {
  local file=$1 local_md5 remote_md5
  local_md5=$(openssl md5 -binary "${REPO_ROOT}/data/${file}" | openssl base64)
  remote_md5=$(gc storage objects describe "gs://${INPUT_BUCKET}/${file}" \
    --format='value(md5_hash)' 2>/dev/null || true)
  if [[ "${local_md5}" == "${remote_md5}" ]]; then
    ok "gs://${INPUT_BUCKET}/${file} up to date"
  else
    gc storage cp "${REPO_ROOT}/data/${file}" "gs://${INPUT_BUCKET}/${file}" \
      --content-type=text/csv >/dev/null
    ok "uploaded gs://${INPUT_BUCKET}/${file}"
  fi
}

# One load job per CSV version, exactly like the Terraform module.
load_csv() {
  local table=$1 file=$2 job_id
  job_id="load_${table}_${ENV}_$(file_md5_hex "${REPO_ROOT}/data/${file}" | cut -c1-12)"
  if bqp show -j "${job_id}" >/dev/null 2>&1; then
    ok "${table}: already loaded by ${job_id}"
    return
  fi
  bqp load --job_id="${job_id}" --source_format=CSV --skip_leading_rows=1 --encoding=UTF-8 \
    --replace "${PROJECT_ID}:${DATASET}.${table}" "gs://${INPUT_BUCKET}/${file}" >/dev/null
  ok "${table}: loaded gs://${INPUT_BUCKET}/${file} (${job_id})"
}

data() {
  log "Buckets (${ENV})"
  local bucket lifecycle
  lifecycle=$(mktemp)
  echo '{"rule": [{"action": {"type": "Delete"}, "condition": {"daysSinceNoncurrentTime": 30}}]}' \
    >"${lifecycle}"
  for bucket in "${INPUT_BUCKET}" "${ARTIFACTS_BUCKET}"; do
    if gc storage buckets describe "gs://${bucket}" >/dev/null 2>&1; then
      ok "gs://${bucket} exists"
    else
      gc storage buckets create "gs://${bucket}" --location="${REGION}" \
        --uniform-bucket-level-access --public-access-prevention
      ok "gs://${bucket} created"
    fi
    gc storage buckets update "gs://${bucket}" --versioning --lifecycle-file="${lifecycle}" \
      --update-labels="${LABELS}" >/dev/null
    ok "gs://${bucket}: uniform access, public access prevention, versioning, lifecycle"
  done
  rm -f "${lifecycle}"

  log "Raw CSVs"
  upload_if_changed data.csv
  upload_if_changed serving_input.csv

  log "BigQuery dataset ${DATASET} (${REGION})"
  local bq_labels=(--label=app:mle-challenge "--label=env:${ENV}" --label=managed-by:bootstrap-sh)
  if bqp show --dataset "${PROJECT_ID}:${DATASET}" >/dev/null 2>&1; then
    ok "dataset exists"
  else
    bqp mk --dataset "${bq_labels[@]}" \
      --description="Flight delay challenge (${ENV}): raw flights, serving input and predictions." \
      "${PROJECT_ID}:${DATASET}" >/dev/null
    ok "dataset created"
  fi
  local table
  for table in raw_flights serving_input predictions; do
    if bqp show "${PROJECT_ID}:${DATASET}.${table}" >/dev/null 2>&1; then
      ok "table ${table} exists"
    else
      bqp mk --table "${bq_labels[@]}" --schema="${REPO_ROOT}/infra/schemas/${table}.json" \
        "${PROJECT_ID}:${DATASET}.${table}" >/dev/null
      ok "table ${table} created (infra/schemas/${table}.json)"
    fi
  done

  log "Load jobs"
  load_csv raw_flights data.csv
  load_csv serving_input serving_input.csv
}

# --- images: Cloud Build running as the single service account ----------------------------------

images() {
  log "Images ${REGISTRY}/{api,training,serving}:${IMAGE_TAG} (Cloud Build as ${SA_ID})"
  local config
  config=$(mktemp)
  {
    echo "steps:"
    local previous="" name
    for name in api training serving; do
      echo "  - id: ${name}"
      echo "    name: gcr.io/cloud-builders/docker"
      echo '    env: ["DOCKER_BUILDKIT=1"]'
      echo "    args: [\"build\", \"-f\", \"Dockerfile.${name}\", \"--build-arg\", \"GIT_SHA=${IMAGE_TAG}\", \"-t\", \"${REGISTRY}/${name}:${IMAGE_TAG}\", \".\"]"
      # Sequential builds share the dependency layer built by the first one.
      [[ -n "${previous}" ]] && echo "    waitFor: [\"${previous}\"]"
      previous=${name}
    done
    echo "images:"
    for name in api training serving; do
      echo "  - ${REGISTRY}/${name}:${IMAGE_TAG}"
    done
    echo "options:"
    echo "  logging: CLOUD_LOGGING_ONLY"
  } >"${config}"
  gc builds submit "${REPO_ROOT}" --region="${REGION}" --config="${config}" \
    --service-account="projects/${PROJECT_ID}/serviceAccounts/${SA_EMAIL}" \
    --default-buckets-behavior=regional-user-owned-bucket
  rm -f "${config}"
  ok "pushed api, training and serving:${IMAGE_TAG}"
}

# --- deploy: equivalent to infra/terraform/modules/cloud_run -------------------------------------

deploy() {
  local common_env="DEPLOYMENT_MODE=gcp,GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION}"
  common_env+=",GCP_BUCKET_ARTIFACTS=${ARTIFACTS_BUCKET},GCP_BUCKET_INPUT=${INPUT_BUCKET}"
  common_env+=",BQ_DATASET=${DATASET},BQ_LOCATION=${REGION},LOG_FORMAT=json"

  log "Cloud Run jobs (${ENV})"
  local spec name memory
  for spec in training:2Gi serving:1Gi; do
    name=${spec%%:*}
    memory=${spec##*:}
    gc run jobs deploy "mle-${name}-${ENV}" --region="${REGION}" --image="$(image_ref "${name}")" \
      --service-account="${SA_EMAIL}" --tasks=1 --parallelism=1 --max-retries=0 \
      --task-timeout=600s --cpu=1 --memory="${memory}" \
      --set-env-vars="${common_env}" --labels="${LABELS}" >/dev/null
    ok "mle-${name}-${ENV} (1 vCPU, ${memory}, timeout 600s, 0 retries)"
  done

  log "Cloud Run service ${API_SERVICE}"
  gc run deploy "${API_SERVICE}" --region="${REGION}" --image="$(image_ref api)" \
    --service-account="${SA_EMAIL}" --port=8080 --cpu=1 --memory=1Gi --concurrency=80 \
    --min-instances="${API_MIN_INSTANCES}" --max-instances="${API_MAX_INSTANCES}" \
    --timeout=900 --cpu-boost --cpu-throttling --ingress=all --allow-unauthenticated \
    --startup-probe="httpGet.path=/health,periodSeconds=2,timeoutSeconds=2,failureThreshold=30" \
    --liveness-probe="httpGet.path=/health,periodSeconds=30,timeoutSeconds=5" \
    --set-env-vars="${common_env},TRAINING_JOB_NAME=${TRAINING_JOB},SERVING_JOB_NAME=${SERVING_JOB},PIPELINE_TIMEOUT_S=840,RESULTS_CACHE_TTL_S=30" \
    --labels="${LABELS}" >/dev/null
  ok "${API_SERVICE}: $(api_url) (min ${API_MIN_INSTANCES}, max ${API_MAX_INSTANCES}, public)"
}

api_url() { gc run services describe "${API_SERVICE}" --region="${REGION}" --format='value(status.url)'; }

# --- seed: first model and predictions, then a smoke test ----------------------------------------

seed() {
  log "Training job ${TRAINING_JOB}"
  gc run jobs execute "${TRAINING_JOB}" --region="${REGION}" --wait
  ok "model trained and promoted (gs://${ARTIFACTS_BUCKET}/latest.txt)"

  log "Serving job ${SERVING_JOB}"
  gc run jobs execute "${SERVING_JOB}" --region="${REGION}" --wait
  ok "predictions written to ${PROJECT_ID}.${DATASET}.predictions"

  log "Smoke test"
  local url
  url=$(api_url)
  curl -fsS --retry 10 --retry-all-errors --retry-delay 3 "${url}/health"
  echo
  curl -fsS "${url}/pipeline/predict/results?page=1&page_size=3"
  echo
}

# --- outputs ---------------------------------------------------------------------------------------

outputs() {
  log "Outputs"
  printf '  %-17s = %s\n' \
    GCP_PROJECT_ID "${PROJECT_ID}" \
    GCP_REGION "${REGION}" \
    GCP_SA_EMAIL "${SA_EMAIL}" \
    GCP_WIF_PROVIDER "projects/$(project_number)/locations/global/workloadIdentityPools/github/providers/github-actions" \
    Registry "${REGISTRY}" \
    "API (${ENV})" "$(api_url 2>/dev/null || echo 'not deployed yet')"
}

main() {
  local command=${1:-all}
  case "${command}" in
  -h | --help | help)
    usage
    return
    ;;
  esac
  preflight
  log "Project ${PROJECT_ID} · env ${ENV} · region ${REGION} · image tag ${IMAGE_TAG}"
  case "${command}" in
  foundation | data | images | deploy | seed | outputs) "${command}" ;;
  all)
    foundation
    data
    images
    deploy
    seed
    outputs
    ;;
  *)
    usage
    die "unknown command: ${command}"
    ;;
  esac
}

main "$@"
