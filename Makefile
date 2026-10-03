.ONESHELL:

.PHONY: help
help:             
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep

.PHONY: venv
venv:
	uv venv

.PHONY: install
install:          ## Install all dependencies
	uv sync --all-extras

.PHONY: install-dev
install-dev:      ## Install runtime + dev extras only
	uv sync --extra dev

.PHONY: install-test
install-test:     ## Install runtime + test extras only
	uv sync --extra test

# --- Code quality ---

.PHONY: lint
lint:             ## Lint the repo and check formatting of the package
	uv run ruff check .
	uv run ruff format --check challenge

.PHONY: format
format:           ## Auto-fix lint findings and format the package
	uv run ruff check --fix challenge
	uv run ruff format challenge

.PHONY: audit
audit:            ## pip audit
	mkdir reports || true
	uv export --frozen --all-extras --no-emit-project --format requirements-txt -o reports/requirements-audit.txt
	uv run pip-audit --disable-pip --no-deps -r reports/requirements-audit.txt

# --- Local deployment ---

.PHONY: run-api-local
run-api-local:          ## Start the API server locally
	DEPLOYMENT_MODE=local uv run uvicorn challenge.api:app --host 0.0.0.0 --port 8000 --no-access-log

.PHONY: training-local
training-local:   ## Run the training job locally
	DEPLOYMENT_MODE=local uv run python -m challenge.training

.PHONY: serving-local
serving-local:    ## Run the serving job locally
	DEPLOYMENT_MODE=local uv run python -m challenge.serving

# --- Tests ---

.PHONY: model-test
model-test:       ## Run model unit tests with coverage
	mkdir reports || true
	uv run pytest --cov-config=.coveragerc --cov-report term --cov-report html:reports/html --cov-report xml:reports/coverage.xml --junitxml=reports/junit.xml --cov=challenge tests/model

.PHONY: pipelines-test
pipelines-test:    ## Run pipeline integration tests with coverage
	mkdir reports || true
	uv run pytest --cov-config=.coveragerc --cov-report term --cov-report html:reports/html --cov-report xml:reports/coverage.xml --junitxml=reports/junit.xml --cov=challenge tests/pipelines

.PHONY: api-test
api-test:         ## Run API tests with coverage
	mkdir reports || true
	uv run pytest --cov-config=.coveragerc --cov-report term --cov-report html:reports/html --cov-report xml:reports/coverage.xml --junitxml=reports/junit.xml --cov=challenge tests/api

STRESS_URL = http://127.0.0.1:8000

.PHONY: stress-test
stress-test:      ## Locust stress test
	mkdir reports || true
	uv run locust -f tests/stress/api_stress.py --print-stats --html reports/stress-test.html --run-time 60s --headless --users 100 --spawn-rate 1 -H $(STRESS_URL)

# --- Containers ---

IMAGE_PREFIX ?= mle-challenge
IMAGE_TAG ?= local
DOCKER_PLATFORM ?= linux/amd64
GIT_SHA ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)

.PHONY: docker-build
docker-build:
	for target in api training serving; do \
		docker buildx build --platform $(DOCKER_PLATFORM) --provenance=false --build-arg GIT_SHA=$(GIT_SHA) \
			-f Dockerfile.$$target -t $(IMAGE_PREFIX)-$$target:$(IMAGE_TAG) --load . || exit 1; \
	done

.PHONY: docker-smoke
docker-smoke:
	docker rm -f $(IMAGE_PREFIX)-smoke >/dev/null 2>&1 || true
	docker run -d --name $(IMAGE_PREFIX)-smoke -p 8080:8080 -e DEPLOYMENT_MODE=local \
		$(IMAGE_PREFIX)-api:$(IMAGE_TAG) >/dev/null
	status=0; \
	curl -fsS --retry 60 --retry-all-errors --retry-delay 1 http://127.0.0.1:8080/health \
		&& echo && curl -fsS "http://127.0.0.1:8080/pipeline/predict/results?page=1&page_size=5" \
		&& echo || status=$$?; \
	[ $$status -eq 0 ] || docker logs $(IMAGE_PREFIX)-smoke; \
	docker rm -f $(IMAGE_PREFIX)-smoke >/dev/null; \
	exit $$status

# --- Infrastructure (Terraform) ---

GCP_PROJECT_ID ?= latam-mle-dzapata
GCP_REGION ?= us-central1
TF_DIR := infra/terraform
TF_ENV ?= staging
TF_STATE_BUCKET ?= $(GCP_PROJECT_ID)-tfstate
TF_VARS ?=

.PHONY: tf-fmt
tf-fmt:           ## Format every Terraform file
	terraform fmt -recursive $(TF_DIR)

.PHONY: tf-validate
tf-validate:      ## Check formatting and validate both Terraform roots
	terraform fmt -recursive -check $(TF_DIR)
	for dir in $(TF_DIR)/bootstrap $(TF_DIR); do \
		terraform -chdir=$$dir init -backend=false -input=false >/dev/null && terraform -chdir=$$dir validate || exit 1; \
	done

.PHONY: tf-state-bucket
tf-state-bucket:  ## One-time: create the versioned Terraform state bucket
	gcloud storage buckets describe gs://$(TF_STATE_BUCKET) --project=$(GCP_PROJECT_ID) >/dev/null 2>&1 || \
		gcloud storage buckets create gs://$(TF_STATE_BUCKET) --project=$(GCP_PROJECT_ID) \
			--location=$(GCP_REGION) --uniform-bucket-level-access --public-access-prevention
	gcloud storage buckets update gs://$(TF_STATE_BUCKET) --project=$(GCP_PROJECT_ID) --versioning

.PHONY: tf-bootstrap-plan
tf-bootstrap-plan: ## One-time: plan the bootstrap root (APIs, service account, WIF, registry)
	terraform -chdir=$(TF_DIR)/bootstrap init -input=false -reconfigure \
		-backend-config=bucket=$(TF_STATE_BUCKET) -backend-config=prefix=bootstrap
	terraform -chdir=$(TF_DIR)/bootstrap plan -input=false -var=project_id=$(GCP_PROJECT_ID) -out=bootstrap.tfplan

.PHONY: tf-bootstrap-apply
tf-bootstrap-apply: ## One-time: apply the plan produced by tf-bootstrap-plan
	terraform -chdir=$(TF_DIR)/bootstrap apply -input=false bootstrap.tfplan

.PHONY: tf-plan
tf-plan:          ## Plan an environment: TF_ENV=staging|prod, extra flags in TF_VARS (e.g. images)
	terraform -chdir=$(TF_DIR) init -input=false -reconfigure \
		-backend-config=bucket=$(TF_STATE_BUCKET) -backend-config=prefix=env/$(TF_ENV)
	terraform -chdir=$(TF_DIR) plan -input=false -var-file=envs/$(TF_ENV).tfvars $(TF_VARS) -out=$(TF_ENV).tfplan

.PHONY: tf-apply
tf-apply:         ## Apply the plan produced by tf-plan
	terraform -chdir=$(TF_DIR) apply -input=false $(TF_ENV).tfplan
