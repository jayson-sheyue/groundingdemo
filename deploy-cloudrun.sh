#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Ensure gcloud is on PATH even when launched from a detached process
export PATH="${PATH}:/Users/apple/google-cloud-sdk/bin:/opt/homebrew/bin:/usr/local/bin"

GCLOUD="$(command -v gcloud)"
echo "Using gcloud at: ${GCLOUD}"

exec "${GCLOUD}" run deploy grounding-demo \
  --source . \
  --project webeye-internal-test \
  --region asia-east1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300 \
  --max-instances 5 \
  --concurrency 40 \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=webeye-internal-test,GOOGLE_CLOUD_LOCATION=global,GOOGLE_GENAI_USE_VERTEXAI=true,GEMINI_MODEL=gemini-2.5-flash,GCS_BUCKET=ysj-bucket-experiment" \
  --quiet
