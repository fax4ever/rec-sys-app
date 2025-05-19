#!/bin/sh
echo "entry point stasting"
yq eval ".online_store.password = \"$DB_PASSWORD\"" -i /app/feature_repo/feature_store.yaml
cat /app/feature_repo/feature_store.yaml
echo "entry point ending"