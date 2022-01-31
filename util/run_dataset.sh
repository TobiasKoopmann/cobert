#!/usr/bin/env bash
docker build -t ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/cobert-dataset:latest -f util/Dockerfile .
docker push ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/cobert-dataset:latest
kubectl -n koopmann delete job cobert-dataset
kubectl -n koopmann create -f util/dataset.yml
