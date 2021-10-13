#!/usr/bin/env bash
docker build -t ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/bert4coauthorrec-dataset:latest -f util/Dockerfile .
docker push ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/bert4coauthorrec-dataset:latest
# kubectl -n koopmann delete job bert4coauthorrec-dataset
kubectl -n koopmann create -f util/dataset.yml
