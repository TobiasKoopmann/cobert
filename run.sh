#!/usr/bin/env zsh

# docker build -t ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/cobert-"$1":latest -f kubernetes/Dockerfile .
# docker push ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/cobert-"$1":latest
kubectl -n koopmann delete job cobert-$1-$2-$3
kubectl -n koopmann create -f tmp.yml
rm tmp.yml
