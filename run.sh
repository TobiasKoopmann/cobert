#!/usr/bin/env zsh

# options are : og-model, nova-model, seq-model,

docker build -t ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/cobert -f kubernetes/Dockerfile .
docker push ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/cobert:latest
sed 's/{{}}/'"$1"'/g' kubernetes/template.yml >> tmp.yml
sed 's/{{}}/'"$1"'/g' config.json >> tmp.json
kubectl -n koopmann delete job cobert-$1
kubectl -n koopmann create -f tmp.yml
rm tmp.yml
rm tmp.json