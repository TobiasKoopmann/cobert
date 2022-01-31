#!/usr/bin/env zsh

docker build -t ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/cobert:latest -f kubernetes/Dockerfile .
docker push ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/cobert:latest
rm tmp.json
sed 's/{{}}/'"$1"'/g' kubernetes/template.yml >> tmp.yml
kubectl -n koopmann delete job cobert-$1
kubectl -n koopmann create -f tmp.yml
rm tmp.yml
