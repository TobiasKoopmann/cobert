#!/usr/bin/env zsh

echo $1 $2
docker build -t ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/cobert-"$1":latest -f kubernetes/Dockerfile .
docker push ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/cobert-"$1":latest
rm tmp.json
sed 's/{{}}/'"$1"'/ ; s/\[\[\]\]/'"$2"'/' kubernetes/template.yml >> tmp.yml
kubectl -n koopmann delete job cobert-"$1"-"$2"
kubectl -n koopmann create -f tmp.yml
rm tmp.yml
