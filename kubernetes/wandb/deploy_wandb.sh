docker build -t ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/lsx-wandb:latest .
docker push ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/lsx-wandb:latest

kubectl -n koopmann delete deployment wandb
kubectl -n koopmann apply -f weightsbiases.yml