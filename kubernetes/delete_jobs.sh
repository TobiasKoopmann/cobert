#!/usr/bin/env zsh

kubectl -n koopmann get pods | awk -F" " '{print $1, $3}' | while read x; do
  if [[ (("$x" == *Completed) || ("$x" == *Error)) && (("$x" == cobert*) || ("$x" == bert4coauthorrec*)) ]]
  then
    name=$(echo $x | awk -F" " '{print $1}')
    echo ${name%??????}
    kubectl -n koopmann delete job ${name%??????}
  fi
done
