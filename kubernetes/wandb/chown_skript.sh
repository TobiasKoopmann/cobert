#!/usr/bin/env bash

for dir in /$1/*; do
  echo "Try changing $dir"
  chown -R 1147:1003 $dir || echo "Cannot own $dir"
done