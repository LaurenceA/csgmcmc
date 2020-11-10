#!/usr/bin/env bash
declare -a S_vals=("1000" "300" "100" "30" "10" "3" "1")

for S in "${S_vals[@]}"
do
  echo "S=$S"
  sed -e "s/{{S}}/$S/g" gz2_job_template.sh | qsub
done
