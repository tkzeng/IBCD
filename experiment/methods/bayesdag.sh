#!/bin/bash

TOP="ex"   # top directory
levels=(5 15 25 50 75 100)
modes=(random sf)

for lvl in "${levels[@]}"; do
  for mode in "${modes[@]}"; do
    for id in "$TOP/$lvl/$mode"/*; do
      csv="$id/Y_matrix.csv"
      if [[ -f "$csv" ]]; then
        echo "$csv"

        # Generate data
        python generate/generate_from_csv.py \
          --csv "$csv" \
          --train-ratio 0.8 \
          --seed 42

        # Move data to data folder
        rm -rf data/Y_matrix
        mv Y_matrix data/Y_matrix
        rm -rf Y_matrix Y_matrix_std

        path="${csv%/Y_matrix.csv}"
        echo "path = $path"

	start=$(date +%s)

        poetry run python -m causica.run_experiment Y_matrix \
          --model_type     bayesdag_linear \
          --model_config   configs/bayesdag/bayesdag_linear.json \
          --dataset_config data/Y_matrix/dataset_config.json \
          --causal_discovery \
          --device         0 \
          --output_dir     "$path" \
          --data_dir       data

	secs=$(( $(date +%s) - start ))
	echo "Time: ${secs}s"

        echo "=== Done with $path ==="$'\n'
      fi
    done
  done
done
