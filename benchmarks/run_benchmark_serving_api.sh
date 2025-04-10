DATA_PATH="." # Replace with your actual path

python3 benchmark_serving_api.py \
  --backend vllm \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --base-url $API_URL \
  --endpoint /v1/completions \
  --api-key $API_KEY \
  --dataset-name sharegpt \
  --dataset-path ${DATA_PATH}/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 10 \
  --save-result \
  --result-filename my_custom_endpoint_serving_bench.json