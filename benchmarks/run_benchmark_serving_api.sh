DATA_PATH="." # Replace with your actual path

python3 benchmark_serving_api.py \
  --backend vllm \
  --model $MODEL_NAME \
  --base-url $API_URL \
  --endpoint /v1/completions \
  --api-key $API_KEY \
  --dataset-name sharegpt \
  --dataset-path ${DATA_PATH}/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 100 \
  --save-result \
  --result-filename my_custom_endpoint_serving_bench.json