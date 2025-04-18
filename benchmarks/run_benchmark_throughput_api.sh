python benchmark_throughput_api.py \
  --openai-api-url $API_URL \
  --endpoint /v1/chat/completions \
  --api-model-name meta-llama/Llama-3.2-3B-Instruct \
  --openai-api-key $API_KEY \
  --tokenizer meta-llama/Llama-3.2-3B-Instruct `# Used for dataset processing` \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 10 `# Adjust as needed` \
  --concurrency 4   `# Adjust based on server capacity` \
  --output-json api_benchmark_results.json `# Optional`
