#!/bin/zsh
# After the generation queue: throughput benches with the GPU otherwise idle.
cd "$(dirname "$0")"
while pgrep -f run_queue.sh >/dev/null; do sleep 30; done
B=./swiftgen/.build3/release/wspgen
$B bench exports/qwen35_fp16_adapter2d/model.aimodel adapter.bin results/bench_swift_fp16_adapter2d.json > logs/bench_fp16.log 2>&1
$B bench exports/qwen35_4bit_adapter2d_ctx33k/model.aimodel adapter.bin results/bench_swift_adapter2d_ctx33k.json > logs/bench_ctx33k.log 2>&1
echo "$(date) benches done" >> logs/queue.log
