#!/bin/zsh
# Runs after the main 8192-cap pass: strength sweep at 8192, then the 32K-cap base vs adapter run.
cd "$(dirname "$0")"
B=./swiftgen/.build2/release/wspgen
while pgrep -f "wspgen gen exports/qwen35_4bit_adapter2d/model.aimodel adapter.bin jobs/main.jsonl" >/dev/null; do sleep 30; done
echo "$(date) main pass done; starting sweep" >> logs/queue.log
$B gen exports/qwen35_4bit_adapter2d/model.aimodel adapter.bin jobs/sweep.jsonl results/gen_sweep.jsonl 248046,248044 9216 > logs/gen_sweep.log 2>&1
echo "$(date) sweep exit=$?; starting 32K run" >> logs/queue.log
$B gen exports/qwen35_4bit_adapter2d_ctx33k/model.aimodel adapter.bin jobs/cap32k.jsonl results/gen_cap32k.jsonl 248046,248044 33792 > logs/gen_cap32k.log 2>&1
echo "$(date) 32K exit=$?" >> logs/queue.log
