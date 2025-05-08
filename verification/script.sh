#!/bin/bash

# 定义日志文件和结果文件
LOG_FILE="output.log"
ACCURACY_FILE="accuracy_results.txt"

# 清空或创建日志文件和结果文件（避免追加旧数据）
> "$LOG_FILE"
> "$ACCURACY_FILE"

# 假设需要修改的配置参数列表（示例：不同的学习率）
ADC_RESOL=("2" "3" "4" "5" "6" "7" "8")

# 循环执行多次
for adc in "${ADC_RESOL[@]}"; do
    # 1. 修改 config.json（示例：修改学习率）
    # 使用 jq 工具修改 JSON 文件（需提前安装 jq: sudo apt install jq）
    jq ".chip_config.core_config.matrix_config.adc_resolution = $adc" ../config.json > tmp.json && mv tmp.json ../config.json

    # 2. 执行程序并捕获输出（假设程序命令是 python my_program.py）
    echo "===== 执行配置：ADC位数=$adc =====" | tee -a "$LOG_FILE"
    program_output=$(CUDA_VISIBLE_DEVICES=6 python verification.py --model_path ../models/ONNX/vgg11.onnx --pipeline_type element 2>&1 | tee -a "$LOG_FILE")
    accuracy=$(echo "$program_output" | grep "准确率：" | tail -n 1 | awk '{print $2}')
    # 3. 提取准确率结果
    # 从日志尾部提取最后一行含“准确率”的内容，并保存到结果文件
    echo "adc_resol=$adc: $accuracy">>"$ACCURACY_FILE"
    #grep "准确率：" "$LOG_FILE" | tail -n 1 | awk -F'：' '{print $2}' >> "$ACCURACY_FILE"
done

echo "所有执行完成！结果保存在 $ACCURACY_FILE"