#!/bin/bash

# 获取脚本所在目录的绝对路径
cwd="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认解压目录为当前目录
extract_dir="."

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--directory)
            extract_dir="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 确保解压目录存在
mkdir -p "$extract_dir"

wget -q https://artifactrepo.wux-g.tools.xfusion.com/artifactory/opensource_general/vllm-ascend/0.12.0rc1/package/vllm-ascend-0.12.0rc1.tar.gz --no-check-certificate
# 提取到指定目录，去掉顶层目录
tar -xzf vllm-ascend-0.12.0rc1.tar.gz -C "$extract_dir" --strip-components=1

wget -q https://artifactrepo.wux-g.tools.xfusion.com/artifactory/opensource_general/catlass/v1.0.1/package/catlass-catlass-v1-stable.tar.gz --no-check-certificate
tar -xzf catlass-catlass-v1-stable.tar.gz -C "$extract_dir/csrc/third_party/catlass" --strip-components=1

# 现在解压后的内容直接位于 extract_dir 中
extracted_dir="$extract_dir"

echo "Extracted to: $extracted_dir"

# 转换文件格式
echo "Converting file formats with dos2unix..."
if command -v dos2unix >/dev/null 2>&1; then
    find "$extracted_dir" -type f -exec dos2unix {} \; 2>/dev/null
    echo "File format conversion completed."
else
    echo "dos2unix not available, skipping file format conversion."
fi

# 应用补丁 - 在切换目录前保存当前目录
original_dir="$PWD"

# 检查series.conf是否存在
if [ -f "$cwd/series.conf" ] && grep -q "patch" "$cwd/series.conf"; then
    echo "Applying patches..."
    echo "Series.conf found at: $cwd/series.conf"
    
    # 检查patch目录是否存在
    if [ ! -d "$cwd/patch" ]; then
        echo "Error: Patch directory '$cwd/patch' not found"
        exit 1
    fi
    
    # 切换到提取的目录应用补丁
    cd "$extracted_dir"
    echo "Now in directory: $PWD"
    
    # 读取补丁系列文件并应用每个补丁
    while IFS= read -r line || [[ -n "$line" ]]; do
        # 跳过空行和注释行
        if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
            continue
        fi
        
        # 去除行首尾的空白字符
        line=$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
        
        if [[ -n "$line" ]]; then
            # 使用绝对路径指向补丁文件
            patch_file="$cwd/$line"
            echo "Looking for patch file: $patch_file"
            
            if [ -f "$patch_file" ]; then
                echo "Applying patch: $line"
                if patch -p1 < "$patch_file"; then
                    echo "Patch applied successfully: $line"
                else
                    echo "Error: Failed to apply patch: $line"
                    exit 1
                fi
            else
                echo "Error: Patch file $patch_file not found"
                exit 1
            fi
        fi
    done < "$cwd/series.conf"
    
    # 返回原始目录
    cd "$original_dir"
    echo "All patches applied successfully."
else
    echo "No series.conf found at $cwd/series.conf, skipping patch application."
    # 列出当前目录内容以便调试
    echo "Current directory contents:"
    ls -la
fi
