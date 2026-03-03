# 基本安装（当前目录）
./apply-patch.sh

# 安装到指定目录
./apply-patch.sh -d /path/to/install/directory

# 文件说明
- `apply-patch.sh` - 主安装脚本
- `series.conf` - 补丁顺序配置文件
- `patch/` - 补丁文件目录

# 注意事项

- 确保有稳定的网络连接和足够的磁盘空间
- 需要系统支持 bash、wget、tar、patch 命令
- 安装目录需要具有写权限
- 请确保在运行脚本前已具备必要的权限
