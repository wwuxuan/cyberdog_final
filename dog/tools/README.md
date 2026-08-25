# 狗端工具脚本

本目录只放调试和测试脚本，不会被 `../main.py` 调用。执行前加载 ROS 环境，并让 Python 能找到正式运行依赖：

```bash
source /etc/mi/ros2_env.conf
cd /home/mi/cyberdog_competition
PYTHONPATH=$PWD/core python3 tools/<脚本名>.py
```

不要在比赛运行期间同时启动这些脚本，尤其是相机和鱼眼相关脚本。
