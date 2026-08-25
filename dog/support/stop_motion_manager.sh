#!/bin/bash
# ============================================================
# 一键关闭 motion_manager，切换到 LCM 直接控制模式
# 适用: CyberDog2 Jetson (mi 用户)
# 用法: bash /home/mi/stop_motion_manager.sh
# 说明:
#   - 狗重启后 cyberdog_bringup 会自动拉起 motion_manager，
#     需要每次开机后运行本脚本将其停掉，之后即可用 LCM 直接控制运动。
#   - 杀掉 motion_manager 不会影响相机/其他节点（实测 launch 保持存活）。
# ============================================================

PATTERN="/opt/ros2/cyberdog/lib/motion_manager/motion_manager"

count_mm() {
    pgrep -f "$PATTERN" | wc -l
}

echo "===== 关闭 motion_manager（切换到 LCM 控制模式）====="

n=$(count_mm)
echo "当前 motion_manager 进程数: $n"

if [ "$n" -eq 0 ]; then
    echo "motion_manager 未在运行，已经是 LCM 控制模式。"
else
    for i in 1 2 3; do
        echo "[$i] 尝试关闭 motion_manager ..."
        pkill -f "$PATTERN"
        sleep 2
        n=$(count_mm)
        if [ "$n" -eq 0 ]; then
            echo "motion_manager 已关闭。"
            break
        else
            echo "仍在运行 ($n)，重试 ..."
        fi
    done
fi

echo ""
if [ "$(count_mm)" -eq 0 ]; then
    echo "OK: 已切换到 LCM 控制模式。现在可直接用 LCM 下发运动指令（robot_control_cmd）。"
    echo "    注意: 狗重启后需重新运行本脚本。"
else
    echo "FAIL: motion_manager 仍存在，可能需要手动处理:"
    pgrep -af "$PATTERN"
fi