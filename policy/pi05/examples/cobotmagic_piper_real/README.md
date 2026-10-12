# CobotMagic PiPER 真机部署

把 pi0 / pi0.5 策略部署到双臂 PiPER（70 mm 夹爪，三个 D435）上的客户端。

完整的 sim2real 流程，包括对齐检查、混合微调和部署，见
[docs/tutorials/sim2real_piper.md](../../../../docs/tutorials/sim2real_piper.md)。

| 文件 | 作用 |
|---|---|
| `main.py` | 连接策略服务端并控制真机 |
| `capture_snapshot.py` | 只读抓拍关节状态和三路图像，用于和仿真叠图 |
| `env.py` | openpi 运行时环境，观测键名与训练数据一致 |
| `hardware.py` | 通过 pyAgxArm 控制机械臂和夹爪，通过 pyrealsense2 读相机 |
| `piper_conversions.py` | 单位换算、关节方向标定和安全限幅 |
| `piper_real_test.py` | 不需要硬件的单元测试 |

在 `policy/pi05` 目录下运行：

```bash
pip install -r examples/cobotmagic_piper_real/requirements.txt
pip install -e packages/openpi-client
python -m examples.cobotmagic_piper_real.main --help
```
