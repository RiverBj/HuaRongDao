import numpy as np

from src.util_funcs import demonstrate, play_manually, train_model


def main():
    """主程序"""
    print("=" * 60)
    print("          华容道 AI 学习系统")
    print("=" * 60)
    print("规则说明:")
    print("  • 曹操(2x2, 红色): 需要移动到红色方框")
    print("  • 五虎将: 4个竖将(2x1) + 关羽横将(1x2)")
    print("  • 兵: 4个小兵(1x1)")
    print("=" * 60)

    while True:
        print("\n请选择模式:")
        print("  1. 训练AI (需要时间，但有趣)")
        print("  2. 观看AI演示 (需要已训练模型)")
        print("  3. 手动玩游戏")
        print("  4. 退出")

        choice = input("\n请输入选择 (1-4): ").strip()

        if choice == "1":
            try:
                episodes = int(input("训练轮数 (建议1000-5000): ") or "2000")
                render = input("显示训练过程? (y/n, 显示会变慢): ").lower() == "y"
                render_freq = 10 if render else 0

                agent, stats = train_model(episodes=episodes, render_freq=render_freq)

                # 显示统计
                if stats["solved"]:
                    success_rate = sum(stats["solved"]) / len(stats["solved"]) * 100
                    avg_steps = np.mean([s for s, solved in zip(stats["steps"], stats["solved"]) if solved])
                    print("\n训练统计:")
                    print(f"  • 成功率: {success_rate:.1f}%")
                    print(f"  • 平均成功步数: {avg_steps:.1f}")
                    print("  • 最佳模型已保存为 'klotski_best_model.pth'")

            except KeyboardInterrupt:
                print("\n训练中断")
            except Exception as e:
                print(f"训练出错: {e}")

        elif choice == "2":
            model_file = input("模型文件路径 (回车使用默认): ").strip()
            if not model_file:
                model_file = "klotski_best_model.pth"
            demonstrate(model_file)

        elif choice == "3":
            play_manually()

        elif choice == "4":
            print("再见！")
            break

        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    main()
