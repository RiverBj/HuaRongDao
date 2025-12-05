#!/usr/bin/env python3
"""华容道游戏主入口"""

import argparse

from src.game import HuarongdaoGame
from src.play_trained import play_trained_model
from src.train import train


def play_game():
    """启动游戏"""
    print("启动华容道游戏...")
    print("操作说明:")
    print("- 点击选择棋子")
    print("- 使用方向键移动选中的棋子")
    print("- ESC键退出游戏")

    game = HuarongdaoGame()
    game.run()


def main():
    parser = argparse.ArgumentParser(description="华容道游戏")
    parser.add_argument(
        "--mode",
        choices=["play", "train", "demo"],
        help="运行模式: play(游玩), train(训练) 或 demo(演示训练模型)",
    )
    parser.add_argument("--episodes", type=int, default=1000, help="训练集数 (仅在train模式下使用)")
    parser.add_argument(
        "--model",
        type=str,
        default="huarongdao_dqn.pth",
        help="模型文件路径 (仅在demo模式下使用)",
    )

    args = parser.parse_args()

    if args.mode == "play":
        play_game()
    elif args.mode == "train":
        print(f"开始训练，共 {args.episodes} 轮...")
        train(args.episodes)
        print("训练完成!")
    elif args.mode == "demo":
        print(f"使用模型 {args.model} 进行演示...")
        play_trained_model(args.model)


if __name__ == "__main__":
    main()
