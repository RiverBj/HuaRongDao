"""使用训练好的模型演示华容道游戏"""

import sys
import time

import numpy as np
import pygame
import torch

from src.constants import *
from src.dqn_agent import DQNAgent
from src.game import HuarongdaoGame
from src.rl_env import HuarongdaoEnv


class TrainedGame(HuarongdaoGame):
    """使用训练好的模型进行演示的游戏类"""

    def __init__(self, model_path="huarongdao_dqn.pth"):
        super().__init__()
        self.model_path = model_path
        self.agent = None
        self.rl_env = HuarongdaoEnv()
        self.load_model()
        
        # 添加演示相关的属性
        self.demo_mode = False
        self.demo_speed = 1  # 1: 正常速度, 2: 快速, 3: 超快速
        self.step_count = 0
        self.episode_count = 0
        
        # 初始化字体用于显示文本
        self.font = pygame.font.Font(None, 36)

    def load_model(self):
        """加载训练好的模型"""
        try:
            state_size = 4 * 5  # 4行5列的棋盘
            action_size = 40    # 10个棋子 * 4个方向
            self.agent = DQNAgent(state_size, action_size)
            
            # 加载模型权重
            checkpoint = torch.load(self.model_path, map_location=self.agent.device)
            self.agent.q_network.load_state_dict(checkpoint)
            self.agent.q_network.eval()  # 设置为评估模式
            
            print(f"成功加载模型: {self.model_path}")
        except FileNotFoundError:
            print(f"未找到模型文件: {self.model_path}，请先训练模型")
            self.agent = None
        except Exception as e:
            print(f"加载模型失败: {e}")
            self.agent = None

    def sync_with_rl_env(self):
        """同步游戏状态与RL环境"""
        # 清空RL环境中的棋子
        self.rl_env.pieces = []
        
        # 根据当前游戏棋子创建RL环境中的对应棋子
        for i, piece in enumerate(self.pieces):
            rl_piece = piece.__class__(
                piece.x, piece.y, piece.width, piece.height, piece.color, piece.name
            )
            # 设置ID（按照rl_env中的编号规则）
            if piece.name == "曹操":
                rl_piece.id = 1
            elif piece.name == "将":
                # 根据位置确定ID
                if piece.width == 1 and piece.height == 2:  # 垂直将
                    if piece.x == 0 and piece.y == 0:
                        rl_piece.id = 2
                    elif piece.x == 3 and piece.y == 0:
                        rl_piece.id = 3
                    elif piece.x == 0 and piece.y == 2:
                        rl_piece.id = 4
                    elif piece.x == 3 and piece.y == 2:
                        rl_piece.id = 5
                elif piece.width == 2 and piece.height == 1:  # 水平将
                    rl_piece.id = 6
            elif piece.name == "兵":
                # 根据位置确定ID
                if piece.x == 1 and piece.y == 3:
                    rl_piece.id = 7
                elif piece.x == 2 and piece.y == 3:
                    rl_piece.id = 8
                elif piece.x == 0 and piece.y == 4:
                    rl_piece.id = 9
                elif piece.x == 3 and piece.y == 4:
                    rl_piece.id = 10
            
            self.rl_env.pieces.append(rl_piece)
        
        # 更新RL环境的棋盘状态
        self.rl_env._update_board()

    def ai_move(self):
        """让AI执行一步移动"""
        if not self.agent:
            return False
            
        # 同步状态
        self.sync_with_rl_env()
        
        # 获取当前状态
        state = self.rl_env._get_state().flatten()
        
        # 使用模型选择动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
            q_values = self.agent.q_network(state_tensor)
            action = np.argmax(q_values.cpu().data.numpy())
        
        # 解析动作
        piece_id = action // 4 + 1  # 棋子ID (1-10)
        direction = action % 4       # 方向 (0-3)
        
        # 将方向映射为(dx, dy)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dx, dy = directions[direction]
        
        # 执行动作
        _, reward, done = self.rl_env.step((piece_id, dx, dy))
        
        # 同步回游戏界面
        self.sync_from_rl_env()
        
        self.step_count += 1
        print(f"AI移动: 棋子ID={piece_id}, 方向={['下', '上', '右', '左'][direction]}, 奖励={reward}")
        
        return done

    def sync_from_rl_env(self):
        """从RL环境同步到游戏界面"""
        for rl_piece in self.rl_env.pieces:
            # 找到对应的界面棋子
            game_piece = None
            for piece in self.pieces:
                if piece.name == rl_piece.name and \
                   piece.width == rl_piece.width and \
                   piece.height == rl_piece.height:
                    # 对于相同类型的棋子，根据原始位置区分
                    if piece.name == "曹操" and rl_piece.id == 1:
                        game_piece = piece
                        break
                    elif piece.name == "将" and rl_piece.id == 2 and piece.x == 0 and piece.y == 0:
                        game_piece = piece
                        break
                    elif piece.name == "将" and rl_piece.id == 3 and piece.x == 3 and piece.y == 0:
                        game_piece = piece
                        break
                    elif piece.name == "将" and rl_piece.id == 4 and piece.x == 0 and piece.y == 2:
                        game_piece = piece
                        break
                    elif piece.name == "将" and rl_piece.id == 5 and piece.x == 3 and piece.y == 2:
                        game_piece = piece
                        break
                    elif piece.name == "将" and rl_piece.id == 6 and piece.x == 1 and piece.y == 2:
                        game_piece = piece
                        break
                    elif piece.name == "兵" and rl_piece.id == 7 and piece.x == 1 and piece.y == 3:
                        game_piece = piece
                        break
                    elif piece.name == "兵" and rl_piece.id == 8 and piece.x == 2 and piece.y == 3:
                        game_piece = piece
                        break
                    elif piece.name == "兵" and rl_piece.id == 9 and piece.x == 0 and piece.y == 4:
                        game_piece = piece
                        break
                    elif piece.name == "兵" and rl_piece.id == 10 and piece.x == 3 and piece.y == 4:
                        game_piece = piece
                        break
            
            if game_piece:
                # 更新位置
                game_piece.x = rl_piece.x
                game_piece.y = rl_piece.y
                game_piece.rect.x = rl_piece.x * CELL_SIZE
                game_piece.rect.y = rl_piece.y * CELL_SIZE
        
        # 更新棋盘
        self.update_board()

    def reset_game(self):
        """重置游戏"""
        super().setup_pieces()
        self.update_board()
        self.step_count = 0
        self.episode_count += 1
        print("游戏已重置")

    def toggle_demo_mode(self):
        """切换演示模式"""
        self.demo_mode = not self.demo_mode
        print(f"演示模式: {'开启' if self.demo_mode else '关闭'}")

    def toggle_demo_speed(self):
        """切换演示速度"""
        self.demo_speed = self.demo_speed % 3 + 1
        speed_names = {1: "正常", 2: "快速", 3: "超快速"}
        print(f"演示速度: {speed_names[self.demo_speed]}")

    def handle_events(self):
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:  # 按R键重置游戏
                    self.reset_game()
                elif event.key == pygame.K_d:  # 按D键切换演示速度
                    self.toggle_demo_speed()
                elif event.key == pygame.K_a:  # 按A键切换自动演示模式
                    self.toggle_demo_mode()
                elif not self.demo_mode and self.selected_piece:
                    moved = False
                    if event.key == pygame.K_UP:
                        moved = self.selected_piece.move(0, -1, self.board)
                    elif event.key == pygame.K_DOWN:
                        moved = self.selected_piece.move(0, 1, self.board)
                    elif event.key == pygame.K_LEFT:
                        moved = self.selected_piece.move(-1, 0, self.board)
                    elif event.key == pygame.K_RIGHT:
                        moved = self.selected_piece.move(1, 0, self.board)

                    if moved:
                        self.update_board()
        
        # 如果处于演示模式，则由AI执行移动
        if self.demo_mode:
            # 控制演示速度
            speed_delay = {1: 0.5, 2: 0.2, 3: 0.05}
            time.sleep(speed_delay[self.demo_speed])
            
            done = self.ai_move()
            if done:
                print("曹操已到达出口，游戏胜利！")
                self.toggle_demo_mode()  # 胜利后停止演示
                
        return True

    def draw(self):
        """绘制游戏画面"""
        self.screen.fill(WHITE)

        # 绘制网格线
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (0, y), (SCREEN_WIDTH, y))

        # 绘制棋子
        for piece in self.pieces:
            piece.draw(self.screen)

        # 显示选中提示
        if self.selected_piece:
            pygame.draw.rect(self.screen, (255, 100, 100), self.selected_piece.rect, 3)

        # 显示状态信息
        mode_text = f"模式: {'演示' if self.demo_mode else '手动'}"
        mode_surface = self.font.render(mode_text, True, BLACK)
        self.screen.blit(mode_surface, (10, 10))
        
        speed_names = {1: "正常", 2: "快速", 3: "超快速"}
        speed_text = f"速度: {speed_names[self.demo_speed]}"
        speed_surface = self.font.render(speed_text, True, BLACK)
        self.screen.blit(speed_surface, (10, 50))
        
        step_text = f"步数: {self.step_count}"
        step_surface = self.font.render(step_text, True, BLACK)
        self.screen.blit(step_surface, (10, 90))
        
        episode_text = f"回合: {self.episode_count}"
        episode_surface = self.font.render(episode_text, True, BLACK)
        self.screen.blit(episode_surface, (10, 130))

        pygame.display.flip()


def play_trained_model(model_path="huarongdao_dqn.pth"):
    """使用训练好的模型进行演示"""
    print("启动华容道AI演示...")
    print("操作说明:")
    print("- 按 A 键切换自动演示模式")
    print("- 按 R 键重置游戏")
    print("- 按 D 键切换演示速度")
    print("- 按 ESC 键退出")
    print("- 在手动模式下，点击选择棋子并通过方向键移动")
    
    game = TrainedGame(model_path)
    game.run()


if __name__ == "__main__":
    play_trained_model()