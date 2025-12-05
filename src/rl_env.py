import random
from typing import List, Optional, Tuple

import numpy as np

from .constants import BOARD_HEIGHT, BOARD_WIDTH
from .piece import Piece


class HuarongdaoEnv:
    """华容道强化学习环境"""

    def __init__(self):
        self.pieces: List[Piece] = []
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.reset()

    def reset(self) -> np.ndarray:
        """重置环境到初始状态"""
        self.pieces = []
        self._setup_initial_pieces()
        self._update_board()
        return self._get_state()

    def _setup_initial_pieces(self):
        """设置初始棋子布局"""
        # 曹操 (2x2) - 编号1
        cao_cao = Piece(1, 0, 2, 2, 0, "曹操")
        cao_cao.id = 1
        self.pieces.append(cao_cao)

        # 4个2x1的将 - 编号2-5
        general1 = Piece(0, 0, 1, 2, 0, "将")
        general1.id = 2
        general2 = Piece(3, 0, 1, 2, 0, "将")
        general2.id = 3
        general3 = Piece(0, 2, 1, 2, 0, "将")
        general3.id = 4
        general4 = Piece(3, 2, 1, 2, 0, "将")
        general4.id = 5
        self.pieces.extend([general1, general2, general3, general4])

        # 1个1x2的将 - 编号6
        vertical_general = Piece(1, 2, 2, 1, 0, "将")
        vertical_general.id = 6
        self.pieces.append(vertical_general)

        # 4个1x1的兵 - 编号7-10
        soldier1 = Piece(1, 3, 1, 1, 0, "兵")
        soldier1.id = 7
        soldier2 = Piece(2, 3, 1, 1, 0, "兵")
        soldier2.id = 8
        soldier3 = Piece(0, 4, 1, 1, 0, "兵")
        soldier3.id = 9
        soldier4 = Piece(3, 4, 1, 1, 0, "兵")
        soldier4.id = 10
        self.pieces.extend([soldier1, soldier2, soldier3, soldier4])

    def _update_board(self):
        """更新棋盘状态"""
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        for piece in self.pieces:
            for i in range(piece.y, piece.y + piece.height):
                for j in range(piece.x, piece.x + piece.width):
                    self.board[i][j] = piece.id
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        return self.board.copy()
    
    def get_valid_actions(self) -> List[Tuple[int, int, int]]:
        """获取有效的动作列表 (piece_id, dx, dy)"""
        actions = []
        for piece in self.pieces:
            # 尝试四个方向的移动
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                # 创建临时棋盘进行检测
                temp_board = self.board.copy()
                # 清除当前棋子占据的位置
                for i in range(piece.y, piece.y + piece.height):
                    for j in range(piece.x, piece.x + piece.width):
                        temp_board[i][j] = 0
                
                # 检查移动是否有效
                new_x = piece.x + dx
                new_y = piece.y + dy
                valid = True
                
                # 检查边界
                if new_x < 0 or new_y < 0 or new_x + piece.width > BOARD_WIDTH or new_y + piece.height > BOARD_HEIGHT:
                    valid = False
                
                # 检查碰撞
                if valid:
                    for i in range(new_y, new_y + piece.height):
                        for j in range(new_x, new_x + piece.width):
                            if temp_board[i][j] != 0:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    actions.append((piece.id, dx, dy))
        
        return actions
    
    def get_random_valid_action(self) -> Optional[Tuple[int, int, int]]:
        """随机选择一个有效的动作"""
        valid_actions = self.get_valid_actions()
        if valid_actions:
            return random.choice(valid_actions)
        return None
    
    def step(self, action: Tuple[int, int, int]) -> Tuple[np.ndarray, float, bool]:
        """
        执行动作
        返回: (新状态, 奖励, 是否结束)
        """
        piece_id, dx, dy = action
        
        # 找到对应的棋子
        piece = None
        for p in self.pieces:
            if p.id == piece_id:
                piece = p
                break
        
        if piece is None:
            # 无效动作
            return self._get_state(), -10, False
        
        # 记录曹操当前位置以计算奖励
        cao_cao = None
        for p in self.pieces:
            if p.id == 1:
                cao_cao = p
                break

        # 创建临时棋盘进行检测
        temp_board = self.board.copy()
        # 清除当前棋子占据的位置
        for i in range(piece.y, piece.y + piece.height):
            for j in range(piece.x, piece.x + piece.width):
                temp_board[i][j] = 0
        
        # 检查移动是否有效
        new_x = piece.x + dx
        new_y = piece.y + dy
        valid = True

        # 检查边界
        if (
            new_x < 0
            or new_y < 0
            or new_x + piece.width > BOARD_WIDTH
            or new_y + piece.height > BOARD_HEIGHT
        ):
            valid = False

        # 检查碰撞
        if valid:
            for i in range(new_y, new_y + piece.height):
                for j in range(new_x, new_x + piece.width):
                    if temp_board[i][j] != 0:
                        valid = False
                        break
                if not valid:
                    break

        reward = 0
        done = False

        if valid:
            # 移动棋子
            piece.x = new_x
            piece.y = new_y
            piece.rect.x = new_x * 100  # CELL_SIZE
            piece.rect.y = new_y * 100  # CELL_SIZE
            self._update_board()

            # 检查是否胜利 (曹操到达出口)
            if piece.id == 1 and piece.x == 1 and piece.y == 3:
                reward = 100
                done = True
            else:
                # 计算移动奖励
                # 如果是曹操移动
                if piece.id == 1:
                    # 曹操向出口方向移动（向下）给予正奖励
                    if dy > 0:
                        reward = 0.5
                    # 曹操远离出口方向移动（向上）给予轻微惩罚
                    elif dy < 0:
                        reward = -0.3
                    # 水平移动给予较小惩罚，避免AI反复横跳
                    else:
                        reward = -0.1
                # 如果是其他棋子移动
                else:
                    reward = -0.05
        else:
            # 无效移动给予较大惩罚
            reward = -10

        return self._get_state(), reward, done

    def render(self):
        """渲染当前状态"""
        print(self.board)