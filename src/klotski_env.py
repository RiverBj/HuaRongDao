import numpy as np


class KlotskiEnv:
    def __init__(self):
        self.rows, self.cols = 5, 4
        # 棋子定义
        self.pieces = {
            1: {"size": (2, 2), "name": "曹操", "type": "曹操", "color": (255, 50, 50)},
            2: {"size": (2, 1), "name": "张飞", "type": "将", "color": (50, 200, 50)},
            3: {"size": (2, 1), "name": "赵云", "type": "将", "color": (50, 150, 255)},
            4: {"size": (2, 1), "name": "马超", "type": "将", "color": (255, 200, 50)},
            5: {"size": (2, 1), "name": "黄忠", "type": "将", "color": (200, 50, 200)},
            6: {"size": (1, 2), "name": "关羽", "type": "将", "color": (255, 150, 50)},  # 只有关羽是1x2
            7: {"size": (1, 1), "name": "兵1", "type": "兵", "color": (50, 220, 220)},
            8: {"size": (1, 1), "name": "兵2", "type": "兵", "color": (220, 220, 50)},
            9: {"size": (1, 1), "name": "兵3", "type": "兵", "color": (180, 120, 50)},
            10: {"size": (1, 1), "name": "兵4", "type": "兵", "color": (200, 150, 255)},
        }

        # 方向定义
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右
        self.direction_names = ["上", "下", "左", "右"]

        # 初始布局（横刀立马）
        self.initial_board = np.array(
            [[2, 1, 1, 3], [2, 1, 1, 3], [4, 6, 6, 5], [4, 7, 8, 5], [9, 0, 0, 10]],  # 关羽横放
            dtype=np.int32,
        )

        self.reset()

    def reset(self):
        """重置游戏"""
        self.board = self.initial_board.copy()
        self.step_count = 0
        return self.get_state()

    def get_state(self):
        """获取状态表示"""
        return self.board.flatten()

    def get_piece_info(self, piece_id):
        """获取棋子的位置和大小信息"""
        positions = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r, c] == piece_id:
                    positions.append((r, c))

        if not positions:
            return None

        # 计算棋子的边界
        rows = [r for r, _ in positions]
        cols = [c for _, c in positions]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)

        return {
            "positions": positions,
            "top_left": (min_r, min_c),
            "size": (max_r - min_r + 1, max_c - min_c + 1),
        }

    def can_move(self, piece_id, direction):
        """检查棋子是否能移动"""
        piece_info = self.get_piece_info(piece_id)
        if piece_info is None:
            return False

        dr, dc = direction
        top_r, left_c = piece_info["top_left"]
        height, width = piece_info["size"]

        # 检查边界
        new_top_r = top_r + dr
        new_left_c = left_c + dc
        if new_top_r < 0 or new_top_r + height > self.rows:
            return False
        if new_left_c < 0 or new_left_c + width > self.cols:
            return False

        # 检查目标位置是否被其他棋子占据
        for r in range(new_top_r, new_top_r + height):
            for c in range(new_left_c, new_left_c + width):
                if self.board[r, c] not in (0, piece_id):
                    return False

        return True

    def move_piece(self, piece_id, direction):
        """移动棋子"""
        if not self.can_move(piece_id, direction):
            return False

        dr, dc = direction

        # 获取棋子当前位置
        positions = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r, c] == piece_id:
                    positions.append((r, c))

        # 创建新棋盘
        new_board = self.board.copy()

        # 清除原位置
        for r, c in positions:
            new_board[r, c] = 0

        # 设置新位置
        for r, c in positions:
            new_board[r + dr, c + dc] = piece_id

        self.board = new_board
        self.step_count += 1
        return True

    def get_legal_moves(self):
        """获取所有合法移动的动作列表"""
        legal_moves = []

        for piece_id in self.pieces.keys():
            if piece_id == 0:  # 空格不能移动
                continue

            for dir_idx, direction in enumerate(self.directions):
                if self.can_move(piece_id, direction):
                    # 编码动作：piece_id * 4 + dir_idx
                    action = piece_id * 4 + dir_idx
                    legal_moves.append(action)

        return legal_moves

    def step(self, action):
        """
        执行动作
        注意：这里假设传入的action一定是合法的
        如果不合法，会返回惩罚并保持状态不变
        """
        # 解码动作
        piece_id = action // 4
        dir_idx = action % 4
        direction = self.directions[dir_idx]

        # 保存移动前的棋盘用于奖励计算
        old_board = self.board.copy()

        # 尝试移动
        moved = self.move_piece(piece_id, direction)

        # 计算奖励
        reward = 0
        done = False

        if not moved:
            # 非法移动（不应该发生，但为了鲁棒性保留）
            reward = -1.0  # 较大惩罚
        elif self.is_solved():
            # 胜利！
            reward = 50.0  # 大幅胜利奖励
            done = True
        else:
            # 一般移动的奖励
            reward = self.calculate_reward(piece_id, direction, old_board)

            # 添加效率奖励：鼓励用更少步数解决
            reward -= 0.02  # 每步轻微惩罚，鼓励尽快完成

        return self.get_state(), reward, done

    def calculate_reward(self, piece_id, direction, old_board):
        """计算移动的奖励"""
        reward = 0

        if piece_id == 1:  # 曹操
            # 曹操的移动有特殊奖励
            dr, dc = direction

            # 检查曹操是否更接近目标位置
            old_pos = self.find_piece_position(old_board, 1)
            new_pos = self.find_piece_position(self.board, 1)

            if old_pos and new_pos:
                old_center_r = old_pos["top_left"][0] + old_pos["size"][0] / 2
                new_center_r = new_pos["top_left"][0] + new_pos["size"][0] / 2

                # 曹操的目标是在底部（第3-4行）
                target_row = 3.5  # 第3.5行（3和4行的中间）

                if abs(new_center_r - target_row) < abs(old_center_r - target_row):
                    reward += 0.3  # 更接近目标
                else:
                    reward -= 0.1  # 远离目标

            # 方向奖励
            if direction == (1, 0):  # 向下（好）
                reward += 0.2
            elif direction == (0, -1) or direction == (0, 1):  # 左右（中等）
                reward += 0.05
            else:  # 向上（通常不好）
                reward -= 0.1
        else:
            # 其他棋子的移动
            reward = -0.01  # 轻微惩罚

        return reward

    def find_piece_position(self, board, piece_id):
        """在指定棋盘中查找棋子的位置"""
        positions = []
        for r in range(self.rows):
            for c in range(self.cols):
                if board[r, c] == piece_id:
                    positions.append((r, c))

        if not positions:
            return None

        rows = [r for r, _ in positions]
        cols = [c for _, c in positions]

        return {
            "top_left": (min(rows), min(cols)),
            "size": (max(rows) - min(rows) + 1, max(cols) - min(cols) + 1),
        }

    def is_solved(self):
        """检查是否获胜"""
        # 曹操应该在最后两行的中间两列(3,1)-(4,2)
        caocao_info = self.find_piece_position(self.board, 1)
        if not caocao_info:
            return False

        top_r, left_c = caocao_info["top_left"]
        return top_r == 3 and left_c == 1

    def render_text(self):
        """文本显示"""
        print(f"\n步数: {self.step_count}")
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                val = self.board[r, c]
                if val == 0:
                    row.append("  ")
                else:
                    name = self.pieces[val]["name"]
                    row.append(f"{name[:1]} ")  # 显示第一个字
            print(" ".join(row))
