# obstacle.py

import pygame
import setting
from .entity import Entity

class Obstacle(Entity):
    """
    障碍物实体类，继承自实体类，可以障碍物碰撞处理
    """

    def __init__(self, x, y, width, height, kill_individual=False, color=None):
        # 障碍物特有属性
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.kill_individual = kill_individual  # 是否杀死个体
        
        # 计算障碍物的中心位置作为实体位置
        pos_x = x + width // 2
        pos_y = y + height // 2
        
        # 确定颜色：优先使用传入的颜色，否则根据kill_individual决定
        if color is not None:
            obstacle_color = color
        elif kill_individual:
            obstacle_color = setting.RED
        else:
            obstacle_color = setting.BLACK
        
        # 调用父类构造函数，设置基础属性
        super().__init__(radius=None, width=width, height=height, pos_x=pos_x, pos_y=pos_y, entity_id=0, 
                        shape='rectangle', color=obstacle_color)
        
        # 障碍物不显示标签
        self.set_label_visible(False)

    def check_collision(self, player_rect):
        """
        检查与玩家的碰撞
        
        参数:
            player_rect (pygame.Rect): 玩家的矩形实体
            
        返回:
            dict: 碰撞信息，包含碰撞方向和是否碰撞
        """
        # 初始化碰撞信息字典，记录各个方向的碰撞状态
        collision_info = {
            'top': False,      # 玩家从上方碰撞（站在障碍物上）
            'bottom': False,   # 玩家从下方碰撞（碰到障碍物底部）
            'left': False,     # 玩家从左侧碰撞
            'right': False,    # 玩家从右侧碰撞
            'collided': False  # 是否发生碰撞
        }
        
        # 使用pygame的colliderect方法检测两个矩形是否相交
        if player_rect.colliderect(self.rect):
            # 发生碰撞，设置碰撞标志为True
            collision_info['collided'] = True
            
            # 计算碰撞方向：通过比较玩家中心与障碍物中心的相对位置
            # dx: 水平方向相对距离（归一化到[-1, 1]）
            # 正值表示玩家在障碍物右侧，负值表示玩家在障碍物左侧
            dx = (player_rect.centerx - self.rect.centerx) / (self.rect.width / 2 + player_rect.width / 2)
            
            # dy: 垂直方向相对距离（归一化到[-1, 1]）
            # 正值表示玩家在障碍物下方，负值表示玩家在障碍物上方
            dy = (player_rect.centery - self.rect.centery) / (self.rect.height / 2 + player_rect.height / 2)
            
            # 判断主要碰撞方向：比较水平方向和垂直方向的相对距离绝对值
            if abs(dx) > abs(dy):
                # 水平方向碰撞更明显（左右碰撞）
                if dx > 0:
                    # 玩家在障碍物右侧，发生右侧碰撞
                    collision_info['right'] = True
                else:
                    # 玩家在障碍物左侧，发生左侧碰撞
                    collision_info['left'] = True
            else:
                # 垂直方向碰撞更明显（上下碰撞）
                if dy > 0:
                    # 玩家在障碍物下方，发生底部碰撞（玩家碰到障碍物底部）
                    collision_info['bottom'] = True
                else:
                    # 玩家在障碍物上方，发生顶部碰撞（玩家站在障碍物上）
                    collision_info['top'] = True
                    
        return collision_info