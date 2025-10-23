# entity.py

import pygame
import setting
import os
import math

class Entity:
    """
    基础实体类，表示游戏中的一个通用实体，支持物理运动和多种形状绘制。

    参数:
        size (int): 实体的尺寸。
        pos_x (float): 初始水平位置。
        pos_y (float): 初始垂直位置。
        entity_id (int): 实体编号，默认为0。
        shape (str): 实体形状，可选 'circle', 'rectangle'，默认为 'rectangle'。
        color (tuple): 实体颜色，默认为黑色。
    """

    def __init__(self, radius=None, width=None, height=None, pos_x=None, pos_y=None, entity_id=0,
                 shape='rectangle', color=setting.BLACK):
        self.entity_id = entity_id
        self.shape = shape
        self.color = color

        # 如果未指定位置，随机生成
        if pos_x is None:
            raise NotImplementedError("位置应该显式指定")
        else:
            self.pos_x = pos_x
            
        if pos_y is None:
            raise NotImplementedError("位置应该显式指定")
        else:
            self.pos_y = pos_y
            
        # 根据形状参数设置属性
        if shape == 'circle':
            if radius is None:
                raise ValueError("圆形实体必须指定半径参数")
            self.radius = radius
        elif shape == 'rectangle':
            if width is None or height is None:
                raise ValueError("矩形实体必须指定宽度和高度参数")
            self.width = width
            self.height = height

            # 创建实体矩形（用于碰撞检测）
            self.rect = pygame.Rect(
                self.pos_x - self.width // 2,
                self.pos_y - self.height // 2,
                self.width,
                self.height
            )
        else:
            raise ValueError(f"不支持的形状: {shape}")
            
        # 物理属性
        self.vel_x = 0
        self.vel_y = 0
        self.acc_x = 0
        self.acc_y = 0
        
        # 状态属性
        self.active = True  # 实体是否处于活动状态
        self.visible = True  # 实体是否可见
        self.label_visible = True  # 标签是否可见
        
        # 使用系统字体来显示编号
        if os.name == 'nt':  # Windows系统
            try:
                self.font = pygame.font.SysFont('simhei', 24)  # 使用黑体
            except pygame.error:
                # 如果字体系统未初始化，创建一个虚拟字体对象
                self.font = None
        else:
            try:
                self.font = pygame.font.Font(None, 24)
            except pygame.error:
                self.font = None

    def update(self):
        """
        更新实体状态，包括物理运动
        """
        if not self.active:
            return
        
        # 应用加速度到速度
        self.vel_x += self.acc_x
        self.vel_y += self.acc_y
        
        # 更新位置
        self.pos_x += self.vel_x
        self.pos_y += self.vel_y
        
        # 更新实体矩形位置
        self.rect.x = self.pos_x - self.width // 2
        self.rect.y = self.pos_y - self.height // 2
        
        # 边界检查
        self._check_boundaries()
    
    def _check_boundaries(self):
        """检查实体是否超出边界"""
        # 左边界
        if self.pos_x < self.width // 2:
            self.pos_x = self.width // 2
            self.vel_x = 0
        
        # 右边界
        if self.pos_x > setting.SCREEN_WIDTH - self.width // 2:
            self.pos_x = setting.SCREEN_WIDTH - self.width // 2
            self.vel_x = 0
        
        # 上边界
        if self.pos_y < self.height // 2:
            self.pos_y = self.height // 2
            self.vel_y = 0
        
        # 下边界
        if self.pos_y > setting.SCREEN_HEIGHT - self.height // 2:
            self.pos_y = setting.SCREEN_HEIGHT - self.height // 2
            self.vel_y = 0
        
        # 更新矩形位置
        self.rect.x = self.pos_x - self.width // 2
        self.rect.y = self.pos_y - self.height // 2

    def draw(self, screen):
        """
        在屏幕上绘制实体及其编号
        
        参数:
            screen (pygame.Surface): 要绘制到的目标表面
        """
        if not self.visible:
            return
            
        # 根据实体状态选择颜色
        if self.active:
            color = self.color
        else:
            color = setting.RED  # 死亡实体显示为红色
        
        # 根据形状参数绘制不同的形状
        if self.shape == 'circle':
            pygame.draw.circle(screen, color, (int(self.pos_x), int(self.pos_y)), self.radius)
        elif self.shape == 'rectangle':
            pygame.draw.rect(screen, color, self.rect)
        
        # 绘制实体编号（如果标签可见且字体已初始化）
        if self.label_visible and self.font is not None:
            id_text = self.font.render(str(self.entity_id), True, setting.WHITE)
            text_rect = id_text.get_rect(center=(self.pos_x, self.pos_y))
            screen.blit(id_text, text_rect)

    # 获取和设置位置的方法
    def get_position(self):
        """获取实体位置"""
        return (self.pos_x, self.pos_y)
    
    def set_position(self, x, y):
        """设置实体位置"""
        self.pos_x = x
        self.pos_y = y
        self.rect.x = x - self.width // 2
        self.rect.y = y - self.height // 2
    
    # 获取和设置速度的方法
    def get_velocity(self):
        """获取实体速度"""
        return (self.vel_x, self.vel_y)
    
    def set_velocity(self, vx, vy):
        """设置实体速度"""
        self.vel_x = vx
        self.vel_y = vy
    
    # 获取和设置加速度的方法
    def get_acceleration(self):
        """获取实体加速度"""
        return (self.acc_x, self.acc_y)
    
    def set_acceleration(self, ax, ay):
        """设置实体加速度"""
        self.acc_x = ax
        self.acc_y = ay
    
    # 获取和设置状态的方法
    def is_active(self):
        """检查实体是否活动"""
        return self.active
    
    def set_active(self, active):
        """设置实体活动状态"""
        self.active = active
    
    def is_visible(self):
        """检查实体是否可见"""
        return self.visible
    
    def set_visible(self, visible):
        """设置实体可见性"""
        self.visible = visible
    
    def is_label_visible(self):
        """检查实体标签是否可见"""
        return self.label_visible
    
    def set_label_visible(self, label_visible):
        """设置实体标签可见性"""
        self.label_visible = label_visible
    
    # 获取实体信息的方法
    def get_info(self):
        """获取实体基本信息"""
        return {
            'id': self.entity_id,
            'position': (self.pos_x, self.pos_y),
            'velocity': (self.vel_x, self.vel_y),
            'acceleration': (self.acc_x, self.acc_y),
            'size': self.size,
            'shape': self.shape,
            'active': self.active,
            'visible': self.visible
        }
    
    def check_collision(self, other_entity):
        """
        检查与另一个实体的碰撞
        
        参数:
            other_entity (Entity): 另一个实体对象
            
        返回:
            bool: 是否发生碰撞
        """
        if not self.active or not other_entity.active:
            return False
            
        # 计算两个实体中心点的距离
        distance = math.sqrt((self.pos_x - other_entity.pos_x)**2 + 
                           (self.pos_y - other_entity.pos_y)**2)
        
        # 根据实体形状计算碰撞半径
        if self.shape == 'circle':
            self_radius = self.radius
        else:  # 矩形实体
            self_radius = max(self.width, self.height) // 2
            
        if other_entity.shape == 'circle':
            other_radius = other_entity.radius
        else:  # 矩形实体
            other_radius = max(other_entity.width, other_entity.height) // 2
        
        # 计算碰撞半径之和
        collision_distance = self_radius + other_radius
        
        return distance < collision_distance
    
    def on_collision(self, other_entity):
        """
        碰撞发生时的回调方法，子类可以重写此方法来实现特定的碰撞行为
        
        参数:
            other_entity (Entity): 发生碰撞的另一个实体
        """
        pass

