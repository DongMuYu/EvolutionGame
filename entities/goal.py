# goal.py

import pygame
import setting
from .entity import Entity

class Goal(Entity):
    """
    目标类，继承自实体类，表示游戏中的目标点。
    
    参数:
        pos_x (float): 初始水平位置。
        pos_y (float): 初始垂直位置。
        goal_id (int): 目标编号，默认为0。
    """

    def __init__(self, pos_x=None, pos_y=None, goal_id=0):
        
        # 无位置时报错
        if pos_x is None:
            raise ValueError("目标位置x不能为空")
        if pos_y is None:
            raise ValueError("目标位置y不能为空")
            
        # 调用父类构造函数，设置基础属性
        super().__init__(radius=setting.GOAL_SIZE, pos_x=pos_x, pos_y=pos_y, 
                        entity_id=goal_id, shape='circle', color=setting.RED)

    def deactivate(self):
        """停用目标"""
        self.set_active(False)
        self.set_visible(False)

    def activate(self):
        """激活目标"""
        self.set_active(True)
        self.set_visible(True)

    def get_info(self):
        """获取目标基本信息"""
        base_info = super().get_info()
        base_info.update({
            'active': self.is_active()
        })
        return base_info