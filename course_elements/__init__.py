"""
课程元素模块
专门负责课程学习中的动态元素（障碍物、目标）的更新和管理
"""

from .level1_elements import Level1ElementsManager
from .level2_elements import Level2ElementsManager
from .level3_elements import Level3ElementsManager

__all__ = ['Level1ElementsManager', 'Level2ElementsManager', 'Level3ElementsManager']