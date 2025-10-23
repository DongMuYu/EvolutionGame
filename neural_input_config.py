# neural_input_config.py
"""
神经网络输入配置文件
定义输入层大小和信号接收函数，方便自定义配置
"""

from types import NoneType
import numpy as np
import setting
import pygame
import math

def encode_ray_exponential(distance, max_distance, is_danger=False, sensitivity=3.0):
    """
    指数编码：近距离时值变化更敏感
    
    语义：
    负值：安全障碍物
    正值：危险障碍物
    绝对值：距离编码 (越近值越大，且近距离变化敏感)
    
    参数:
        distance: 射线检测到的距离
        max_distance: 最大检测距离
        is_danger: 检测到的障碍物是否危险（True=危险，False=安全）
        sensitivity: 指数敏感度参数
    """
    if distance >= max_distance:
        return 0.0  # 无障碍物
    
    normalized = distance / max_distance
    
    # 核心：使用指数函数增强近距离敏感性
    # 当distance较小时，normalized*sensitivity变化更剧烈
    encoded_magnitude = np.exp(-sensitivity * normalized)
    
    # 判断危险程度，基于障碍物类型而非距离
    if is_danger:
        # 危险障碍物：返回正值，值越大越危险
        # print(f"危险障碍物编码: {encoded_magnitude}")
        return encoded_magnitude  # 范围: (0.0, 1.0]
    else:
        # 安全障碍物：返回负值，绝对值越大距离越近
        # print(f"安全障碍物编码: {-encoded_magnitude}")
        return -encoded_magnitude  # 范围: [-1.0, 0.0)

def get_input_size():
    """
    获取输入层大小
    
    返回:
        int: 输入层大小
    """
    return 14 # 目标相对x位置, 目标相对y位置 + 12个射线检测输入

def draw_sensor_inputs(screen, individual, goals, obstacles=None, draw_enabled=False):
    """
    绘制传感器输入效果，包括射线检测和相对位置向量线
    
    参数:
        screen: pygame屏幕对象
        individual: 个体对象
        goals: 目标列表
        obstacles: 障碍物列表（可选）
        draw_enabled: 是否启用绘制，默认为False
    """
    if not draw_enabled:
        return
    
    # 12个射线检测输入，每30度一条检查射线，从x轴正方向开始
    max_ray_distance = setting.MAX_RAY_DISTANCE  # 最大射线检测距离
    ray_angles = [i * 30 for i in range(12)]  # 0°, 30°, 60°, ..., 330°
    
    # 绘制射线
    for angle_deg in ray_angles:
        # 将角度转换为弧度
        angle_rad = math.radians(angle_deg)
        
        # 计算射线方向向量
        ray_dir_x = math.cos(angle_rad)
        ray_dir_y = math.sin(angle_rad)
        
        # 射线检测：检测与屏幕边界的碰撞
        # 计算射线与屏幕边界的交点
        t_x = float('inf')
        t_y = float('inf')
        
        # 与左右边界的交点
        if ray_dir_x > 0:
            t_x = (setting.SCREEN_WIDTH - individual.pos_x) / ray_dir_x
        elif ray_dir_x < 0:
            t_x = (0 - individual.pos_x) / ray_dir_x
        
        # 与上下边界的交点
        if ray_dir_y > 0:
            t_y = (setting.SCREEN_HEIGHT - individual.pos_y) / ray_dir_y
        elif ray_dir_y < 0:
            t_y = (0 - individual.pos_y) / ray_dir_y
        
        # 取最近的交点
        t = min(t_x, t_y)
        
        # 默认距离为最大射线距离
        distance = max_ray_distance
        is_danger = False  # 屏幕边界视为不危险障碍物
        
        # 如果射线方向不为0，计算与屏幕边界的距离
        if t != float('inf'):
            distance = min(t, max_ray_distance)
            
            # 检测与障碍物的碰撞
            if obstacles:
                # 射线与障碍物的碰撞检测
                closest_obstacle_distance = float('inf')
                closest_obstacle = None
                
                for obstacle in obstacles:
                    # 检查障碍物是否激活
                    if obstacle.active:
                        # 射线与矩形碰撞检测算法
                        # 计算射线与矩形边界的交点
                        t_min = 0
                        t_max = distance
                        
                        # 检查x方向
                        if abs(ray_dir_x) < 1e-6:
                            # 射线平行于x轴
                            if individual.pos_x < obstacle.rect.x or individual.pos_x > obstacle.rect.x + obstacle.width:
                                continue  # 射线不在矩形范围内
                        else:
                            # 计算射线与矩形左右边界的交点
                            t1 = (obstacle.rect.x - individual.pos_x) / ray_dir_x
                            t2 = (obstacle.rect.x + obstacle.width - individual.pos_x) / ray_dir_x
                            
                            # 确保t1 <= t2
                            if t1 > t2:
                                t1, t2 = t2, t1
                            
                            # 更新有效区间
                            t_min = max(t_min, t1)
                            t_max = min(t_max, t2)
                            
                            # 检查是否有有效交点
                            if t_min > t_max:
                                continue
                        
                        # 检查y方向
                        if abs(ray_dir_y) < 1e-6:
                            # 射线平行于y轴
                            if individual.pos_y < obstacle.rect.y or individual.pos_y > obstacle.rect.y + obstacle.height:
                                continue  # 射线不在矩形范围内
                        else:
                            # 计算射线与矩形上下边界的交点
                            t1 = (obstacle.rect.y - individual.pos_y) / ray_dir_y
                            t2 = (obstacle.rect.y + obstacle.height - individual.pos_y) / ray_dir_y
                            
                            # 确保t1 <= t2
                            if t1 > t2:
                                t1, t2 = t2, t1
                            
                            # 更新有效区间
                            t_min = max(t_min, t1)
                            t_max = min(t_max, t2)
                            
                            # 检查是否有有效交点
                            if t_min > t_max:
                                continue
                        
                        # 如果有有效交点，更新最近障碍物距离
                        if t_min >= 0 and t_min < closest_obstacle_distance:
                            closest_obstacle_distance = t_min
                            closest_obstacle = obstacle
                
                # 如果检测到障碍物，更新距离和危险程度
                if closest_obstacle:
                    distance = closest_obstacle_distance
                    # 根据障碍物的kill_individual属性判断危险程度
                    is_danger = closest_obstacle.kill_individual
        
        # 使用指数编码方式编码距离，传递障碍物危险度信息
        encoded_value = encode_ray_exponential(distance, max_ray_distance, is_danger)
        
        # 根据射线检测值确定射线长度和颜色
        if encoded_value > 0:  # 危险障碍物
            ray_color = (255, 0, 0)  # 红色射线
        elif encoded_value < 0:  # 安全障碍物
            ray_color = (0, 0, 0)  # 黑色射线
        else:  # 无障碍物
            ray_color = (128, 128, 128)  # 灰色射线（无障碍物）
        
        # 计算射线终点 - 确保射线在碰撞点停止
        ray_end_x = individual.pos_x + ray_dir_x * distance
        ray_end_y = individual.pos_y + ray_dir_y * distance
        
        # 绘制射线
        pygame.draw.line(screen, ray_color, 
                       (individual.pos_x, individual.pos_y), 
                       (ray_end_x, ray_end_y), 2)
        
        # 如果检测到障碍物或击中边界，绘制碰撞点
        if encoded_value != 0:
            # 计算碰撞点位置
            collision_x = individual.pos_x + ray_dir_x * distance
            collision_y = individual.pos_y + ray_dir_y * distance
            
            # 确保碰撞点坐标是有效数字
            if collision_x is not None and collision_y is not None:
                try:
                    # 检查是否为无穷大或NaN
                    if not (math.isfinite(collision_x) and math.isfinite(collision_y)):
                        # 如果坐标不是有限值，跳过绘制这个碰撞点
                        pass
                    else:
                        # 转换为整数并确保是有效数字
                        collision_x_int = int(collision_x)
                        collision_y_int = int(collision_y)
                        
                        # 绘制碰撞点 - 边界碰撞和障碍物碰撞都使用蓝色
                        point_color = (0, 0, 255)  # 蓝色碰撞点
                        
                        # 绘制碰撞点
                        pygame.draw.circle(screen, point_color, 
                                         (collision_x_int, collision_y_int), 5)
                except (TypeError, ValueError, OverflowError):
                    # 如果转换失败，跳过绘制这个碰撞点
                    pass
    
    # 绘制相对位置向量线
    if goals:
        # 按编号顺序查找未被触碰的激活目标
        active_goals = []
        for goal in goals:
            # 只考虑激活且未被触碰的目标
            if goal.active and goal.entity_id not in individual.reached_goal_ids:
                active_goals.append(goal)
        
        # 按编号排序，找到最小编号的未被触碰目标
        if active_goals:
            # 按编号排序
            active_goals.sort(key=lambda g: g.entity_id)
            # 使用最小编号的未被触碰目标
            target_goal = active_goals[0]
            
            # 绘制从个体到目标的向量线
            pygame.draw.line(screen, (0, 255, 0), 
                           (individual.pos_x, individual.pos_y), 
                           (target_goal.pos_x, target_goal.pos_y), 2)

def get_sensor_inputs(individual, goals, obstacles):
    """
    获取传感器输入值 - 自身与目标的相对位置2个值 + 12个射线检测输入
    按顺序查找未被触碰的最小编号目标
    
    参数:
        individual: 种群个体对象
        goals: 目标列表
        obstacles: 障碍物列表（可选）
        
    返回:
        list: 归一化的相对位置输入值数组
    """
    inputs = []
    
    # 按编号顺序查找未被触碰的激活目标
    if goals:
        active_goals = []
        for goal in goals:
            # 只考虑激活且未被触碰的目标
            if goal.active and goal.entity_id not in individual.reached_goal_ids:
                active_goals.append(goal)
        
        # 按编号排序，找到最小编号的未被触碰目标
        if active_goals:
            # 按编号排序
            active_goals.sort(key=lambda g: g.entity_id)
            # 使用最小编号的未被触碰目标
            target_goal = active_goals[0]
            
            # 计算相对位置（目标位置 - 自身位置）
            dx = target_goal.pos_x - individual.pos_x
            dy = target_goal.pos_y - individual.pos_y
            
            # 使用tanh函数归一化相对位置到[-1, 1]范围
            scale_factor = 0.01  # 缩放因子，使输入值在tanh的有效范围内
            # x方向：正值为目标在右侧，负值为目标在左侧
            inputs.append(np.tanh(dx * scale_factor))  # 相对x位置
            # y方向：正值为目标在下侧，负值为目标在上侧
            inputs.append(np.tanh(dy * scale_factor))  # 相对y位置
        else:
            # 如果没有未被触碰的激活目标，使用0作为默认值
            inputs.append(0.0)  # 相对x位置
            inputs.append(0.0)  # 相对y位置
    else:
        # 如果没有目标，使用0作为默认值
        inputs.append(0.0)  # 相对x位置
        inputs.append(0.0)  # 相对y位置
    
    # 12个射线检测输入，每30度一条检查射线，从x轴正方向开始
    max_ray_distance = setting.MAX_RAY_DISTANCE  # 最大射线检测距离
    ray_angles = [i * 30 for i in range(12)]  # 0°, 30°, 60°, ..., 330°
    
    for angle_deg in ray_angles:
        # 将角度转换为弧度
        angle_rad = math.radians(angle_deg)
        
        # 计算射线方向向量
        ray_dir_x = math.cos(angle_rad)
        ray_dir_y = math.sin(angle_rad)
        
        # 射线检测：检测与屏幕边界的碰撞
        # 计算射线与屏幕边界的交点
        t_x = float('inf')
        t_y = float('inf')
        
        # 与左右边界的交点
        if ray_dir_x > 0:
            t_x = (setting.SCREEN_WIDTH - individual.pos_x) / ray_dir_x
        elif ray_dir_x < 0:
            t_x = (0 - individual.pos_x) / ray_dir_x
        
        # 与上下边界的交点
        if ray_dir_y > 0:
            t_y = (setting.SCREEN_HEIGHT - individual.pos_y) / ray_dir_y
        elif ray_dir_y < 0:
            t_y = (0 - individual.pos_y) / ray_dir_y
        
        # 取最近的交点
        t = min(t_x, t_y)
        
        # 默认距离为最大射线距离
        distance = max_ray_distance
        is_danger = False  # 屏幕边界视为不危险障碍物
        
        # 如果射线方向不为0，计算与屏幕边界的距离
        if t != float('inf'):
            distance = min(t, max_ray_distance)
            
            # 检测与障碍物的碰撞
            if obstacles:
                         
                # 射线与障碍物的碰撞检测
                closest_obstacle_distance = float('inf')
                closest_obstacle = None
                
                for obstacle in obstacles:
                    # 检查障碍物是否激活
                    if obstacle.active:
                        # 射线与矩形碰撞检测算法
                        # 计算射线与矩形边界的交点
                        t_min = 0
                        t_max = distance
                        
                        # 检查x方向
                        if abs(ray_dir_x) < 1e-6:
                            # 射线平行于x轴
                            if individual.pos_x < obstacle.rect.x or individual.pos_x > obstacle.rect.x + obstacle.width:
                                continue  # 射线不在矩形范围内
                        else:
                            # 计算射线与矩形左右边界的交点
                            t1 = (obstacle.rect.x - individual.pos_x) / ray_dir_x
                            t2 = (obstacle.rect.x + obstacle.width - individual.pos_x) / ray_dir_x
                            
                            # 确保t1 <= t2
                            if t1 > t2:
                                t1, t2 = t2, t1
                            
                            # 更新有效区间
                            t_min = max(t_min, t1)
                            t_max = min(t_max, t2)
                            
                            # 检查是否有有效交点
                            if t_min > t_max:
                                continue
                        
                        # 检查y方向
                        if abs(ray_dir_y) < 1e-6:
                            # 射线平行于y轴
                            if individual.pos_y < obstacle.rect.y or individual.pos_y > obstacle.rect.y + obstacle.height:
                                continue  # 射线不在矩形范围内
                        else:
                            # 计算射线与矩形上下边界的交点
                            t1 = (obstacle.rect.y - individual.pos_y) / ray_dir_y
                            t2 = (obstacle.rect.y + obstacle.height - individual.pos_y) / ray_dir_y
                            
                            # 确保t1 <= t2
                            if t1 > t2:
                                t1, t2 = t2, t1
                            
                            # 更新有效区间
                            t_min = max(t_min, t1)
                            t_max = min(t_max, t2)
                            
                            # 检查是否有有效交点
                            if t_min > t_max:
                                continue
                        
                        # 如果有有效交点，更新最近障碍物距离
                        if t_min >= 0 and t_min < closest_obstacle_distance:
                            closest_obstacle_distance = t_min
                            closest_obstacle = obstacle
                
                # 如果检测到障碍物，更新距离和危险程度
                if closest_obstacle:
                    distance = closest_obstacle_distance
                    # 根据障碍物的kill_individual属性判断危险程度
                    is_danger = closest_obstacle.kill_individual
        
        # 使用指数编码方式编码距离，传递障碍物危险度信息
        encoded_value = encode_ray_exponential(distance, max_ray_distance, is_danger)
        inputs.append(encoded_value)
    
    return inputs

def get_input_config():
    """
    获取完整的输入配置信息
    
    返回:
        dict: 包含输入配置信息的字典
    """
    ray_descriptions = []
    for i in range(12):
        angle = i * 30
        ray_descriptions.append(f'射线{i+1} ({angle}°方向障碍物检测, 指数编码)')
    
    return {
        'input_size': get_input_size(),
        'input_description': [
            '目标相对x位置 (tanh归一化到[-1,1])',
            '目标相对y位置 (tanh归一化到[-1,1])'
        ] + ray_descriptions,
        'input_range': '前2个输入使用tanh归一化到[-1,1]范围，后12个射线输入使用指数编码到[-1,1]范围'
    }

# 默认配置
default_input_size = get_input_size()
default_sensor_function = get_sensor_inputs

if __name__ == "__main__":
    # 测试输入配置
    config = get_input_config()
    print("神经网络输入配置:")
    print(f"输入层大小: {config['input_size']}")
    print(f"输入描述: {config['input_description']}")
    print(f"输入范围: {config['input_range']}")