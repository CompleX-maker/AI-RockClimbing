import json
import os
import random
import logging
import numpy as np
from datetime import datetime
from models import ClimbingRule
from flask import current_app  # 修复：使用current_app替代app导入

logger = logging.getLogger(__name__)


class RuleBasedRouteGenerator:
    """基于规则的专业线路生成器"""

    def __init__(self):
        """初始化线路生成器"""
        self.name = "IFSC Level 3 Certified Route Setter"
        self.last_generation_time = datetime.now()
        self.rules = []

        self.load_rules()

    def load_rules(self):
        """加载专业线路设计规则"""
        try:
            # 使用current_app替代app导入
            with current_app.app_context():
                # 从数据库加载规则
                self.rules = ClimbingRule.query.all()
                logger.info(f"✅ 从数据库加载 {len(self.rules)} 条专业规则")

        except Exception as e:
            logger.error(f"❌ 加载规则失败: {str(e)}")
            # 使用默认规则
            self.rules = self._get_default_rules()

    def _get_default_rules(self):
        """获取默认专业规则"""
        default_rules = [
            # 基础规则
            {
                'rule_id': 1,
                'rule_name': '线路宽度规则',
                'rule_description': '每条攀登线路的宽度应不小于1.8m',
                'rule_category': 'safety',
                'rule_parameters': {
                    'min_width': 1.8,
                    'unit': 'm'
                },
                'rule_priority': 1,
                'is_active': True
            },
            # 规则继续...
        ]
        logger.info("⚠️ 使用默认规则集")
        return default_rules

    def generate_route(self, grade_level, available_holds, available_actions):
        """生成专业线路"""
        try:
            logger.info(f"🎯 生成 {grade_level} 难度线路")

            # 基于规则选择岩点
            selected_holds = self._select_holds_by_rules(grade_level, available_holds)

            # 生成岩点位置
            route_holds = self._generate_hold_positions(selected_holds, grade_level)

            # 基于规则选择动作
            selected_actions = self._select_actions_by_rules(grade_level, available_actions, len(route_holds))

            # 生成动作序列
            action_sequence = self._generate_action_sequence(selected_actions, route_holds)

            # 验证线路
            validation_score = self._validate_route(route_holds, action_sequence, grade_level)

            return {
                'holds': route_holds,
                'actions': action_sequence,
                'validation_score': validation_score,
                'difficulty_description': self._get_difficulty_description(grade_level),
                'movement_focus': self._get_movement_focus(grade_level)
            }

        except Exception as e:
            logger.error(f"❌ 线路生成失败: {str(e)}")
            return {
                'error': str(e),
                'holds': [],
                'actions': [],
                'validation_score': 0
            }

    def _select_holds_by_rules(self, grade_level, available_holds):
        """基于规则选择岩点"""
        # 简化版规则应用
        return sorted(available_holds, key=lambda x: abs(x.difficulty_score - grade_level))[:10]

    def _generate_hold_positions(self, holds, grade_level):
        """生成岩点位置"""
        route_holds = []
        for i, hold in enumerate(holds):
            # 位置生成逻辑
            progress = i / max(1, len(holds) - 1)
            x_base = 0.5 + (random.random() - 0.5) * 0.4  # 中心区域
            x_range = 0.3 + grade_level * 0.05
            x = max(0.1, min(0.9, x_base + (random.random() - 0.5) * x_range))

            y_range = 0.7
            y = 0.15 + progress * y_range

            # 岩点类型
            hold_type = "MIDDLE"
            if i == 0:
                hold_type = "START"
                y = 0.1 + random.random() * 0.1
            elif i == len(holds) - 1:
                hold_type = "END"
                y = 0.85 + random.random() * 0.1

            route_holds.append({
                'id': hold.id,
                'name': hold.name,
                'x': x,
                'y': y,
                'type': hold_type,
                'difficulty': f"V{hold.difficulty_score}",
                'shape': hold.shape,
                'size': hold.size,
                'color': '#28a745' if hold_type == 'START' else ('#dc3545' if hold_type == 'END' else '#6f42c1')
            })

        return route_holds

    def _select_actions_by_rules(self, grade_level, available_actions, hold_count):
        """基于规则选择动作"""
        # 简化版规则应用
        return available_actions[:min(hold_count - 1, len(available_actions))]

    def _generate_action_sequence(self, actions, holds):
        """生成动作序列"""
        action_sequence = []
        for i, action in enumerate(actions):
            if i < len(holds) - 1:
                action_sequence.append({
                    'id': action.id,
                    'name': action.name,
                    'category': action.category,
                    'description': action.technical_points or action.common_usage,
                    'difficulty_level': action.difficulty_level,
                    'from_hold': holds[i]['id'],
                    'to_hold': holds[i + 1]['id']
                })

        return action_sequence

    def _validate_route(self, holds, actions, grade_level):
        """验证线路"""
        score = 100

        # 基本验证
        if len(holds) < 4:
            score -= 20
        if len(actions) < len(holds) - 1:
            score -= 15

        # 难度匹配
        avg_difficulty = sum([float(hold['difficulty'][1:]) for hold in holds]) / len(holds)
        difficulty_diff = abs(avg_difficulty - grade_level)
        score -= difficulty_diff * 5

        return max(0, score)

    def _get_difficulty_description(self, grade_level):
        """获取难度描述"""
        descriptions = {
            1: "入门级线路，适合初学者掌握基本技术",
            2: "初级线路，需要基础平衡和协调能力",
            3: "中级线路，包含基础动作序列和平衡控制",
            4: "中高级线路，需要良好技术基础和体力",
            5: "高级线路，包含复杂动作序列和动态技巧",
            6: "专家级线路，需要精湛技术和强健体能",
            7: "大师级线路，仅限高水平运动员挑战",
            8: "宗师级线路，极限技术与体力的结合",
            9: "传奇级线路，接近人体极限的挑战",
            10: "神话级线路，人类攀登能力的终极挑战"
        }
        return descriptions.get(grade_level, f"V{grade_level} 专业线路")

    def _get_movement_focus(self, grade_level):
        """获取动作焦点"""
        focuses = {
            1: ["基础抓握", "简单踩点", "身体平衡"],
            2: ["对角支撑", "重心转移", "基本脚法"],
            3: ["动态抓点", "平衡控制", "精确脚法"],
            4: ["侧拉动作", "反手抓握", "重心控制"],
            5: ["动态技巧", "复杂平衡", "高跨步"],
            6: ["静止点抓取", "垂膝式", "锁膝法"],
            7: ["连续动态", "高难度平衡", "特殊技巧"],
            8: ["极限动态", "复杂动作组合", "高级脚法"],
            9: ["连续静止点", "极限平衡", "完美协调"],
            10: ["传奇技巧", "极限耐力", "完美执行"]
        }
        return focuses.get(grade_level, ["专业技巧", "完美执行"])