from flask_sqlalchemy import SQLAlchemy

# 初始化SQLAlchemy
db = SQLAlchemy()


class WallHold(db.Model):
    __tablename__ = 'wall_holds'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    type = db.Column(db.String(20), nullable=False)
    size_width = db.Column(db.Float)  # 宽度(cm)
    size_thickness = db.Column(db.Float)  # 厚度(cm)
    difficulty_score = db.Column(db.Integer)  # 难度评分
    common_usage = db.Column(db.Text)  # 常见用途
    remark = db.Column(db.Text)  # 备注
    image_path = db.Column(db.String(255))  # 图片路径
    position_x = db.Column(db.Float, default=0.5)  # X坐标 (0-1归一化)
    position_y = db.Column(db.Float, default=0.5)  # Y坐标 (0-1归一化)
    hold_type = db.Column(db.Enum('START', 'END', 'MIDDLE', 'FOOT'), default='MIDDLE')
    shape = db.Column(db.String(20), default='ROUND')  # 岩点形状
    size = db.Column(db.String(10), default='MEDIUM')  # 岩点大小
    color = db.Column(db.String(20), default='#FFD700')  # 岩点颜色
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'size_width': self.size_width,
            'size_thickness': self.size_thickness,
            'difficulty': f"V{self.difficulty_score}" if self.difficulty_score else "V3",
            'position_x': self.position_x,
            'position_y': self.position_y,
            'hold_type': self.hold_type,
            'shape': self.shape,
            'size': self.size,
            'color': self.color
        }


class ClimbingAction(db.Model):
    __tablename__ = 'climbing_actions'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)  # 动作名称
    category = db.Column(db.Enum('basic_grip', 'advanced_grip', 'support_technique', 'foot_technique',
                                 'balance_technique', 'dynamic_movement', 'body_positioning',
                                 'push_pull', 'counter_force', 'hand_sequence', 'crack_climbing'),
                         nullable=False)  # 动作类别
    difficulty_level = db.Column(db.String(20), nullable=False)  # 攀岩难度等级 (V1-V10)
    skill_level_required = db.Column(db.Enum('beginner', 'intermediate', 'advanced', 'expert'),
                                     nullable=False)  # 所需技能等级
    key_muscle_groups = db.Column(db.String(100))  # 关键肌群
    common_usage = db.Column(db.Text)  # 常见用途
    technical_points = db.Column(db.Text)  # 技术要点
    hold_requirements = db.Column(db.Text)  # 岩点要求
    body_position_3d = db.Column(db.JSON)  # 3D身体位置
    transition_moves = db.Column(db.Text)  # 过渡动作
    video_reference = db.Column(db.String(255))  # 视频参考
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category,
            'difficulty_level': self.difficulty_level,
            'skill_level_required': self.skill_level_required,
            'key_muscle_groups': self.key_muscle_groups,
            'common_usage': self.common_usage,
            'technical_points': self.technical_points,
            'body_position_3d': self.body_position_3d
        }


class ClimbingRule(db.Model):
    __tablename__ = 'climbing_rules'
    id = db.Column(db.Integer, primary_key=True)
    rule_name = db.Column(db.String(100), nullable=False)  # 规则名称
    description = db.Column(db.Text, nullable=False)  # 规则描述
    value_min = db.Column(db.DECIMAL(10, 2))  # 最小值
    value_max = db.Column(db.DECIMAL(10, 2))  # 最大值
    unit = db.Column(db.String(20))  # 单位
    rule_type = db.Column(db.Enum('dimension', 'load', 'safety', 'technique'), nullable=False)  # 规则类型
    applicable_grade = db.Column(db.String(20))  # 适用等级
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    def to_dict(self):
        return {
            'id': self.id,
            'rule_name': self.rule_name,
            'description': self.description,
            'value_min': float(self.value_min) if self.value_min else None,
            'value_max': float(self.value_max) if self.value_max else None,
            'unit': self.unit,
            'rule_type': self.rule_type,
            'applicable_grade': self.applicable_grade
        }


class GeneratedRoute(db.Model):
    __tablename__ = 'generated_routes'
    id = db.Column(db.Integer, primary_key=True)
    grade = db.Column(db.String(10), nullable=False)  # 难度等级 V1-V10
    hold_sequence = db.Column(db.JSON)  # 岩点ID序列
    action_sequence = db.Column(db.JSON)  # 动作ID序列
    validation_score = db.Column(db.Float)  # 线路质量评分
    movement_description = db.Column(db.Text)  # 动作描述
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    def to_dict(self):
        return {
            'route_id': self.id,
            'grade': self.grade,
            'hold_sequence': self.hold_sequence,
            'action_sequence': self.action_sequence,
            'validation_score': self.validation_score,
            'movement_description': self.movement_description
        }


class RouteValidation(db.Model):
    __tablename__ = 'route_validations'
    id = db.Column(db.Integer, primary_key=True)
    route_id = db.Column(db.Integer, db.ForeignKey('generated_routes.id'), nullable=False)
    rule_id = db.Column(db.Integer, db.ForeignKey('climbing_rules.id'), nullable=False)
    is_compliant = db.Column(db.Boolean, nullable=False)
    measured_value = db.Column(db.DECIMAL(10, 2))
    notes = db.Column(db.Text)
    validated_at = db.Column(db.DateTime, default=db.func.current_timestamp())