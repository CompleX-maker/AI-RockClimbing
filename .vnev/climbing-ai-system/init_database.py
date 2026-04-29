import mysql.connector
import os
from dotenv import load_dotenv
import pandas as pd
import json
import logging
import numpy as np
import warnings
from datetime import datetime
import re

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 禁用警告
warnings.filterwarnings("ignore", category=Warning)


class DatabaseInitializer:
    def __init__(self):
        self.db_config = {
            'host': os.getenv('MYSQL_HOST', 'localhost'),
            'user': os.getenv('MYSQL_USER', 'root'),
            'password': os.getenv('MYSQL_PASSWORD', 'czl18677288781'),
            'charset': 'utf8mb4',
            'collation': 'utf8mb4_general_ci'
        }
        self.database_name = os.getenv('MYSQL_DB', 'climbing_ai')
        self.connection = None
        self.cursor = None

    def create_database(self):
        """创建数据库"""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            self.cursor = self.connection.cursor()

            # 创建数据库
            self.cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS {self.database_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci")
            self.cursor.execute(f"USE {self.database_name}")

            logger.info(f"✅ 数据库 '{self.database_name}' 创建/验证成功!")
            return True
        except Exception as e:
            logger.error(f"❌ 创建数据库失败: {str(e)}")
            return False

    def create_tables(self):
        """创建所有表"""
        try:
            # 先删除现有表（确保结构正确）
            tables = ["route_validations", "generated_routes", "climbing_rules", "climbing_actions", "wall_holds"]
            for table in tables:
                self.cursor.execute(f"DROP TABLE IF EXISTS {table}")

            # 岩点表
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS wall_holds (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(50) NOT NULL,
                type VARCHAR(20) NOT NULL,
                size_width FLOAT COMMENT '宽度(cm)',
                size_thickness FLOAT COMMENT '厚度(cm)',
                difficulty_score INT COMMENT '难度评分',
                common_usage TEXT COMMENT '常见用途',
                remark TEXT COMMENT '备注',
                image_path VARCHAR(255) COMMENT '图片路径',
                position_x FLOAT DEFAULT 0.5 COMMENT 'X坐标 (0-1归一化)',
                position_y FLOAT DEFAULT 0.5 COMMENT 'Y坐标 (0-1归一化)',
                hold_type ENUM('START', 'END', 'MIDDLE', 'FOOT') DEFAULT 'MIDDLE',
                shape VARCHAR(20) DEFAULT 'ROUND',
                size VARCHAR(10) DEFAULT 'MEDIUM',
                color VARCHAR(20) DEFAULT '#FFD700',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
            """)

            # 攀岩动作表 - 修复difficulty_level类型
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS climbing_actions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL COMMENT '动作名称',
                category ENUM('basic_grip', 'advanced_grip', 'support_technique', 'foot_technique', 
                            'balance_technique', 'dynamic_movement', 'body_positioning', 
                            'push_pull', 'counter_force', 'hand_sequence', 'crack_climbing') NOT NULL COMMENT '动作类别',
                difficulty_level VARCHAR(20) NOT NULL COMMENT '攀岩难度等级 (V1-V10)',
                skill_level_required ENUM('beginner', 'intermediate', 'advanced', 'expert') NOT NULL COMMENT '所需技能等级',
                key_muscle_groups VARCHAR(100) COMMENT '关键肌群',
                common_usage TEXT COMMENT '常见用途',
                technical_points TEXT COMMENT '技术要点',
                hold_requirements TEXT COMMENT '岩点要求',
                body_position_3d JSON COMMENT '3D身体位置',
                transition_moves TEXT COMMENT '过渡动作',
                video_reference VARCHAR(255) COMMENT '视频参考',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
            """)

            # 规则表
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS climbing_rules (
                id INT AUTO_INCREMENT PRIMARY KEY,
                rule_name VARCHAR(100) NOT NULL COMMENT '规则名称',
                description TEXT NOT NULL COMMENT '规则描述',
                value_min DECIMAL(10,2) COMMENT '最小值',
                value_max DECIMAL(10,2) COMMENT '最大值',
                unit VARCHAR(20) COMMENT '单位',
                rule_type ENUM('dimension', 'load', 'safety', 'technique') NOT NULL COMMENT '规则类型',
                applicable_grade VARCHAR(20) COMMENT '适用等级',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
            """)

            # 生成的线路表
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS generated_routes (
                id INT AUTO_INCREMENT PRIMARY KEY,
                grade VARCHAR(10) NOT NULL COMMENT '难度等级 V1-V10',
                hold_sequence JSON COMMENT '岩点ID序列',
                action_sequence JSON COMMENT '动作ID序列',
                validation_score FLOAT COMMENT '线路质量评分',
                movement_description TEXT COMMENT '动作描述',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
            """)

            # 线路验证表
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS route_validations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                route_id INT NOT NULL,
                rule_id INT NOT NULL,
                is_compliant BOOLEAN NOT NULL,
                measured_value DECIMAL(10,2),
                notes TEXT,
                validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (route_id) REFERENCES generated_routes(id),
                FOREIGN KEY (rule_id) REFERENCES climbing_rules(id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
            """)

            self.connection.commit()
            logger.info("✅ 所有表创建成功!")
            return True
        except Exception as e:
            logger.error(f"❌ 创建表失败: {str(e)}")
            return False

    def parse_holds_from_excel(self, file_path):
        """从Excel文件解析岩点数据 - 修复版"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"⚠️ 岩点文件未找到: {file_path}")
                return []

            # 读取Excel文件
            df = pd.read_excel(file_path, sheet_name=0)

            # 清理列名
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

            # 调试信息
            logger.info(f"📊 Excel文件列名: {list(df.columns)}")
            logger.info(f"📊 数据行数: {len(df)}")

            # 确保必要列存在
            required_columns = ['hold_id', 'type', 'size_width(cm)', 'size_thickness(cm)', 'difficulty_score',
                                'common_usage']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.error(f"❌ Excel文件缺少必要列: {missing_columns}")
                return []

            holds_data = []

            for idx, row in df.iterrows():
                try:
                    # 调试信息
                    logger.debug(f"🔍 处理第 {idx} 行数据")

                    # 提取必要字段
                    hold_id = self._safe_int(row.get('hold_id', idx + 1))
                    hold_type = self._safe_string(row.get('type', 'unknown')).lower()
                    name = self._generate_hold_name(hold_type, hold_id)

                    # 尺寸处理
                    width = self._safe_float(row.get('size_width(cm)', 0))
                    thickness = self._safe_float(row.get('size_thickness(cm)', 0))
                    difficulty_score = self._safe_int(row.get('difficulty_score', 1))

                    # 用途和备注
                    common_usage = self._safe_string(row.get('common_usage', ''))
                    remark = self._safe_string(row.get('remark', ''))

                    # 确定岩点类型
                    hold_category = self._categorize_hold_type(hold_type)

                    # 生成位置
                    position_x, position_y = self._generate_hold_position(hold_category, hold_id)

                    holds_data.append((
                        name, hold_type, width, thickness, difficulty_score,
                        common_usage, remark, None,  # image_path
                        position_x, position_y, hold_category,
                        'ROUND', 'MEDIUM', self._get_hold_color(hold_category)
                    ))

                    logger.debug(f"✅ 成功处理岩点行 {idx}: {name}")

                except Exception as e:
                    # 修复：使用字符串连接而不是格式化
                    error_msg = "⚠️ 处理岩点行 " + str(idx) + " 时出错: " + str(e)
                    logger.warning(error_msg)
                    continue

            logger.info(f"✅ 从Excel解析出 {len(holds_data)} 个岩点")
            return holds_data
        except Exception as e:
            logger.error(f"❌ 解析岩点Excel文件失败: {str(e)}")
            return []

    def _safe_string(self, value):
        """安全转换字符串"""
        if pd.isna(value) or value is None:
            return ""
        try:
            cleaned = str(value).strip()
            # 处理可能的格式化字符
            cleaned = cleaned.replace('%', '').replace('$', '').replace(',', '')
            return cleaned
        except:
            return ""

    def _safe_float(self, value):
        """安全转换浮点数"""
        if pd.isna(value) or value is None:
            return 0.0
        try:
            # 处理可能的字符串格式
            value_str = str(value).strip()
            # 移除非数字字符（除了小数点和负号）
            value_str = re.sub(r'[^\d.-]', '', value_str)
            if value_str == '' or value_str == '-':
                return 0.0
            return float(value_str)
        except:
            return 0.0

    def _safe_int(self, value):
        """安全转换整数"""
        if pd.isna(value) or value is None:
            return 0
        try:
            value_str = str(value).strip()
            value_str = re.sub(r'[^\d-]', '', value_str)
            if value_str == '' or value_str == '-':
                return 0
            return int(float(value_str))
        except:
            return 0

    def _generate_hold_name(self, hold_type, hold_id):
        """生成岩点名称"""
        type_map = {
            'jug': '大把手点',
            'mini jug': '小把手点',
            'crimp': '小扣点',
            'sloper': '斜面点',
            'pocket': '指洞点',
            'pinch': '捏点',
            'under cling': '反提点',
            'side pull': '侧拉点',
            'wrap': '包点',
            'volume': '造型点'
        }

        base_name = type_map.get(hold_type.lower(), hold_type)
        return f"{base_name} (ID: {hold_id})"

    def _categorize_hold_type(self, hold_type):
        """分类岩点类型"""
        hold_type = hold_type.lower()

        if 'jug' in hold_type or 'volume' in hold_type:
            return 'START'
        elif 'crimp' in hold_type or 'pocket' in hold_type or 'pinch' in hold_type:
            return 'MIDDLE'
        elif 'under cling' in hold_type or 'side pull' in hold_type:
            return 'MIDDLE'
        elif 'wrap' in hold_type:
            return 'MIDDLE'
        elif 'sloper' in hold_type:
            return 'MIDDLE'
        else:
            return 'MIDDLE'

    def _generate_hold_position(self, hold_category, hold_id):
        """生成岩点位置"""
        # 根据类别生成不同的位置
        if hold_category == 'START':
            return 0.5, 0.1
        elif hold_category == 'END':
            return 0.5, 0.95
        elif hold_category == 'FOOT':
            return 0.5 + (hold_id % 3 - 1) * 0.2, 0.2 + (hold_id // 3) * 0.1
        else:
            # 随机生成中间点位置
            return 0.3 + (hold_id % 5) * 0.15, 0.3 + (hold_id // 5) * 0.15

    def _get_hold_color(self, hold_category):
        """根据类别获取岩点颜色"""
        color_map = {
            'START': '#28a745',
            'END': '#dc3545',
            'FOOT': '#17a2b8',
            'MIDDLE': '#6f42c1'
        }
        return color_map.get(hold_category, '#6f42c1')

    def parse_actions_from_excel(self, file_path):
        """从Excel文件解析动作数据"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"⚠️ 动作文件未找到: {file_path}")
                return []

            # 读取Excel文件
            df = pd.read_excel(file_path, sheet_name=0)

            # 清理列名
            cleaned_columns = []
            for col in df.columns:
                if pd.isna(col) or str(col).strip() == 'null' or str(col).strip() == '':
                    cleaned_columns.append(f'col_{len(cleaned_columns)}')
                else:
                    cleaned_columns.append(
                        str(col).strip().replace(' ', '_').replace('(', '_').replace(')', '_').lower())

            df.columns = cleaned_columns

            # 确保必要列存在
            required_columns = ['move_id', 'category', 'difficulty_level', 'skill_level_required', 'technical_points']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.error(f"❌ Excel文件缺少必要列: {missing_columns}")
                return []

            actions_data = []

            for idx, row in df.iterrows():
                try:
                    # 提取必要字段
                    move_id = self._safe_int(row.get('move_id', idx + 1))
                    category_name = self._safe_string(row.get('category', 'unknown')).lower()
                    difficulty_level = self._safe_string(row.get('difficulty_level', 'V1'))
                    skill_level = self._safe_string(row.get('skill_level_required', 'beginner')).lower()
                    technical_points = self._safe_string(row.get('technical_points', ''))

                    # 生成动作名称
                    name = row.get('动作名称', self._generate_action_name(category_name, move_id))

                    # 确定动作类别
                    action_category = self._map_category_to_enum(category_name)

                    # 提取其他字段
                    muscle_groups = self._safe_string(row.get('key_muscle_groups', ''))
                    common_usage = self._safe_string(row.get('common_usage', ''))
                    hold_requirements = self._safe_string(row.get('hold_requirements', ''))
                    transition_moves = self._safe_string(row.get('transition_moves', ''))

                    # 处理3D位置数据
                    body_position_str = row.get('body_position_3d', '')
                    body_position = None
                    if body_position_str and not pd.isna(body_position_str):
                        try:
                            # 尝试解析JSON
                            body_position = json.loads(str(body_position_str))
                        except:
                            # 创建默认结构
                            body_position = {
                                "center_of_gravity": [0.0, 1.0, 0.3],
                                "hips_angle": "30°"
                            }

                    actions_data.append((
                        name, action_category, difficulty_level, skill_level,
                        muscle_groups, common_usage, technical_points,
                        hold_requirements, json.dumps(body_position) if body_position else None,
                        transition_moves, None  # video_reference
                    ))
                except Exception as e:
                    # 修复：使用字符串连接而不是格式化
                    error_msg = "⚠️ 处理动作行 " + str(idx) + " 时出错: " + str(e)
                    logger.warning(error_msg)
                    continue

            logger.info(f"✅ 从Excel解析出 {len(actions_data)} 个攀岩动作")
            return actions_data
        except Exception as e:
            logger.error(f"❌ 解析动作Excel文件失败: {str(e)}")
            return []

    def _generate_action_name(self, category_name, move_id):
        """生成动作名称"""
        base_names = {
            'dynamic_move': '动态抓点',
            'balance_technique': '平衡控制',
            'footwork': '脚法技术',
            'drop_knee': '垂膝式',
            'heel_hook': '脚跟钩',
            'flagging': '挂旗法',
            'high_step': '高跨步',
            'deadpoint': '静止点',
            'side_pull': '侧拉',
            'gaston': '反撑'
        }

        base_name = base_names.get(category_name.lower(), category_name)
        return f"{base_name} (ID: {move_id})"

    def _map_category_to_enum(self, category_name):
        """映射动作类别到枚举值"""
        category_name = category_name.lower()

        if 'dynamic' in category_name or 'deadpoint' in category_name or 'high_step' in category_name:
            return 'dynamic_movement'
        elif 'balance' in category_name or 'flagging' in category_name:
            return 'balance_technique'
        elif 'foot' in category_name or 'heel_hook' in category_name:
            return 'foot_technique'
        elif 'drop_knee' in category_name or 'gaston' in category_name:
            return 'counter_force'
        elif 'side_pull' in category_name:
            return 'basic_grip'
        elif 'crack' in category_name or 'hand_sequence' in category_name:
            return 'hand_sequence'
        elif 'push_pull' in category_name:
            return 'push_pull'
        elif 'basic_grip' in category_name:
            return 'basic_grip'
        else:
            return 'basic_grip'

    def parse_rules_from_excel(self, file_path):
        """从Excel文件解析规则数据"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"⚠️ 规则文件未找到: {file_path}")
                return []

            # 读取Excel文件
            df = pd.read_excel(file_path, sheet_name=0)

            # 确保数据存在
            if df.empty:
                logger.warning("⚠️ 规则Excel文件为空")
                return []

            rules_data = []
            rule_count = 0

            # 处理每一行规则
            for idx, row in df.iterrows():
                # 获取规则文本 (处理不同格式的Excel)
                rule_text = ""
                for col in df.columns:
                    cell_value = str(row[col]).strip()
                    if cell_value and len(cell_value) > 5:  # 至少5个字符才认为是有效规则
                        rule_text = cell_value
                        break

                if not rule_text:
                    continue

                # 提取规则内容
                rule_name, description, value_min, value_max, unit, rule_type = self._parse_rule_text(rule_text)

                # 确保规则文本不是标题或空行
                if "规则" in rule_text or "说明" in rule_text or len(rule_text) < 10:
                    continue

                rules_data.append((
                    rule_name, description, value_min, value_max, unit, rule_type, None
                ))
                rule_count += 1

            logger.info(f"✅ 从Excel解析出 {rule_count} 条攀岩规则")
            return rules_data
        except Exception as e:
            logger.error(f"❌ 解析规则Excel文件失败: {str(e)}")
            return []

    def _parse_rule_text(self, rule_text):
        """解析规则文本"""
        rule_text = rule_text.strip()

        # 提取数值
        value_min = None
        value_max = None
        unit = None

        # 查找数字和单位
        number_matches = re.findall(r'(\d+\.?\d*)\s*(kN|m|kg|cm|mm)', rule_text)
        for num_str, unit_str in number_matches:
            try:
                num = float(num_str)
                if '不小于' in rule_text or '不应小于' in rule_text or '不少于' in rule_text:
                    value_min = num
                    unit = unit_str
                elif '不大于' in rule_text or '不应大于' in rule_text or '不超过' in rule_text:
                    value_max = num
                    unit = unit_str
                elif '等于' in rule_text:
                    value_min = num
                    value_max = num
                    unit = unit_str
            except:
                pass

        # 确定规则类型
        rule_type = 'safety'
        if '宽度' in rule_text or '高度' in rule_text or '尺寸' in rule_text:
            rule_type = 'dimension'
        elif '承载力' in rule_text or '载荷' in rule_text or '抗拉力' in rule_text or 'kN' in rule_text:
            rule_type = 'load'
        elif '技术' in rule_text or '动作' in rule_text:
            rule_type = 'technique'

        # 生成规则名称
        rule_name = "安全规则"
        if '宽度' in rule_text:
            rule_name = "线路宽度规则"
        elif '承载力' in rule_text:
            rule_name = "保护系统承载力规则"
        elif '高度' in rule_text:
            rule_name = "岩壁高度规则"
        elif '载荷' in rule_text:
            rule_name = "岩板载荷规则"

        return rule_name, rule_text, value_min, value_max, unit, rule_type

    def import_holds(self, holds_data):
        """导入岩点数据到数据库"""
        if not holds_data:
            logger.warning("⚠️ 没有岩点数据可导入")
            return False

        try:
            # 清除现有数据
            self.cursor.execute("DELETE FROM wall_holds")

            # 插入新数据
            self.cursor.executemany("""
            INSERT INTO wall_holds 
            (name, type, size_width, size_thickness, difficulty_score, common_usage, 
             remark, image_path, position_x, position_y, hold_type, shape, size, color)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, holds_data)

            self.connection.commit()
            logger.info(f"✅ 成功导入 {len(holds_data)} 个岩点到数据库")
            return True
        except Exception as e:
            logger.error(f"❌ 导入岩点数据失败: {str(e)}")
            return False

    def import_actions(self, actions_data):
        """导入动作数据到数据库"""
        if not actions_data:
            logger.warning("⚠️ 没有动作数据可导入")
            return False

        try:
            # 清除现有数据
            self.cursor.execute("DELETE FROM climbing_actions")

            # 插入新数据
            self.cursor.executemany("""
            INSERT INTO climbing_actions 
            (name, category, difficulty_level, skill_level_required, key_muscle_groups,
             common_usage, technical_points, hold_requirements, body_position_3d,
             transition_moves, video_reference)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, actions_data)

            self.connection.commit()
            logger.info(f"✅ 成功导入 {len(actions_data)} 个攀岩动作到数据库")
            return True
        except Exception as e:
            logger.error(f"❌ 导入动作数据失败: {str(e)}")
            return False

    def import_rules(self, rules_data):
        """导入规则数据到数据库"""
        if not rules_data:
            logger.warning("⚠️ 没有规则数据可导入")
            return False

        try:
            # 清除现有数据
            self.cursor.execute("DELETE FROM climbing_rules")

            # 插入新数据
            self.cursor.executemany("""
            INSERT INTO climbing_rules 
            (rule_name, description, value_min, value_max, unit, rule_type, applicable_grade)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, rules_data)

            self.connection.commit()
            logger.info(f"✅ 成功导入 {len(rules_data)} 条攀岩规则到数据库")
            return True
        except Exception as e:
            logger.error(f"❌ 导入规则数据失败: {str(e)}")
            return False

    def generate_sample_positions(self):
        """为岩点生成示例位置（基于规则）"""
        try:
            self.cursor.execute("SELECT id, hold_type FROM wall_holds")
            holds = self.cursor.fetchall()

            updates = []
            for hold_id, hold_type in holds:
                if hold_type == 'START':
                    position_x = 0.5
                    position_y = 0.1
                elif hold_type == 'END':
                    position_x = 0.5
                    position_y = 0.95
                elif hold_type == 'FOOT':
                    position_x = 0.5 + (hold_id % 3 - 1) * 0.2
                    position_y = 0.2 + (hold_id // 3) * 0.1
                else:
                    # 中间点
                    position_x = 0.3 + (hold_id % 5) * 0.1
                    position_y = 0.3 + (hold_id // 5) * 0.1

                updates.append((position_x, position_y, hold_id))

            if updates:
                self.cursor.executemany("""
                UPDATE wall_holds 
                SET position_x = %s, position_y = %s 
                WHERE id = %s
                """, updates)
                self.connection.commit()
                logger.info(f"🔄 为 {len(updates)} 个岩点生成了示例位置")

            return True
        except Exception as e:
            logger.error(f"❌ 生成岩点位置失败: {str(e)}")
            return False

    def close_connection(self):
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def initialize_database(self):
        """初始化整个数据库"""
        try:
            # 1. 创建数据库
            if not self.create_database():
                return False

            # 2. 创建表
            if not self.create_tables():
                return False

            # 3. 导入岩点数据
            holds_file = os.path.join(os.path.dirname(__file__), 'hold.xlsx')
            holds_data = self.parse_holds_from_excel(holds_file)
            if holds_data:
                self.import_holds(holds_data)

            # 4. 导入动作数据
            actions_file = os.path.join(os.path.dirname(__file__), 'action.xlsx')
            actions_data = self.parse_actions_from_excel(actions_file)
            if actions_data:
                self.import_actions(actions_data)

            # 5. 导入规则数据
            rules_file = os.path.join(os.path.dirname(__file__), 'rule.xlsx')
            rules_data = self.parse_rules_from_excel(rules_file)
            if rules_data:
                self.import_rules(rules_data)
            else:
                # 如果规则文件解析失败，使用硬编码规则
                logger.warning("⚠️ 规则文件解析失败，使用默认规则")
                default_rules = [
                    ("线路宽度规则", "每条攀登线路的宽度应不小于1.8m。", 1.8, None, "m", "dimension",
                     "V1,V2,V3,V4,V5,V6,V7,V8,V9,V10"),
                    ("保护系统承载力规则", "每个顶端保护系统承载力应不小于8kN。", 8.0, None, "kN", "load",
                     "V1,V2,V3,V4,V5,V6,V7,V8,V9,V10"),
                    ("保护挂片规则", "每个保护挂片应与结构直接链接，且承载力不小于8kN。", 8.0, None, "kN", "load",
                     "V1,V2,V3,V4,V5,V6,V7,V8,V9,V10"),
                    ("岩板静载荷规则", "岩板耐受静载荷应不小于4kN。", 4.0, None, "kN", "load",
                     "V1,V2,V3,V4,V5,V6,V7,V8,V9,V10"),
                    ("岩板动载荷规则", "岩板的耐受动载荷应不小于6kN。", 6.0, None, "kN", "load",
                     "V1,V2,V3,V4,V5,V6,V7,V8,V9,V10"),
                    ("支点孔抗拉力规则", "支点孔抗拉力应不小于3kN。", 3.0, None, "kN", "load",
                     "V1,V2,V3,V4,V5,V6,V7,V8,V9,V10"),
                    ("岩壁高度规则", "用于攀石活动的人工岩壁有效垂直高度应不超过5m。", None, 5.0, "m", "dimension",
                     "V1,V2,V3,V4,V5,V6,V7,V8,V9,V10")
                ]
                self.import_rules(default_rules)

            # 6. 生成示例位置
            self.generate_sample_positions()

            logger.info("🎉 数据库初始化完成!")
            return True
        except Exception as e:
            logger.error(f"❌ 数据库初始化失败: {str(e)}")
            return False


if __name__ == "__main__":
    # 初始化数据库
    db_init = DatabaseInitializer()
    success = db_init.initialize_database()
    db_init.close_connection()

    if success:
        print("\n" + "=" * 60)
        print("✅ 数据库初始化成功!")
        print("📋 创建的表:")
        print("   - wall_holds (岩点表)")
        print("   - climbing_actions (攀岩动作表)")
        print("   - climbing_rules (规则表)")
        print("   - generated_routes (生成的线路表)")
        print("   - route_validations (线路验证表)")
        print("\n🔍 使用说明:")
        print("1. 在MySQL Workbench中，展开 'climbing_ai' 数据库查看表")
        print("2. 右键点击表名，选择 'Select Rows - Limit 1000' 查看数据")
        print("3. 使用 'Table Inspector' 查看表结构")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 数据库初始化失败!")
        print("💡 请检查:")
        print(f"   - MySQL 服务是否运行")
        print(f"   - 用户名/密码是否正确 (root/czl18677288781)")
        print(f"   - 文件路径是否正确 (hold.xlsx, action.xlsx, rule.xlsx)")
        print(f"   - 是否已安装必要的Python包 (pandas, openpyxl)")
        print("=" * 60)