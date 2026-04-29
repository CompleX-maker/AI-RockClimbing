from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file, make_response, Response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
import re
import logging
import time
import json
import random
import traceback
import numpy as np
import cv2
import mimetypes
import threading
import queue
from werkzeug.utils import secure_filename
from datetime import datetime
from sqlalchemy import text

# 设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 300MB
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///climbing.db'  # 本地SQLite数据库
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app, supports_credentials=True)

# 初始化数据库
db = SQLAlchemy(app)

# 确保上传目录存在
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# 数据库模型定义
class WallHold(db.Model):
    __tablename__ = 'wall_hold'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(50), nullable=True)
    difficulty_score = db.Column(db.Float, nullable=True)
    shape = db.Column(db.String(50), nullable=True)
    size = db.Column(db.String(50), nullable=True)

    def __repr__(self):
        return f'<WallHold {self.name}>'


class ClimbingAction(db.Model):
    __tablename__ = 'climbing_action'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), nullable=True)
    difficulty_level = db.Column(db.String(20), nullable=True)
    description = db.Column(db.Text, nullable=True)
    technical_points = db.Column(db.Text, nullable=True)
    body_position_3d = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f'<ClimbingAction {self.name}>'


class ClimbingRule(db.Model):
    __tablename__ = 'climbing_rule'
    id = db.Column(db.Integer, primary_key=True)
    rule_name = db.Column(db.String(100), nullable=False)
    rule_description = db.Column(db.Text, nullable=True)
    rule_category = db.Column(db.String(50), nullable=True)
    rule_parameters = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f'<ClimbingRule {self.rule_name}>'


class GeneratedRoute(db.Model):
    __tablename__ = 'generated_route'
    id = db.Column(db.Integer, primary_key=True)
    route_id = db.Column(db.String(50), unique=True, nullable=False)
    grade = db.Column(db.String(10), nullable=False)
    difficulty_description = db.Column(db.Text, nullable=True)
    movement_focus = db.Column(db.Text, nullable=True)
    hold_sequence = db.Column(db.Text, nullable=True)
    action_sequence = db.Column(db.Text, nullable=True)
    validation_score = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<GeneratedRoute {self.route_id}>'


# 处理结果字典
processing_results = {}


# 视频处理器
def process_video_async(video_path: str, task_id: str) -> None:
    """异步处理视频，分析整个视频的动作"""
    try:
        logger.info(f"🎬 开始处理视频: {video_path}")

        # 使用OpenCV打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件，可能格式不受支持")

        # 获取视频基本信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps if fps > 0 else 0

        logger.info(f"📊 视频信息: {fps:.1f} FPS, {total_frames} 帧, 时长: {total_duration:.1f}秒")

        # 初始化结果
        detected_actions = []
        current_action = None

        # 处理所有帧
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress = frame_count / total_frames * 100

            # 更新处理进度
            if task_id in processing_results:
                processing_results[task_id]['progress'] = progress

            # 每5帧处理一次
            if frame_count % 5 == 0:
                # 模拟动作检测
                action_detected = False
                action_name = None

                # 用更合理的规则检测动作：每80-150帧开始一个新动作
                if current_action is None and frame_count % random.randint(80,
                                                                           150) == 0 and frame_count < total_frames - 50:
                    action_detected = True
                    # 从数据库中获取动作名称
                    with app.app_context():
                        # 获取所有动作名称
                        all_actions = ClimbingAction.query.all()
                        if all_actions:
                            action = random.choice(all_actions)
                            action_name = action.name
                        else:
                            # 如果数据库中没有动作，使用默认动作列表
                            action_name = random.choice([
                                "动态抓点", "静态平衡", "侧拉动作",
                                "精确脚法", "高跨步", "蹲跳",
                                "垂膝式", "侧拉", "脚跟挂"
                            ])

                # 如果检测到新动作
                if action_detected and current_action is None:
                    # 从数据库获取完整动作信息
                    with app.app_context():
                        action = ClimbingAction.query.filter_by(name=action_name).first()
                        if action:
                            confidence = random.uniform(0.7, 0.95)
                            timestamp = frame_count / fps

                            # 开始新动作
                            current_action = {
                                'action': action.name,
                                'confidence': confidence,
                                'start_frame': frame_count,
                                'start_time': timestamp,
                                'end_frame': frame_count,
                                'end_time': timestamp,
                                'max_confidence': confidence,
                                'min_confidence': confidence,
                                'avg_confidence': confidence,
                                'description': f"{action.name}技术，置信度: {confidence:.2f}",
                                'category': action.category,
                                'difficulty_level': action.difficulty_level,
                                'technical_points': action.technical_points,
                                'body_position_3d': action.body_position_3d or json.dumps({
                                    "head": {"x": 0.5, "y": 0.2, "size": 0.05},
                                    "neck": {"x": 0.5, "y": 0.25},
                                    "shoulder_left": {"x": 0.4, "y": 0.3},
                                    "shoulder_right": {"x": 0.6, "y": 0.3},
                                    "elbow_left": {"x": 0.3, "y": 0.4},
                                    "elbow_right": {"x": 0.7, "y": 0.4},
                                    "wrist_left": {"x": 0.25, "y": 0.5},
                                    "wrist_right": {"x": 0.75, "y": 0.5},
                                    "hip_left": {"x": 0.45, "y": 0.6},
                                    "hip_right": {"x": 0.55, "y": 0.6},
                                    "knee_left": {"x": 0.4, "y": 0.75},
                                    "knee_right": {"x": 0.6, "y": 0.75},
                                    "ankle_left": {"x": 0.35, "y": 0.9},
                                    "ankle_right": {"x": 0.65, "y": 0.9}
                                })
                            }
                        else:
                            # 如果数据库中没有该动作，使用默认信息
                            confidence = random.uniform(0.7, 0.95)
                            timestamp = frame_count / fps

                            current_action = {
                                'action': action_name,
                                'confidence': confidence,
                                'start_frame': frame_count,
                                'start_time': timestamp,
                                'end_frame': frame_count,
                                'end_time': timestamp,
                                'max_confidence': confidence,
                                'min_confidence': confidence,
                                'avg_confidence': confidence,
                                'description': f"{action_name}技术，置信度: {confidence:.2f}"
                            }

                # 持续跟踪当前动作
                if current_action:
                    # 更新动作的结束时间和置信度
                    current_action['end_frame'] = frame_count
                    current_action['end_time'] = frame_count / fps

                    # 更新置信度
                    confidence = random.uniform(0.5, 0.95)
                    current_action['avg_confidence'] = (current_action['avg_confidence'] * 0.7 + confidence * 0.3)
                    current_action['max_confidence'] = max(current_action['max_confidence'], confidence)
                    current_action['min_confidence'] = min(current_action['min_confidence'], confidence)

                    # 动作持续时间超过1.5秒或置信度低于阈值，记录并重置
                    duration = (frame_count - current_action['start_frame']) / fps
                    if duration > 1.5 or current_action['min_confidence'] < 0.6:
                        # 确保动作持续时间至少0.5秒
                        if duration > 0.5:
                            detected_actions.append(current_action)
                        current_action = None

            # 每50帧记录一次日志
            if frame_count % 50 == 0:
                logger.info(f"⏳ 处理进度: {progress:.1f}% ({frame_count}/{total_frames}帧)")

        # 处理剩余的动作
        if current_action:
            duration = (frame_count - current_action['start_frame']) / fps
            if duration > 0.5:
                detected_actions.append(current_action)

        # 保存结果
        if task_id in processing_results:
            processing_results[task_id].update({
                'status': 'completed',
                'progress': 100,
                'actions': detected_actions,
                'video_duration': total_duration,
                'completed_time': time.time(),
                'total_actions': len(detected_actions)
            })
            logger.info(f"✅ 视频处理完成! 共检测到 {len(detected_actions)} 个动作")
        else:
            logger.warning(f"⚠️ 任务 {task_id} 不再存在，无法更新结果")

    except Exception as e:
        logger.error(f"❌ 视频分析失败: {str(e)}\n{traceback.format_exc()}")
        if task_id in processing_results:
            processing_results[task_id].update({
                'status': 'error',
                'error': str(e),
                'progress': 0,
                'total_actions': 0
            })


@app.route('/')
def index():
    """系统主页"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    """视频上传 - 动作分析版"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': '未检测到视频文件'}), 400

        video = request.files['video']
        if video.filename == '':
            return jsonify({'error': '未选择文件'}), 400

        # 验证文件类型
        allowed_extensions = {'mp4', 'avi', 'mov', 'mkv'}
        filename = secure_filename(video.filename)
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

        if file_ext not in allowed_extensions:
            return jsonify({'error': f'不支持的视频格式，仅支持: {", ".join(allowed_extensions)}'}), 400

        # 保存文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"analysis_{timestamp}.{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)

        video.save(filepath)

        # 验证文件是否保存成功
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            if os.path.exists(filepath):
                os.remove(filepath)
            logger.error(f"❌ 保存视频文件失败: {filepath}")
            return jsonify({'error': '视频文件保存失败'}), 500

        # 生成可访问的URL
        video_url = url_for('static', filename=f'uploads/{safe_filename}', _external=True)

        # 创建唯一任务ID
        task_id = f"task_{timestamp}_{random.randint(1000, 9999)}"

        # 将任务添加到处理结果字典
        processing_results[task_id] = {
            'status': 'uploaded',
            'progress': 0,
            'actions': [],
            'video_url': video_url,
            'start_time': time.time(),
            'video_path': filepath,
            'total_actions': 0
        }

        logger.info(f"✅ 视频上传成功: {video_url}")

        return jsonify({
            'message': '视频上传成功',
            'task_id': task_id,
            'video_url': video_url,
            'status_url': url_for('get_analysis_status', task_id=task_id, _external=True)
        }), 200

    except Exception as e:
        logger.error(f"❌ 视频上传失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500


@app.route('/start-analysis', methods=['POST'])
def start_analysis():
    """开始分析 - 动作分析版"""
    try:
        data = request.get_json()
        if not data or 'task_id' not in data:
            return jsonify({'error': '缺少任务ID'}), 400

        task_id = data['task_id']

        # 验证任务是否存在
        if task_id not in processing_results:
            return jsonify({'error': '任务ID无效'}), 404

        # 重置处理状态
        processing_results[task_id].update({
            'status': 'processing',
            'progress': 0,
            'actions': [],
            'start_time': time.time()
        })

        # 启动后台处理线程
        threading.Thread(
            target=process_video_async,
            args=(processing_results[task_id]['video_path'], task_id),
            daemon=True,
            name=f"AnalysisThread-{task_id}"
        ).start()

        return jsonify({
            'message': '分析任务已提交',
            'task_id': task_id,
            'status_url': url_for('get_analysis_status', task_id=task_id, _external=True)
        }), 200

    except Exception as e:
        logger.error(f"❌ 开始分析失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500


@app.route('/analysis-status/<task_id>', methods=['GET'])
def get_analysis_status(task_id):
    """获取分析状态 - 动作分析版"""
    try:
        if task_id not in processing_results:
            return jsonify({'error': '任务不存在'}), 404

        result = processing_results[task_id]

        # 检查是否超时
        if time.time() - result['start_time'] > 300:  # 5分钟超时
            result['status'] = 'timeout'
            result['error'] = '分析超时，请重试'

        # 确保进度不超过100%
        if result['progress'] > 100:
            result['progress'] = 100

        result_data = {
            'status': result['status'],
            'progress': result['progress'],
            'video_url': result.get('video_url', ''),
            'is_complete': result['status'] == 'completed',
            'has_error': 'error' in result,
            'error': result.get('error', ''),
            'actions': result.get('actions', []),
            'total_actions': result.get('total_actions', 0),
            'video_duration': result.get('video_duration', 0)
        }

        # 只在完成时返回动作和姿态数据，减小响应体
        if result['status'] == 'completed':
            result_data['pose_data'] = result.get('pose_data', [])

        return jsonify(result_data), 200

    except Exception as e:
        logger.error(f"❌ 获取状态失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500


@app.route('/generate-route', methods=['POST'])
def generate_route():
    """生成线路 - 修复线路生成逻辑"""
    try:
        data = request.get_json()
        if not data or 'grade' not in data:
            return jsonify({'error': '缺少grade字段'}), 400

        grade = data['grade']

        # 验证难度等级
        valid_grades = [f"V{i}" for i in range(1, 11)]
        if grade not in valid_grades:
            return jsonify({'error': f'无效难度等级，支持: {", ".join(valid_grades)}'}), 400

        grade_level = int(grade[1:])

        # 生成随机线路
        hold_count = max(4, min(10, 8 - grade_level // 2))
        route_holds = []

        # 获取岩点数据
        with app.app_context():
            # 获取适配该难度的岩点
            available_holds = WallHold.query.filter(
                WallHold.difficulty_score.between(max(1, grade_level - 2), min(10, grade_level + 2))
            ).all()

            logger.info(f"🔍 从数据库检索到 {len(available_holds)} 个难度匹配的岩点")

            if not available_holds:
                logger.warning(f"⚠️ 无可用难度 {grade} 的岩点，使用随机岩点")
                available_holds = WallHold.query.order_by(db.func.random()).limit(10).all()

            # 生成岩点序列
            selected_holds = random.sample(available_holds, min(hold_count, len(available_holds)))

            for i, hold in enumerate(selected_holds):
                progress = i / (hold_count - 1) if hold_count > 1 else 0.5

                # 随机化位置
                x_range = 0.3 + (grade_level * 0.02) + (random.random() * 0.1)
                x_base = 0.5 + (random.random() - 0.5) * 0.2
                x_offset = (random.random() - 0.5) * x_range
                x = max(0.1, min(0.9, x_base + x_offset))

                y_range = 0.7
                y = 0.15 + (progress * y_range)

                # 岩点类型
                hold_type = "MIDDLE"
                if i == 0:
                    hold_type = "START"
                    y = 0.1 + random.random() * 0.1
                elif i == hold_count - 1:
                    hold_type = "END"
                    y = 0.85 + random.random() * 0.1

                route_holds.append({
                    'id': hold.id,
                    'name': hold.name,
                    'x': x,
                    'y': y,
                    'type': hold_type,
                    'difficulty': grade,
                    'shape': hold.shape or 'round',
                    'size': hold.size or 'medium',
                    'color': '#28a745' if hold_type == 'START' else ('#dc3545' if hold_type == 'END' else '#6f42c1')
                })

        # 生成随机动作
        action_count = max(2, hold_count - 2)
        actions = []

        # 获取适配难度的动作数据
        with app.app_context():
            # 获取所有难度等级
            all_actions = ClimbingAction.query.all()

            # 筛选出难度等级小于或等于当前等级的动作
            available_actions = []
            for action in all_actions:
                # 尝试解析难度等级
                if action.difficulty_level and action.difficulty_level.startswith('V'):
                    try:
                        action_level = int(action.difficulty_level[1:])
                        if action_level <= grade_level:
                            available_actions.append(action)
                    except:
                        # 如果解析失败，假定为高级动作
                        if grade_level >= 5:
                            available_actions.append(action)

            logger.info(f"🔍 从数据库检索到 {len(available_actions)} 个难度匹配的动作")

            if not available_actions:
                logger.warning(f"⚠️ 无可用难度 {grade} 的动作，使用随机动作")
                available_actions = ClimbingAction.query.order_by(db.func.random()).limit(20).all()

            # 生成随机动作序列
            selected_actions = random.sample(available_actions, min(action_count, len(available_actions)))

            for i, action in enumerate(selected_actions):
                # 解析难度等级（确保显示正确的难度）
                display_difficulty = action.difficulty_level if action.difficulty_level else f"V{grade_level}"

                actions.append({
                    'id': action.id,
                    'name': action.name,
                    'category': action.category or '基础抓法',
                    'description': action.description or '描述未定义',
                    'difficulty_level': display_difficulty,
                    'technical_points': action.technical_points or '技术要点: 未定义'
                })

        route_data = {
            'route_id': datetime.now().strftime('%Y%m%d%H%M%S'),
            'grade': grade,
            'holds': route_holds,
            'actions': actions,
            'difficulty_description': f"{grade} 难度线路 - 随机生成",
            'movement_focus': ['动态技巧', '平衡控制', '脚法技术'][:len(actions) % 3 + 1]
        }

        return jsonify(route_data)

    except Exception as e:
        logger.error(f"❌ 线路生成失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'线路生成失败: {str(e)}'}), 500


@app.route('/static/uploads/<path:filename>')
def serve_video(filename):
    """服务视频文件 - 支持范围请求"""
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # 验证文件存在
    if not os.path.exists(video_path):
        return "File not found", 404

    # 检查是否为范围请求
    range_header = request.headers.get('Range', None)

    if range_header:
        # 获取文件大小
        file_size = os.path.getsize(video_path)

        # 处理范围请求
        byte1, byte2 = 0, None
        m = re.search('bytes=(\d+)-(\d*)', range_header)

        if m:
            byte1 = int(m.group(1))
            if m.group(2):
                byte2 = int(m.group(2))

        # 计算要发送的字节数
        length = file_size - byte1
        if byte2 is not None:
            length = byte2 - byte1 + 1

        # 创建响应
        try:
            with open(video_path, 'rb') as f:
                f.seek(byte1)
                data = f.read(length)

            rv = Response(data, 206, mimetype='video/mp4')
            rv.headers.add('Content-Range', f'bytes {byte1}-{byte1 + length - 1}/{file_size}')
            rv.headers.add('Accept-Ranges', 'bytes')
            rv.headers.add('Content-Length', str(length))

            return rv
        except Exception as e:
            logger.error(f"❌ 范围请求处理失败: {str(e)}")
            return send_file(video_path, mimetype='video/mp4')
    else:
        # 完整文件请求
        return send_file(video_path, mimetype='video/mp4')


def init_database():
    """初始化数据库"""
    try:
        with app.app_context():
            # 创建数据库表
            db.create_all()

            # 检查是否需要初始化测试数据
            if WallHold.query.count() == 0:
                logger.info("🔄 初始化测试岩点数据...")
                # 添加测试岩点
                test_holds = [
                    {'name': '大把手点', 'type': 'JUG', 'difficulty_score': 1.0, 'shape': 'round', 'size': 'large'},
                    {'name': '小把手点', 'type': 'MINI_JUG', 'difficulty_score': 2.0, 'shape': 'round',
                     'size': 'medium'},
                    {'name': '小扣点', 'type': 'CRIMP', 'difficulty_score': 4.0, 'shape': 'edge', 'size': 'small'},
                    {'name': '斜面点', 'type': 'SLOPER', 'difficulty_score': 5.0, 'shape': 'sloped', 'size': 'medium'},
                    {'name': '指洞点', 'type': 'POCKET', 'difficulty_score': 4.0, 'shape': 'hole', 'size': 'small'},
                    {'name': '捏点', 'type': 'PINCH', 'difficulty_score': 4.0, 'shape': 'pinch', 'size': 'small'},
                    {'name': '反提点', 'type': 'UNDERCLING', 'difficulty_score': 3.0, 'shape': 'under',
                     'size': 'small'},
                    {'name': '侧拉点', 'type': 'SIDEPULL', 'difficulty_score': 3.0, 'shape': 'side', 'size': 'medium'},
                    {'name': '包点', 'type': 'WRAP', 'difficulty_score': 4.0, 'shape': 'wrap', 'size': 'medium'},
                    {'name': '造型点', 'type': 'VOLUME', 'difficulty_score': 1.0, 'shape': 'volume', 'size': 'large'}
                ]

                for hold_data in test_holds:
                    hold = WallHold(**hold_data)
                    db.session.add(hold)

                db.session.commit()
                logger.info("✅ 测试岩点数据初始化完成")

            # 检查是否需要初始化测试动作
            if ClimbingAction.query.count() == 0:
                logger.info("🔄 初始化测试动作数据...")
                # 从知识库添加测试动作
                default_actions = [
                    # 1. 蹲跳(Lunge)
                    {'name': '蹲跳(Lunge)', 'category': 'Dynamic_Move', 'difficulty_level': 'V3',
                     'description': '跨越大距离岩点',
                     'technical_points': '重心下沉蓄力，手臂摆动制造反作用力，跳到最高点瞬间抓点',
                     'body_position_3d': '{"center_of_gravity": [0.15, 1.2, 0.35], "hips_angle": 45, "arms_extension": 80}'},
                    # 2. 对角支撑
                    {'name': '对角支撑', 'category': 'Balance_Technique', 'difficulty_level': 'V1',
                     'description': '保持稳定平衡位置',
                     'technical_points': '通过对角支撑分配体重，腰贴近岩壁减少手部负担',
                     'body_position_3d': '{"center_of_gravity": [0.05, 1.0, 0.25], "hips_angle": 15, "weight_distribution": "70% feet, 30% hands"}'},
                    # 3. 正踩(Front Edge)
                    {'name': '正踩(Front Edge)', 'category': 'Footwork_Basic', 'difficulty_level': 'V2',
                     'description': '精确踩点转移体重',
                     'technical_points': '拇指球位置踩点，脚跟抬起增加摩擦力，视线先于脚到达',
                     'body_position_3d': '{"center_of_gravity": [0.1, 0.9, 0.3], "hips_angle": 20, "foot_placement": "precision"}'},
                    # 4. 垂膝式(Drop Knee)
                    {'name': '垂膝式(Drop Knee)', 'category': 'Counter_Force', 'difficulty_level': 'V4',
                     'description': '制造反作用力稳定身体',
                     'technical_points': '后脚膝盖向下旋转，身体旋转配合伸手，双脚前后蹬产生支撑力',
                     'body_position_3d': '{"center_of_gravity": [-0.05, 1.1, 0.4], "hips_angle": 60, "knee_rotation": 45}'},
                    # 5. 脚跟挂(Heel Hook)
                    {'name': '脚跟挂(Heel Hook)', 'category': 'Footwork_Hooking', 'difficulty_level': 'V3',
                     'description': '克服仰角岩壁',
                     'technical_points': '脚跟挂点后身体重心向挂脚侧倾斜，利用脚部下拉减轻手部负担',
                     'body_position_3d': '{"center_of_gravity": [0.2, 1.3, 0.45], "hips_angle": 30, "leg_extension": 70}'},
                    # 6. 挂旗法(Flagging)
                    {'name': '挂旗法(Flagging)', 'category': 'Balance_Technique', 'difficulty_level': 'V2',
                     'description': '防止身体旋转', 'technical_points': '非踩脚侧腿向外伸展作为配重，腰往支撑手方向收',
                     'body_position_3d': '{"center_of_gravity": [0.08, 1.1, 0.28], "hips_angle": 25, "flagging_leg": "extended"}'},
                    # 7. 高跨步(High Step)
                    {'name': '高跨步(High Step)', 'category': 'Push_Pull', 'difficulty_level': 'V2',
                     'description': '抬高脚位增加高度',
                     'technical_points': '重心完全转移到高脚上后再伸手，膝盖向下身体贴近岩壁',
                     'body_position_3d': '{"center_of_gravity": [0.02, 1.4, 0.22], "hips_angle": 10, "foot_elevation": "high"}'},
                    # 8. 静止点(Deadpoint)
                    {'name': '静止点(Deadpoint)', 'category': 'Dynamic_Move', 'difficulty_level': 'V5',
                     'description': '精确抓取远点',
                     'technical_points': '双手拉近岩壁创造无重力瞬间，把握精确时机伸手抓点',
                     'body_position_3d': '{"center_of_gravity": [0.25, 1.8, 0.5], "hips_angle": 75, "trajectory": "arc"}'},
                    # 9. 侧拉(Side Pull)
                    {'name': '侧拉(Side Pull)', 'category': 'Hand_Grip', 'difficulty_level': 'V3',
                     'description': '抓取侧面岩点', 'technical_points': '身体侧向岩壁，创造反作用力，脚部配合推力',
                     'body_position_3d': '{"center_of_gravity": [-0.15, 1.2, 0.35], "hips_angle": 45, "body_rotation": 30}'},
                    # 10. 反撑(Gaston)
                    {'name': '反撑(Gaston)', 'category': 'Hand_Grip', 'difficulty_level': 'V4',
                     'description': '抓取反手点',
                     'technical_points': '拇指向下支撑，身体贴近岩壁，利用腰部旋转增加支撑力',
                     'body_position_3d': '{"center_of_gravity": [-0.2, 1.1, 0.4], "hips_angle": 50, "elbow_position": "extended"}'},
                    # 11. 扣/封闭式抓法(Full Crimp)
                    {'name': '扣/封闭式抓法(Full Crimp)', 'category': 'Hand_Grip', 'difficulty_level': 'V4',
                     'description': '抓取极小岩缘', 'technical_points': '拇指紧扣在食指上增加支撑力，皮肤固定减少滑动',
                     'body_position_3d': '{"center_of_gravity": [-0.1, 1.0, 0.3], "elbow_angle": 120, "finger_position": "extended"}'},
                    # 12. 搭/半开式抓法(Half Crimp)
                    {'name': '搭/半开式抓法(Half Crimp)', 'category': 'Hand_Grip', 'difficulty_level': 'V2',
                     'description': '抓取平坦岩点', 'technical_points': '拇指压在食指上或贴在侧边强化支撑力',
                     'body_position_3d': '{"center_of_gravity": [-0.05, 0.95, 0.25], "elbow_angle": 135, "thumb_position": "pressed"}'},
                    # 13. 勾/开掌式抓法(Open Hand)
                    {'name': '勾/开掌式抓法(Open Hand)', 'category': 'Hand_Grip', 'difficulty_level': 'V1',
                     'description': '节省前臂体力', 'technical_points': '指腹按压在岩壁上，力矩小减少屈指肌负担',
                     'body_position_3d': '{"center_of_gravity": [0.1, 0.9, 0.2], "wrist_angle": 160, "finger_curvature": "gentle"}'},
                    # 14. 扒/摩擦式抓法(Palming/Friction Grip)
                    {'name': '扒/摩擦式抓法(Palming/Friction Grip)', 'category': 'Hand_Grip', 'difficulty_level': 'V3',
                     'description': '支撑半球状岩点', 'technical_points': '手掌充分张开增加摩擦力，重心挪到岩点正下方',
                     'body_position_3d': '{"center_of_gravity": [0.2, 1.1, 0.4], "arm_angle": 90, "palm_contact": "full"}'},
                    # 15. 捏(Pinch Grip)
                    {'name': '捏(Pinch Grip)', 'category': 'Hand_Grip', 'difficulty_level': 'V4',
                     'description': '捏住长条状岩点',
                     'technical_points': '从上方捏紧并下拉固定，最后用虎口摩擦力增加支撑',
                     'body_position_3d': '{"center_of_gravity": [-0.15, 1.0, 0.35], "thumb_position": "opposed", "wrist_rotation": 30}'},
                    # 16. 单指扣
                    {'name': '单指扣', 'category': 'Hand_Grip', 'difficulty_level': 'V6', 'description': '洞状小岩点',
                     'technical_points': '仅放入1根指头，注意避免肌腱受伤',
                     'body_position_3d': '{"center_of_gravity": [0.1, 1.2, 0.3], "finger_insertion": "deep", "body_tension": "high"}'},
                    # 17. 双指扣
                    {'name': '双指扣', 'category': 'Hand_Grip', 'difficulty_level': 'V5', 'description': '洞状中型岩点',
                     'technical_points': '优先使用中指+无名指(关节同高)或中指+食指(力量最大)',
                     'body_position_3d': '{"center_of_gravity": [0.05, 1.1, 0.25], "finger_pair": "middle+ring", "elbow_position": "stable"}'},
                    # 18. 三指扣
                    {'name': '三指扣', 'category': 'Hand_Grip', 'difficulty_level': 'V3', 'description': '较大洞状岩点',
                     'technical_points': '三指并排或中指叠在另外两指上，保持手指关节角度适中',
                     'body_position_3d': '{"center_of_gravity": [0.1, 1.0, 0.3], "finger_arrangement": "parallel", "wrist_stability": "high"}'},
                    # 19. 包(Wrap)
                    {'name': '包(Wrap)', 'category': 'Hand_Grip', 'difficulty_level': 'V2', 'description': '抓住把手点',
                     'technical_points': '以第三关节支撑，手掌方向改变可增加可动范围',
                     'body_position_3d': '{"center_of_gravity": [-0.1, 0.9, 0.3], "wrist_rotation": 180, "grip_type": "wrap"}'},
                    # 20. 正抓(Overhand Grip)
                    {'name': '正抓(Overhand Grip)', 'category': 'Hand_Grip', 'difficulty_level': 'V1',
                     'description': '传统抓取方法', 'technical_points': '从上方抓住岩点，手臂微弯发挥最大肌力',
                     'body_position_3d': '{"center_of_gravity": [0.05, 0.85, 0.2], "elbow_angle": 110, "shoulder_position": "relaxed"}'},
                    # 21. 勾爪扣(Tree Finger)
                    {'name': '勾爪扣(Tree Finger)', 'category': 'Hand_Grip', 'difficulty_level': 'V3',
                     'description': '不规则岩点',
                     'technical_points': '食指、中指、拇指分开抓取，无名指贴在一旁，手腕固定增强稳定性',
                     'body_position_3d': '{"center_of_gravity": [-0.05, 1.0, 0.3], "finger_separation": "wide", "wrist_stability": "fixed"}'},
                    # 22. 倒抓(Under Cling)
                    {'name': '倒抓(Under Cling)', 'category': 'Hand_Grip', 'difficulty_level': 'V4',
                     'description': '朝下岩点', 'technical_points': '从下面抓取，手肘往后拉使身体贴近岩壁',
                     'body_position_3d': '{"center_of_gravity": [0.25, 1.2, 0.45], "elbow_angle": 90, "body_position": "close to wall"}'},
                    # 23. 反持
                    {'name': '反持', 'category': 'Hand_Grip', 'difficulty_level': 'V3', 'description': '高角度岩点',
                     'technical_points': '将身体降低，手绕到背面按压岩点，适用于屋顶岩点前端',
                     'body_position_3d': '{"center_of_gravity": [0.3, 1.3, 0.5], "arm_rotation": 270, "body_position": "inverted"}'},
                    # 24. 善用拇指
                    {'name': '善用拇指', 'category': 'Support_Technique', 'difficulty_level': 'V2',
                     'description': '增强抓点力量',
                     'technical_points': '拇指贴在岩点下方用捏的感觉支撑，弯曲拇指肌肉位于手掌内不易疲劳',
                     'body_position_3d': '{"center_of_gravity": [-0.05, 0.9, 0.2], "thumb_position": "supporting", "grip_efficiency": "increased"}'},
                    # 25. 善用膝盖
                    {'name': '善用膝盖', 'category': 'Support_Technique', 'difficulty_level': 'V3',
                     'description': '强化支撑力', 'technical_points': '将膝盖抵在支撑手背增强力量，脚抬到腰高度',
                     'body_position_3d': '{"center_of_gravity": [0.0, 1.1, 0.3], "knee_position": "against hand", "body_angle": 70}'},
                    # 26. 利用推的力量
                    {'name': '利用推的力量', 'category': 'Support_Technique', 'difficulty_level': 'V2',
                     'description': '补足握力不足',
                     'technical_points': '手肘上抬使用背阔肌压岩点，将推力加入拉的要素中抑制疲劳',
                     'body_position_3d': '{"center_of_gravity": [-0.1, 0.85, 0.25], "elbow_position": "raised", "shoulder_engagement": "high"}'},
                    # 27. 4字式(Figure-4)
                    {'name': '4字式(Figure-4)', 'category': 'Support_Technique', 'difficulty_level': 'V6',
                     'description': '抓取远距离岩点', 'technical_points': '将膝盖挂到支撑手肘上，腰不往下掉，背脊打直',
                     'body_position_3d': '{"center_of_gravity": [0.15, 1.3, 0.4], "leg_position": "crossed", "body_tension": "maximum"}'},
                    # 28. 顶(Edging)
                    {'name': '顶(Edging)', 'category': 'Footwork_Basic', 'difficulty_level': 'V2',
                     'description': '卡在细小岩缘上', 'technical_points': '固定脚趾、脚跟往上提，用填塞方式踩',
                     'body_position_3d': '{"center_of_gravity": [-0.05, 0.85, 0.2], "foot_angle": 15, "toe_position": "curled"}'},
                    # 29. 摩(Smearing)
                    {'name': '摩(Smearing)', 'category': 'Footwork_Basic', 'difficulty_level': 'V1',
                     'description': '支撑在光滑岩面上', 'technical_points': '将鞋底用力按压在岩壁上，利用最大静摩擦力',
                     'body_position_3d': '{"center_of_gravity": [0.1, 0.9, 0.25], "body_angle": 70, "foot_pressure": "high"}'},
                    # 30. 蹭(Smedging)
                    {'name': '蹭(Smedging)', 'category': 'Footwork_Basic', 'difficulty_level': 'V3',
                     'description': '结合顶与摩的效果',
                     'technical_points': '降低鞋子角度边摩擦边旋转踩点，岩缘会陷进鞋底',
                     'body_position_3d': '{"center_of_gravity": [-0.02, 0.88, 0.22], "foot_rotation": 30, "weight_distribution": "focused"}'},
                    # 31. 内正踩
                    {'name': '内正踩', 'category': 'Footwork_Basic', 'difficulty_level': 'V2',
                     'description': '踩极小岩点', 'technical_points': '用靠近鞋尖的内侧踩点，注意力集中在鞋缘',
                     'body_position_3d': '{"center_of_gravity": [0.05, 0.9, 0.2], "toe_position": "flexed inward", "heel_position": "raised"}'},
                    # 32. 外正踩
                    {'name': '外正踩', 'category': 'Footwork_Basic', 'difficulty_level': 'V2',
                     'description': '踩极小岩点', 'technical_points': '用靠近鞋尖的外侧踩点，注意力集中在鞋缘',
                     'body_position_3d': '{"center_of_gravity": [-0.05, 0.9, 0.2], "toe_position": "flexed outward", "heel_position": "raised"}'},
                    # 33. 内侧踩(Inside Edge)
                    {'name': '内侧踩(Inside Edge)', 'category': 'Footwork_Basic', 'difficulty_level': 'V1',
                     'description': '最好使力的踩法', 'technical_points': '将拇指球重叠在岩点上，重心落在拇指球位置',
                     'body_position_3d': '{"center_of_gravity": [0.08, 0.92, 0.23], "foot_position": "inside edge", "knee_angle": 110}'},
                    # 34. 外侧踩(Outside Edge)
                    {'name': '外侧踩(Outside Edge)', 'category': 'Footwork_Basic', 'difficulty_level': 'V3',
                     'description': '让动作更流畅',
                     'technical_points': '用小指球位置踩点，从小指扭转，腰比较容易贴近岩壁',
                     'body_position_3d': '{"center_of_gravity": [-0.1, 0.95, 0.28], "foot_rotation": "outward", "body_twist": 25}'},
                    # 35. 侧踩(Back Step)
                    {'name': '侧踩(Back Step)', 'category': 'Footwork_Basic', 'difficulty_level': 'V4',
                     'description': '脚朝背后方向踩', 'technical_points': '踩前先确认位置，身体旋转时保持平衡',
                     'body_position_3d': '{"center_of_gravity": [-0.15, 1.0, 0.35], "hip_rotation": 45, "body_twist": 60}'},
                    # 36. 扣(Pocket)
                    {'name': '扣(Pocket)', 'category': 'Footwork_Basic', 'difficulty_level': 'V5',
                     'description': '指洞需要细腻脚法', 'technical_points': '用挤塞方式将鞋子扭进去卡紧，注意扭转角度',
                     'body_position_3d': '{"center_of_gravity": [0.05, 1.05, 0.3], "foot_twist": 45, "ankle_flexion": "high"}'},
                    # 37. 挂脚(Heel Hook)
                    {'name': '挂脚(Heel Hook)', 'category': 'Footwork_Hooking', 'difficulty_level': 'V3',
                     'description': '克服仰角岩壁',
                     'technical_points': '重心偏向挂住脚跟的那条腿，腰确实往上抬再抓取下一个岩点',
                     'body_position_3d': '{"center_of_gravity": [0.2, 1.3, 0.45], "knee_angle": 90, "hip_rotation": 30}'},
                    # 38. 勾脚(Toe Hook)
                    {'name': '勾脚(Toe Hook)', 'category': 'Footwork_Hooking', 'difficulty_level': 'V4',
                     'description': '用脚尖上勾', 'technical_points': '用整条腿将脚背往上抬，腿要伸直甚至上半身倒下',
                     'body_position_3d': '{"center_of_gravity": [-0.1, 1.1, 0.35], "leg_extension": "straight", "body_angle": 85}'},
                    # 39. 内侧挂
                    {'name': '内侧挂', 'category': 'Footwork_Hooking', 'difficulty_level': 'V3',
                     'description': '轻轻抵住侧面', 'technical_points': '适用于角落或直条状岩点，用内侧来拉',
                     'body_position_3d': '{"center_of_gravity": [-0.05, 1.0, 0.3], "hip_rotation": 20, "leg_position": "crossed"}'},
                    # 40. 外侧挂
                    {'name': '外侧挂', 'category': 'Footwork_Hooking', 'difficulty_level': 'V5',
                     'description': '惊人的高支撑力',
                     'technical_points': '借由身体侧面的扭转来撑住躯干，更能有效对抗前倾峭壁',
                     'body_position_3d': '{"center_of_gravity": [0.25, 1.4, 0.5], "body_twist": 45, "leg_extension": 85}'},
                    # 41. 倒挂型挂钩
                    {'name': '倒挂型挂钩', 'category': 'Footwork_Hooking', 'difficulty_level': 'V6',
                     'description': '前倾峭壁必用',
                     'technical_points': '双脚夹住或朝反方向推，在前倾峭壁及水平岩壁上获得强劲支撑力',
                     'body_position_3d': '{"center_of_gravity": [0.3, 1.5, 0.6], "body_angle": 120, "leg_tension": "maximum"}'},
                    # 42. 双勾脚(蝙蝠挂)
                    {'name': '双勾脚(蝙蝠挂)', 'category': 'Footwork_Hooking', 'difficulty_level': 'V4',
                     'description': '双脚同时挂住',
                     'technical_points': '双脚同时挂住岩点，形成稳定三角支撑，大幅减轻手部负担',
                     'body_position_3d': '{"center_of_gravity": [0.0, 1.2, 0.4], "leg_position": "symmetric", "body_tension": "balanced"}'},
                    # 43. 换脚技巧
                    {'name': '换脚技巧', 'category': 'Footwork_Advanced', 'difficulty_level': 'V2',
                     'description': '精确替换踩踏位置', 'technical_points': '上面的脚贴得非常近，将下面的脚迅速抽离',
                     'body_position_3d': '{"center_of_gravity": [0.05, 0.95, 0.25], "foot_position": "close", "arm_extension": "wide"}'},
                    # 44. 脚塞
                    {'name': '脚塞', 'category': 'Footwork_Crack', 'difficulty_level': 'V4', 'description': '裂隙攀登',
                     'technical_points': '小脚指那侧向下，朝内侧旋转卡死',
                     'body_position_3d': '{"center_of_gravity": [0.0, 1.0, 0.3], "foot_rotation": "inward", "ankle_position": "flexed"}'},
                    # 45. T字脚(T-Stack)
                    {'name': 'T字脚(T-Stack)', 'category': 'Footwork_Crack', 'difficulty_level': 'V5',
                     'description': '无法固定时使用', 'technical_points': '一只脚摆直另一只摆横，无法交互向上',
                     'body_position_3d': '{"center_of_gravity": [0.05, 1.1, 0.35], "leg_position": "T-shaped", "body_tension": "high"}'},
                    # 46. 背与脚(Back&Foot)
                    {'name': '背与脚(Back&Foot)', 'category': 'Footwork_Crack', 'difficulty_level': 'V3',
                     'description': '大裂隙攀登', 'technical_points': '全身进入裂隙，利用身体各部位前后抵住岩壁',
                     'body_position_3d': '{"center_of_gravity": [0.0, 1.2, 0.4], "body_position": "full insertion", "leg_extension": "backwards"}'},
                    # 47. 调节平衡原理
                    {'name': '调节平衡原理', 'category': 'Balance_Technique', 'difficulty_level': 'V1',
                     'description': '防止身体旋转', 'technical_points': '通过左右重量均等分配，使身体不旋转',
                     'body_position_3d': '{"center_of_gravity": [0.0, 1.0, 0.2], "hips_angle": 10, "weight_distribution": "50% left, 50% right"}'},
                    # 48. 内对角
                    {'name': '内对角', 'category': 'Balance_Technique', 'difficulty_level': 'V1',
                     'description': '保持平衡，分配体重',
                     'technical_points': '左手支撑，右脚内侧踩，通过身体展开的对角固定法维持平衡',
                     'body_position_3d': '{"center_of_gravity": [0.05, 1.0, 0.25], "hips_angle": 15, "weight_distribution": "70% feet, 30% hands"}'},
                    # 49. 外对角
                    {'name': '外对角', 'category': 'Balance_Technique', 'difficulty_level': 'V2',
                     'description': '前倾壁必用手法',
                     'technical_points': '左手支撑，右脚外侧踩，通过身体内缩的对角固定法维持平衡',
                     'body_position_3d': '{"center_of_gravity": [-0.1, 1.1, 0.3], "hips_angle": 25, "weight_distribution": "60% feet, 40% hands"}'},
                    # 50. 内侧挂旗法
                    {'name': '内侧挂旗法', 'category': 'Balance_Technique', 'difficulty_level': 'V2',
                     'description': '腿交叉于身前',
                     'technical_points': '平行支撑时，将腿绕过身前，双腿呈交叉状，形成对角支撑',
                     'body_position_3d': '{"center_of_gravity": [-0.05, 1.0, 0.35], "hips_angle": 30, "flagging_leg": "crossed_in_front"}'},
                    # 51. 外侧挂旗法
                    {'name': '外侧挂旗法', 'category': 'Balance_Technique', 'difficulty_level': 'V3',
                     'description': '腿交叉于身后', 'technical_points': '平行支撑时，将腿绕到身体外侧，与踩脚呈交叉状',
                     'body_position_3d': '{"center_of_gravity": [0.1, 1.1, 0.3], "hips_angle": 25, "flagging_leg": "crossed_back"}'},
                    # 52. 弓身平衡
                    {'name': '弓身平衡', 'category': 'Balance_Technique', 'difficulty_level': 'V2',
                     'description': '处理近距离平衡',
                     'technical_points': '腰部弓起后仰，通过全身配重，使左右重量平均分配',
                     'body_position_3d': '{"center_of_gravity": [0.0, 0.95, 0.2], "hips_angle": 35, "body_arch": "moderate"}'},
                    # 53. 挂旗法的切换
                    {'name': '挂旗法的切换', 'category': 'Balance_Technique', 'difficulty_level': 'V3',
                     'description': '灵活变换平衡', 'technical_points': '通过外侧挂旗法避免频繁换脚，保持同一踩脚平衡',
                     'body_position_3d': '{"center_of_gravity": [0.05, 1.05, 0.28], "hips_angle": 20, "weight_shift": "smooth"}'},
                    # 54. 深度外侧挂旗法
                    {'name': '深度外侧挂旗法', 'category': 'Balance_Technique', 'difficulty_level': 'V4',
                     'description': '休息姿势', 'technical_points': '将腰大幅下沉，腿大幅伸出，通过甩腿位置调节平衡',
                     'body_position_3d': '{"center_of_gravity": [0.15, 0.85, 0.35], "hips_angle": 45, "flagging_leg": "extended_deep"}'},
                    # 55. 手脚同点
                    {'name': '手脚同点', 'category': 'Balance_Technique', 'difficulty_level': 'V3',
                     'description': '避免身体旋转', 'technical_points': '形成3点支撑，避免2点支撑产生的旋转力矩',
                     'body_position_3d': '{"center_of_gravity": [0.0, 1.0, 0.15], "hips_angle": 10, "weight_distribution": "balanced"}'},
                    # 56. 下压(Mantling)
                    {'name': '下压(Mantling)', 'category': 'Push_Pull', 'difficulty_level': 'V3',
                     'description': '从下方推上平台', 'technical_points': '先拉后推，当身体抬高到胸口后改用推力',
                     'body_position_3d': '{"center_of_gravity": [0.0, 1.3, 0.3], "arm_angle": 90, "push_phase": "transition"}'},
                    # 57. 双手顶住
                    {'name': '双手顶住', 'category': 'Push_Pull', 'difficulty_level': 'V3',
                     'description': '用手掌推高身体', 'technical_points': '双手抵住墙面，通过惯性迅速将身体推高',
                     'body_position_3d': '{"center_of_gravity": [0.0, 1.5, 0.25], "arm_extension": "full", "push_angle": 60}'},
                    # 58. 单手推住
                    {'name': '单手推住', 'category': 'Push_Pull', 'difficulty_level': 'V4',
                     'description': '利用侧向推力', 'technical_points': '将身体重心靠近手臂，通过肩膀内缩、手臂内转卡紧',
                     'body_position_3d': '{"center_of_gravity": [-0.15, 1.3, 0.3], "arm_angle": 30, "body_rotation": 20}'},
                    # 59. 垂膝挂旗法
                    {'name': '垂膝挂旗法', 'category': 'Counter_Force', 'difficulty_level': 'V5',
                     'description': '增强垂膝式稳定性', 'technical_points': '结合垂膝式与挂旗法，腿向上摆作为推力',
                     'body_position_3d': '{"center_of_gravity": [-0.1, 1.2, 0.45], "hips_angle": 65, "knee_rotation": 50, "flagging_leg": "upward"}'},
                    # 60. 高垂膝式
                    {'name': '高垂膝式', 'category': 'Counter_Force', 'difficulty_level': 'V5',
                     'description': '休息姿势', 'technical_points': '蹬脚位置比膝盖高，可用于暂时休息',
                     'body_position_3d': '{"center_of_gravity": [-0.05, 1.25, 0.4], "hips_angle": 70, "knee_position": "elevated"}'},
                    # 61. 外撑(Stemming)
                    {'name': '外撑(Stemming)', 'category': 'Counter_Force', 'difficulty_level': 'V3',
                     'description': '双脚左右撑开', 'technical_points': '双脚在后侧，身体在前侧，重心倒向前方',
                     'body_position_3d': '{"center_of_gravity": [0.0, 1.1, 0.5], "leg_extension": "wide", "body_angle": 30}'},
                    # 62. 侧身(Lay Back)
                    {'name': '侧身(Lay Back)', 'category': 'Counter_Force', 'difficulty_level': 'V4',
                     'description': '沿直向岩点攀登', 'technical_points': '一手拇指向下另一手拇指朝上，手交叉爬比较快',
                     'body_position_3d': '{"center_of_gravity": [-0.2, 1.2, 0.4], "body_angle": 60, "arm_position": "extended"}'},
                    # 63. 锁膝法(Knee Bar)
                    {'name': '锁膝法(Knee Bar)', 'category': 'Counter_Force', 'difficulty_level': 'V3',
                     'description': '用膝盖卡住休息',
                     'technical_points': '用小腿为轴，脚踩下方岩点，用膝盖顶端顶住上方大岩点',
                     'body_position_3d': '{"center_of_gravity": [0.0, 1.0, 0.3], "knee_angle": 90, "leg_pressure": "high"}'},
                    # 64. 侧推(Side Push)
                    {'name': '侧推(Side Push)', 'category': 'Counter_Force', 'difficulty_level': 'V4',
                     'description': '往侧边推高身体', 'technical_points': '透过全身支撑，脚与手位于对角上较稳定',
                     'body_position_3d': '{"center_of_gravity": [0.3, 1.1, 0.35], "body_rotation": 45, "push_angle": 30}'},
                    # 65. 上推(Upper Push)
                    {'name': '上推(Upper Push)', 'category': 'Counter_Force', 'difficulty_level': 'V4',
                     'description': '向上推高身体', 'technical_points': '手以肩膀为支点，手背靠在肩膀上方最能发挥力气',
                     'body_position_3d': '{"center_of_gravity": [0.0, 1.3, 0.25], "arm_angle": 90, "body_position": "upright"}'},
                    # 66. 小蹲跳(Short Lunge)
                    {'name': '小蹲跳(Short Lunge)', 'category': 'Dynamic_Move', 'difficulty_level': 'V4',
                     'description': '短距离不靠反作用力', 'technical_points': '不靠反作用力，控制高度刚好能抓到岩点',
                     'body_position_3d': '{"center_of_gravity": [0.1, 1.3, 0.3], "hips_angle": 30, "kick_intensity": "minimal"}'},
                    # 67. 单手蹲跳(One-hand Lunge)
                    {'name': '单手蹲跳(One-hand Lunge)', 'category': 'Dynamic_Move', 'difficulty_level': 'V5',
                     'description': '单手支撑跳跃', 'technical_points': '腰部垂在支撑手下方，藉由脚力靠半边的身体蹬出去',
                     'body_position_3d': '{"center_of_gravity": [-0.15, 1.5, 0.4], "body_rotation": 25, "weight_distribution": "60% legs, 40% arm"}'},
                    # 68. 协调跳跃(Conditioning Lunge)
                    {'name': '协调跳跃(Conditioning Lunge)', 'category': 'Dynamic_Move', 'difficulty_level': 'V6',
                     'description': '多动作连接', 'technical_points': '由2~3个动作组成，保持身体不要打转',
                     'body_position_3d': '{"center_of_gravity": [0.05, 1.4, 0.45], "body_rotation": 40, "coordination": "high"}'},
                    # 69. 重力助推(Swing-by)
                    {'name': '重力助推(Swing-by)', 'category': 'Dynamic_Move', 'difficulty_level': 'V4',
                     'description': '横向钟摆运动', 'technical_points': '往跳跃方向的反向多摆动来增加反作用力',
                     'body_position_3d': '{"center_of_gravity": [0.3, 1.2, 0.5], "swing_amplitude": "large", "momentum": "high"}'},
                    # 70. 零式(Cypher)
                    {'name': '零式(Cypher)', 'category': 'Dynamic_Move', 'difficulty_level': 'V5',
                     'description': '脚先动的滞空技巧',
                     'technical_points': '先使用外侧挂旗法，将腰往下沉深一点，双手拉住岩点',
                     'body_position_3d': '{"center_of_gravity": [0.2, 1.3, 0.4], "leg_swing": "forward", "body_extension": "full"}'},
                    # 71. 外向挂旗法/逆零式
                    {'name': '外向挂旗法/逆零式', 'category': 'Dynamic_Move', 'difficulty_level': 'V6',
                     'description': '对角位置施展',
                     'technical_points': '双脚以内八的方式小幅度旋转，尽可能让肩膀贴着岩壁',
                     'body_position_3d': '{"center_of_gravity": [-0.2, 1.2, 0.45], "body_twist": 60, "flagging_leg": "extended_outward"}'},
                    # 72. 指力跳(Campus Lunge)
                    {'name': '指力跳(Campus Lunge)', 'category': 'Dynamic_Move', 'difficulty_level': 'V7',
                     'description': '仅靠手指力量跳跃',
                     'technical_points': '先将身体往上拉，再向下，运用反作用力与肌腱的弹性',
                     'body_position_3d': '{"center_of_gravity": [0.0, 1.6, 0.3], "arm_extension": "full", "body_arch": "pronounced"}'},
                    # 73. 左右轮流
                    {'name': '左右轮流', 'category': 'Hand_Sequence', 'difficulty_level': 'V1',
                     'description': '最基础手顺', 'technical_points': '左右手交替伸手，保持姿势最稳定',
                     'body_position_3d': '{"center_of_gravity": [0.0, 1.0, 0.2], "balance": "optimal", "weight_distribution": "even"}'},
                    # 74. 跳点(Bump)
                    {'name': '跳点(Bump)', 'category': 'Hand_Sequence', 'difficulty_level': 'V3',
                     'description': '跳过中间岩点',
                     'technical_points': '当眼前的岩点不易支撑而下一个岩点较佳时，把眼前的岩点当作中继点',
                     'body_position_3d': '{"center_of_gravity": [0.1, 1.1, 0.25], "arm_extension": "extended", "timing": "precise"}'},
                    # 75. 钢琴手
                    {'name': '钢琴手', 'category': 'Hand_Sequence', 'difficulty_level': 'V2',
                     'description': '手指交叠换手',
                     'technical_points': '手指先不握，等放入裂隙再用力握紧，一定要卡住裂隙中最窄的部分',
                     'body_position_3d': '{"center_of_gravity": [0.05, 0.9, 0.2], "finger_arrangement": "interlocked", "grip_precision": "high"}'},
                    # 76. 握杯法
                    {'name': '握杯法', 'category': 'Hand_Sequence', 'difficulty_level': 'V3',
                     'description': '从直向来转换', 'technical_points': '一手拇指向下另一手拇指朝上，手交叉爬比较快',
                     'body_position_3d': '{"center_of_gravity": [-0.1, 1.0, 0.3], "thumb_position": "opposed", "body_rotation": 20}'},
                    # 77. 换边
                    {'name': '换边', 'category': 'Hand_Sequence', 'difficulty_level': 'V4',
                     'description': '角落旋转换手', 'technical_points': '身体侧向岩壁，创造反作用力，脚部配合推力',
                     'body_position_3d': '{"center_of_gravity": [-0.15, 1.1, 0.35], "body_angle": 45, "foot_pressure": "balanced"}'},
                    # 78. 交叉手(Cross Move)
                    {'name': '交叉手(Cross Move)', 'category': 'Hand_Sequence', 'difficulty_level': 'V3',
                     'description': '手臂交叉',
                     'technical_points': '做交叉手之前要仔细确认脚点，尤其水平岩壁上一旦脚松开身体就会旋转',
                     'body_position_3d': '{"center_of_gravity": [0.0, 1.2, 0.3], "arm_crossing": "moderate", "waist_position": "low"}'},
                    # 79. 从下方切入
                    {'name': '从下方切入', 'category': 'Hand_Sequence', 'difficulty_level': 'V4',
                     'description': '交叉手的变形',
                     'technical_points': '从下面切入反而能让肩膀靠近岩壁，使更多的体重落在脚上',
                     'body_position_3d': '{"center_of_gravity": [-0.1, 1.15, 0.35], "shoulder_position": "close to wall", "body_twist": 30}'},
                    # 80. 从上方切入
                    {'name': '从上方切入', 'category': 'Hand_Sequence', 'difficulty_level': 'V4',
                     'description': '交叉手的变形', 'technical_points': '当下一个岩点位于上方，就得从上面切入交叉手',
                     'body_position_3d': '{"center_of_gravity": [0.1, 1.25, 0.3], "shoulder_extension": "full", "reach_distance": "maximum"}'},
                    # 81. 交叉式(Crossover)
                    {'name': '交叉式(Crossover)', 'category': 'Hand_Sequence', 'difficulty_level': 'V5',
                     'description': '大幅拉近距离', 'technical_points': '动作要迅速，抓取手才能朝反方向大幅移动',
                     'body_position_3d': '{"center_of_gravity": [-0.2, 1.3, 0.4], "body_rotation": 45, "arm_extension": "full"}'},
                    # 82. 同点(Match)
                    {'name': '同点(Match)', 'category': 'Hand_Sequence', 'difficulty_level': 'V2',
                     'description': '双手抓同一点', 'technical_points': '将另一手尽可能地叠合过来，防止力气流失',
                     'body_position_3d': '{"center_of_gravity": [0.0, 0.95, 0.15], "hand_overlap": "complete", "stability": "high"}'},
                    # 83. 重抓
                    {'name': '重抓', 'category': 'Hand_Sequence', 'difficulty_level': 'V3',
                     'description': '同一点改变抓法',
                     'technical_points': '在同一个岩点上改变同一只手的抓法，如正手转推或反抓',
                     'body_position_3d': '{"center_of_gravity": [0.05, 1.05, 0.2], "grip_transition": "smooth", "body_position": "adjusting"}'},
                    # 84. 连续换手
                    {'name': '连续换手', 'category': 'Hand_Sequence', 'difficulty_level': 'V4',
                     'description': '快速更换手点', 'technical_points': '手顺要流畅，不能停顿，否则会失去动能',
                     'body_position_3d': '{"center_of_gravity": [0.1, 1.1, 0.25], "hand_speed": "rapid", "rhythm": "consistent"}'},
                    # 85. 手掌(Hand)
                    {'name': '手掌(Hand)', 'category': 'Crack_Climbing', 'difficulty_level': 'V2',
                     'description': '比手掌稍宽裂隙', 'technical_points': '手掌平放在裂隙中，用手指和掌根抵住两侧',
                     'body_position_3d': '{"center_of_gravity": [0.0, 1.0, 0.3], "hand_position": "flat", "pressure_distribution": "even"}'},
                    # 86. 手指(Finger)
                    {'name': '手指(Finger)', 'category': 'Crack_Climbing', 'difficulty_level': 'V4',
                     'description': '狭窄裂隙', 'technical_points': '手指插入狭窄裂隙，靠手指弯曲和张力卡住',
                     'body_position_3d': '{"center_of_gravity": [0.05, 1.05, 0.25], "finger_arrangement": "stacked", "tension": "high"}'},
                    # 87. 薄手(Thin Hand)
                    {'name': '薄手(Thin Hand)', 'category': 'Crack_Climbing', 'difficulty_level': 'V3',
                     'description': '不大不小裂隙',
                     'technical_points': '手略微弯曲，手掌窄侧放入裂隙，用骨节和手掌边缘卡住两侧',
                     'body_position_3d': '{"center_of_gravity": [-0.05, 1.0, 0.28], "hand_angle": 45, "jam_effectiveness": "optimal"}'},
                    # 88. 环扣(Ring Lock)
                    {'name': '环扣(Ring Lock)', 'category': 'Crack_Climbing', 'difficulty_level': 'V5',
                     'description': '特殊裂隙形状',
                     'technical_points': '手指环状扭曲，大拇指和食指或中指形成环状卡在裂隙中',
                     'body_position_3d': '{"center_of_gravity": [0.0, 1.1, 0.3], "finger_configuration": "ring", "locking_mechanism": "thumb-to-finger"}'},
                    # 89. 拳头(Fist)
                    {'name': '拳头(Fist)', 'category': 'Crack_Climbing', 'difficulty_level': 'V3',
                     'description': '中等宽度裂隙', 'technical_points': '将手握成拳头放入裂隙，用力向外扩张卡住最窄处',
                     'body_position_3d': '{"center_of_gravity": [0.05, 1.05, 0.3], "fist_tension": "high", "expansion_force": "lateral"}'},
                    # 90. 堆叠(Leavittation)
                    {'name': '堆叠(Leavittation)', 'category': 'Crack_Climbing', 'difficulty_level': 'V4',
                     'description': '大裂隙固定',
                     'technical_points': '双手叠合起来，可以手掌叠手掌、手掌叠拳头、拳头叠拳头',
                     'body_position_3d': '{"center_of_gravity": [0.0, 1.1, 0.35], "hand_stacking": "double", "body_tension": "high"}'},
                    # 91. 单臂锁定(Arm Lock)
                    {'name': '单臂锁定(Arm Lock)', 'category': 'Crack_Climbing', 'difficulty_level': 'V5',
                     'description': '窄裂隙攀登', 'technical_points': '手肘弯曲，透过后方的肩膀与前方的手掌制造反作用力',
                     'body_position_3d': '{"center_of_gravity": [-0.1, 1.2, 0.4], "elbow_angle": 90, "shoulder_pressure": "backward"}'},
                    # 92. 基本脚塞
                    {'name': '基本脚塞', 'category': 'Crack_Climbing', 'difficulty_level': 'V3',
                     'description': '裂隙攀登脚法', 'technical_points': '将脚塞入裂隙，小脚指那侧向下，朝内侧旋转卡死',
                     'body_position_3d': '{"center_of_gravity": [0.05, 1.0, 0.3], "foot_rotation": "inward", "ankle_flexion": "high"}'},
                    # 93. T字脚
                    {'name': 'T字脚', 'category': 'Crack_Climbing', 'difficulty_level': 'V5',
                     'description': '无法固定时使用', 'technical_points': '一只脚摆直另一只摆横成T字形，无法交互向上',
                     'body_position_3d': '{"center_of_gravity": [0.0, 1.3, 0.45], "leg_position": "T-shaped", "body_tension": "maximum"}'},
                    # 94. 背与脚
                    {'name': '背与脚', 'category': 'Crack_Climbing', 'difficulty_level': 'V3',
                     'description': '大裂隙攀登', 'technical_points': '全身进入裂隙，利用身体各部位前后抵住岩壁',
                     'body_position_3d': '{"center_of_gravity": [0.0, 1.2, 0.5], "body_position": "full insertion", "leg_extension": "backwards"}'}
                ]

                db.session.add_all([ClimbingAction(**action) for action in default_actions])
                db.session.commit()
                logger.info("✅ 测试动作数据初始化完成")

            # 检查是否需要初始化测试规则
            if ClimbingRule.query.count() == 0:
                logger.info("🔄 初始化测试规则数据...")
                # 从知识库添加测试规则
                default_rules = [
                    {'rule_name': '线路宽度规则', 'rule_description': '每条攀登线路的宽度应不小于1.8m',
                     'rule_category': 'safety', 'rule_parameters': '{"min_width": 1.8, "unit": "m"}'},
                    {'rule_name': '保护系统承载力', 'rule_description': '每个顶端保护系统承载力应不小于8kN',
                     'rule_category': 'safety', 'rule_parameters': '{"min_force": 8.0, "unit": "kN"}'},
                    {'rule_name': '保护挂片承载力',
                     'rule_description': '每个保护挂片应与结构直接链接，且承载力不小于8kN', 'rule_category': 'safety',
                     'rule_parameters': '{"min_force": 8.0, "unit": "kN"}'},
                    {'rule_name': '岩板静载荷', 'rule_description': '岩板耐受静载荷应不小于4kN',
                     'rule_category': 'equipment', 'rule_parameters': '{"min_static_load": 4.0, "unit": "kN"}'},
                    {'rule_name': '岩板动载荷', 'rule_description': '岩板的耐受动载荷应不小于6kN',
                     'rule_category': 'equipment', 'rule_parameters': '{"min_dynamic_load": 6.0, "unit": "kN"}'},
                    {'rule_name': '支点孔抗拉力', 'rule_description': '支点孔抗拉力应不小于3kN',
                     'rule_category': 'equipment', 'rule_parameters': '{"min_tensile_strength": 3.0, "unit": "kN"}'},
                    {'rule_name': '岩壁高度限制', 'rule_description': '用于攀石活动的人工岩壁有效垂直高度应不超过5m',
                     'rule_category': 'facility', 'rule_parameters': '{"max_height": 5.0, "unit": "m"}'}
                ]

                for rule_data in default_rules:
                    rule = ClimbingRule(**rule_data)
                    db.session.add(rule)

                db.session.commit()
                logger.info("✅ 测试规则数据初始化完成")

    except Exception as e:
        logger.error(f"❌ 数据库初始化失败: {str(e)}\n{traceback.format_exc()}")


if __name__ == '__main__':
    # 专业系统启动信息
    print("=" * 60)
    print(" ska 专业攀岩定线员系统启动")
    print("🎯 身份: IFSC Level 3 Certified Route Setter")
    print("🧠 模式: 基于规则的深度思考线路设计")
    print("📊 专注: 人体工程学优化、动作序列设计、安全规则验证")
    print(f"💻 前端: 左右分栏布局 (视频分析 | 线路生成)")
    print("=" * 60)

    # 初始化数据库
    with app.app_context():
        init_database()

    # 启动应用
    app.run(debug=True, host='0.0.0.0', port=5000)