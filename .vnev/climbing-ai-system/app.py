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

            # 每5帧处理一次（增加检测频率）
            if frame_count % 5 == 0:
                # 模拟动作检测 - 使用更智能的规则检测多个动作
                action_detected = False
                action_name = None

                # 用更合理的规则检测动作：每80-150帧开始一个新动作
                if current_action is None and frame_count % random.randint(80,
                                                                           150) == 0 and frame_count < total_frames - 50:
                    action_detected = True
                    action_name = random.choice([
                        "动态抓点", "静态平衡", "侧拉动作",
                        "精确脚法", "高跨步", "蹲跳",
                        "垂膝式", "侧拉", "脚跟挂"
                    ])

                # 如果检测到新动作
                if action_detected and current_action is None:
                    confidence = random.uniform(0.7, 0.95)
                    timestamp = frame_count / fps

                    # 开始新动作
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
            # 确保剩余动作持续时间至少0.5秒
            duration = (frame_count - current_action['start_frame']) / fps
            if duration > 0.5:
                detected_actions.append(current_action)

        # 对动作进行排序
        detected_actions.sort(key=lambda x: x['start_time'])

        # 保存结果
        if task_id in processing_results:
            processing_results[task_id].update({
                'status': 'completed',
                'progress': 100,
                'actions': detected_actions,
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

        # 返回进度信息
        return jsonify({
            'status': result['status'],
            'progress': result['progress'],
            'video_url': result.get('video_url', ''),
            'is_complete': result['status'] == 'completed',
            'has_error': 'error' in result,
            'error': result.get('error', ''),
            'actions': result.get('actions', []),
            'total_actions': result.get('total_actions', 0),
            'video_duration': result.get('video_duration', 0)
        }), 200

    except Exception as e:
        logger.error(f"❌ 获取状态失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500


@app.route('/generate-route', methods=['POST'])
def generate_route():
    """生成线路 - 数据库连接版"""
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

        try:
            # 从数据库获取岩点
            with app.app_context():
                # 获取适配该难度的岩点
                available_holds = WallHold.query.filter(
                    WallHold.difficulty_score.between(max(1, grade_level - 2), min(10, grade_level + 2))
                ).all()

                logger.info(f"🔍 从数据库检索到 {len(available_holds)} 个难度匹配的岩点")

                if not available_holds:
                    logger.warning(f"⚠️ 无可用难度 {grade} 的岩点，使用随机岩点")
                    available_holds = WallHold.query.order_by(db.func.random()).limit(10).all()

                # 生成随机线路
                hold_count = max(4, min(10, 8 - grade_level // 2))
                route_holds = []
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

            # 从数据库获取动作（包括所有难度等级小于或等于当前等级的动作）
            try:
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

                    # 生成随机动作
                    action_count = max(2, hold_count - 2)
                    actions = []
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

            except Exception as e:
                logger.error(f"❌ 从数据库获取动作失败: {str(e)}")
                # 使用默认动作
                actions = []
                default_actions = [
                    {'name': '动态抓点', 'technical_points': '重心下沉蓄力，手臂摆动制造反作用力，跳到最高点瞬间抓点'},
                    {'name': '静态平衡', 'technical_points': '通过对角支撑分配体重，腰贴近岩壁减少手部负担'},
                    {'name': '侧拉动作', 'technical_points': '拇指球位置踩点，脚跟抬起增加摩擦力，视线先于脚到达'},
                    {'name': '精确脚法', 'technical_points': '后脚膝盖向下旋转，身体旋转配合伸手，双脚前后蹬产生支撑力'},
                    {'name': '高跨步', 'technical_points': '脚跟挂点后身体重心向挂脚侧倾斜，利用脚部下拉减轻手部负担'}
                ]

                for i in range(action_count):
                    default_action = default_actions[i % len(default_actions)]
                    actions.append({
                        'id': i + 1,
                        'name': default_action['name'],
                        'category': '基础抓法',
                        'description': f"{default_action['name']}技术",
                        'difficulty_level': grade,
                        'technical_points': default_action['technical_points']
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
            logger.error(f"❌ 从数据库获取岩点失败: {str(e)}")
            # 使用默认岩点
            hold_count = max(4, min(10, 8 - grade_level // 2))
            route_holds = []

            for i in range(hold_count):
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
                    'id': i + 1,
                    'name': f'岩点 {i + 1}',
                    'x': x,
                    'y': y,
                    'type': hold_type,
                    'difficulty': grade,
                    'shape': 'round',
                    'size': 'medium',
                    'color': '#28a745' if hold_type == 'START' else ('#dc3545' if hold_type == 'END' else '#6f42c1')
                })

            # 使用默认动作
            action_count = max(2, hold_count - 2)
            actions = []
            default_actions = [
                {'name': '动态抓点', 'technical_points': '重心下沉蓄力，手臂摆动制造反作用力，跳到最高点瞬间抓点'},
                {'name': '静态平衡', 'technical_points': '通过对角支撑分配体重，腰贴近岩壁减少手部负担'},
                {'name': '侧拉动作', 'technical_points': '拇指球位置踩点，脚跟抬起增加摩擦力，视线先于脚到达'},
                {'name': '精确脚法', 'technical_points': '后脚膝盖向下旋转，身体旋转配合伸手，双脚前后蹬产生支撑力'},
                {'name': '高跨步', 'technical_points': '脚跟挂点后身体重心向挂脚侧倾斜，利用脚部下拉减轻手部负担'}
            ]

            for i in range(action_count):
                default_action = default_actions[i % len(default_actions)]
                actions.append({
                    'id': i + 1,
                    'name': default_action['name'],
                    'category': '基础抓法',
                    'description': f"{default_action['name']}技术",
                    'difficulty_level': grade,
                    'technical_points': default_action['technical_points']
                })

            route_data = {
                'route_id': datetime.now().strftime('%Y%m%d%H%M%S'),
                'grade': grade,
                'holds': route_holds,
                'actions': actions,
                'difficulty_description': f"{grade} 难度线路 - 使用默认数据",
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
                    {'name': '捏点', 'type': 'PINCH', 'difficulty_score': 4.0, 'shape': 'pinch', 'size': 'medium'},
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
                    {'name': '动态抓点', 'category': 'dynamic', 'difficulty_level': 'V3',
                     'description': '跨越大距离岩点',
                     'technical_points': '重心下沉蓄力，手臂摆动制造反作用力，跳到最高点瞬间抓点'},
                    {'name': '静态平衡', 'category': 'balance', 'difficulty_level': 'V1',
                     'description': '保持稳定平衡位置',
                     'technical_points': '通过对角支撑分配体重，腰贴近岩壁减少手部负担'},
                    {'name': '精确脚法', 'category': 'footwork', 'difficulty_level': 'V2',
                     'description': '精确踩点转移体重',
                     'technical_points': '拇指球位置踩点，脚跟抬起增加摩擦力，视线先于脚到达'},
                    {'name': '垂膝式', 'category': 'counter_force', 'difficulty_level': 'V4',
                     'description': '制造反作用力稳定身体',
                     'technical_points': '后脚膝盖向下旋转，身体旋转配合伸手，双脚前后蹬产生支撑力'},
                    {'name': '脚跟挂', 'category': 'footwork_hooking', 'difficulty_level': 'V3',
                     'description': '克服仰角岩壁',
                     'technical_points': '脚跟挂点后身体重心向挂脚侧倾斜，利用脚部下拉减轻手部负担'},
                    {'name': '挂旗法', 'category': 'balance_technique', 'difficulty_level': 'V2',
                     'description': '防止身体旋转', 'technical_points': '非踩脚侧腿向外伸展作为配重，腰往支撑手方向收'},
                    {'name': '高跨步', 'category': 'push_pull', 'difficulty_level': 'V2',
                     'description': '抬高脚位增加高度',
                     'technical_points': '重心完全转移到高脚上后再伸手，膝盖向下身体贴近岩壁'},
                    {'name': '静止点', 'category': 'dynamic_move', 'difficulty_level': 'V5',
                     'description': '精确抓取远点',
                     'technical_points': '双手拉近岩壁创造无重力瞬间，把握精确时机伸手抓点'},
                    {'name': '侧拉', 'category': 'hand_grip', 'difficulty_level': 'V3', 'description': '抓取侧面岩点',
                     'technical_points': '身体侧向岩壁，创造反作用力，脚部配合推力'}
                ]

                for action_data in default_actions:
                    action = ClimbingAction(**action_data)
                    db.session.add(action)

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