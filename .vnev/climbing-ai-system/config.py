import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'climbing_ai_secret_key_2025')

    # MySQL 配置
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_USER = os.getenv('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'czl18677288781')
    MYSQL_DB = os.getenv('MYSQL_DB', 'climbing_ai')

    # 数据库 URI
    SQLALCHEMY_DATABASE_URI = f'mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}?charset=utf8mb4'

    # 上传文件夹
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = 300 * 1024 * 1024  # 300MB limit

    # SQLAlchemy 配置
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'connect_args': {
            'charset': 'utf8mb4'
        }
    }

    @staticmethod
    def init_app(app):
        if not os.path.exists(Config.UPLOAD_FOLDER):
            os.makedirs(Config.UPLOAD_FOLDER)

        # 验证数据库连接
        try:
            import mysql.connector
            connection = mysql.connector.connect(
                host=Config.MYSQL_HOST,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DB
            )
            connection.close()
            print("✅ 数据库连接测试成功!")
        except Exception as e:
            print(f"❌ 数据库连接测试失败: {str(e)}")