import logging
import os
from datetime import datetime


def setup_logger(save_name, log_dir="./log", log_level="INFO", enable_debug=False):
    """
    简单优化的日志设置函数

    Args:
        save_name: 日志文件名前缀
        log_dir: 日志目录
        log_level: 日志级别 ("DEBUG", "INFO", "WARNING", "ERROR")
        enable_debug: 是否启用debug级别输出
    """
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger()

    if logger.handlers:
        return logger

    if enable_debug:
        logger.setLevel(logging.DEBUG)
        console_level = logging.DEBUG
        file_level = logging.DEBUG
    else:
        logger.setLevel(getattr(logging, log_level.upper()))
        console_level = getattr(logging, log_level.upper())
        file_level = logging.INFO

    # 1. 控制台输出 - 简洁格式
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    simple_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # 2. 常规日志文件 - 详细格式
    file_handler = logging.FileHandler(f"{log_dir}/{save_name}_{current_time}.log", encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(simple_formatter)
    logger.addHandler(file_handler)

    # 3. 错误日志文件 - 单独记录ERROR及以上级别
    error_handler = logging.FileHandler(f"{log_dir}/{save_name}_error_{current_time}.log", encoding="utf-8")
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)

    # 打印日志配置信息
    print(f"日志配置完成:")
    print(f"  常规日志: {log_dir}/{save_name}_{current_time}.log")
    print(f"  错误日志: {log_dir}/{save_name}_error_{current_time}.log")
    print(f"  命令行输出级别: {log_level}")
    if enable_debug:
        print(f"  调试模式已启用")

    return logger, f"{save_name}_{current_time}"


def get_logger(name=None):
    return logging.getLogger(name)
