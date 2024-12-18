import logging
import os

def setup_logger(log_file='./logs/app.log', level=logging.INFO):
    # 创建日志记录器
    log_name = log_file.split('/')[-1].split('.')[0]
    logger = logging.getLogger(log_name)
    logger.setLevel(level)

    # 防止重复添加 handler
    if not logger.handlers:
        # 文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # 终端处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        # 添加处理器到 logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger