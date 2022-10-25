import logging

def get_logger(path, name='Training_logger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.filemode='w'
    formatter = logging.Formatter("%(asctime)s %(levelname)s : %(message)s","%Y-%m-%d %H:%M:%S")

    handler = logging.FileHandler(path)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger