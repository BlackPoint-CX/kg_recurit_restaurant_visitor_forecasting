import logging

SUB_DIRS = ['chart', 'data', 'log', 'model', 'src']
PROJECT_ROOT = '/home/bp/GitRepos/kg_recurit_restaurant_visitor_forecasting/'
PACKAGE_ROOT = PROJECT_ROOT + 'kg_recurit_restaurant_visitor_forecasting/'
chart_dir = PACKAGE_ROOT + 'chart/'
data_dir = PACKAGE_ROOT + 'data/'
log_dir = PACKAGE_ROOT + 'log/'
model_dir = PACKAGE_ROOT + 'model/'
src_dir = PACKAGE_ROOT + 'src/'


def set_logger(logger_name):
    logger_name = logger_name.lower()
    logger = logging.getLogger(name=logger_name)
    logger.setLevel(level=logging.INFO)
    logger_path = log_dir + logger_name + '.log'
    logger_handler = logging.FileHandler(filename=logger_path, mode='w+', encoding='utf-8')
    logger_formatter = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')
    logger_handler.setFormatter(fmt=logger_formatter)
    logger.addHandler(hdlr=logger_handler)
    return logger
