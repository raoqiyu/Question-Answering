import logging

formatter = logging.Formatter('[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] %(message)s')
console_handle = logging.StreamHandler()
console_handle.setFormatter(formatter)

logger = logging.getLogger('basice')

logger.addHandler(console_handle)
logger.setLevel(logging.DEBUG)



