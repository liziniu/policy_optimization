import logging
from logging import handlers

level_relations = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'crit': logging.CRITICAL
}


def create_logger(
        filename,
        level='info',
        fmt='{} %(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
):
    logger = logging.getLogger(filename)
    format_str = logging.Formatter(fmt)
    logger.setLevel(level_relations.get(level))

    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    logger.addHandler(sh)

    th = handlers.RotatingFileHandler(filename=filename, encoding='utf-8')
    th.setFormatter(format_str)
    logger.addHandler(th)

    return logger


class Logger(logging.Logger):

    def __init__(
            self,
            log_path,
            name='root',
            level='info',
            fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    ):
        logging.Logger.__init__(self, name)

        self.setLevel(level_relations.get(level))

        # custom format
        format_str = logging.Formatter(fmt)

        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        self.addHandler(sh)

        filename = "%s/log.txt" % log_path
        th = handlers.RotatingFileHandler(filename=filename, encoding='utf-8')
        th.setFormatter(format_str)
        self.addHandler(th)


if __name__ == '__main__':
    log = create_logger('all.log', level='debug')
    log.debug('debug')
    log.info('info')
