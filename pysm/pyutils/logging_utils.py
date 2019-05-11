#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
import logging.config
import os
from pathlib import Path

from pyutils.config_utils import Configuration, load_config


class DynamicFileHandler(logging.Handler):

    def __init__(self, basedir, mode='a', encoding=None, delay=False, folder_base: bool = False) -> None:
        self.base_dir = Path(basedir)
        self.folder_base = folder_base
        self.mode = mode
        self.encoding = encoding
        self.delay = delay
        self.level = None
        self.formatter = None
        self.logger_name = None
        self.file_handler = None

        if self.mode == 'a':
            assert not self.folder_base, "Wrong configuration, shouldn't be folder base if you are appending to a log file"

    def get_log_file(self, logger_name):
        """Logfile will store in subdirectory of base_dir

        basedir = /log
        E.g:
            name: app => /log/app.log
            name: app.abc => /log/app/abc.log
            name: abc.system.report.abc => /log/app/system.report/abc.log

        In case it is folder base:
            name: app => /log/app_log/s001.log
            name: app.abc => /log/app/abc_log/s001.log
            name: app.system.report.abc => /log/app/system.report/abc_log/s001.log
        """
        namespaces = logger_name.split(".")

        if not self.folder_base:
            if len(namespaces) == 1:
                filename = self.base_dir / f"{namespaces[0]}.log"
            elif len(namespaces) == 2:
                filename = self.base_dir / namespaces[0] / f"{namespaces[1]}.log"
            else:
                filename = self.base_dir / namespaces[0] / ".".join(namespaces[1:-1]) / f"{namespaces[-1]}.log"

            filename.parent.mkdir(exist_ok=True, parents=True)
        else:
            if len(namespaces) == 1:
                dname = self.base_dir / f"{namespaces[0]}_log"
            elif len(namespaces) == 2:
                dname = self.base_dir / namespaces[0] / f"{namespaces[1]}_log"
            else:
                dname = self.base_dir / namespaces[0] / ".".join(namespaces[1:-1]) / f"{namespaces[-1]}_log"

            dname.mkdir(exist_ok=True, parents=True)
            files = sorted(dname.iterdir(), reverse=True)
            if len(files) == 0:
                filename = dname / "s001.log"
            else:
                filename = dname / f"s{int(files[0].stem[1:]) + 1:03d}.log"

        return filename

    def setLevel(self, level):
        self.level = level

    def setFormatter(self, fmt):
        self.formatter = fmt

    def set_logger_name(self, name):
        self.logger_name = name
        self.file_handler = logging.FileHandler(self.get_log_file(name), self.mode, self.encoding, self.delay)
        self.file_handler.setLevel(self.level)
        self.file_handler.setFormatter(self.formatter)

    def get_name(self):
        return self.file_handler.get_name()

    def set_name(self, name):
        return self.file_handler.set_name(name)

    def createLock(self):
        self.file_handler.createLock()

    def acquire(self):
        self.file_handler.acquire()

    def release(self):
        self.file_handler.release()

    def format(self, record):
        self.file_handler.format(record)

    def emit(self, record):
        self.file_handler.emit(record)

    def handle(self, record):
        self.file_handler.handle(record)

    def flush(self):
        self.file_handler.flush()

    def close(self):
        self.file_handler.close()

    def handleError(self, record):
        self.file_handler.handleError(record)


# SET TRACE LEVEL HERE
logging.addLevelName(5, "TRACE")
logging.TRACE = 5


def log_trace(self, msg, *args, **kwargs):
    self.log(5, msg, *args, **kwargs)


logging.Logger.trace = log_trace


class Logger(object):

    instance = None

    def __init__(self, config: Configuration, init_logging: bool) -> None:
        super().__init__()
        self.config = config
        self.init_logging = init_logging
        self.loggers = set()
        self.handlers = {}
        self.unrolling_logger()

        if self.init_logging:
            self.init()

    @staticmethod
    def get_instance(config, init_logging):
        if Logger.instance is None:
            Logger.instance = Logger(config, init_logging)
        return Logger.instance

    def unrolling_logger(self):
        # Update logging handlers
        deleting_handler_names = []
        new_handlers = []
        for handler_name in self.config.logging.handlers:
            handler = self.config.logging.handlers[handler_name]

            if handler_name.startswith('$rolling') and handler_name[-1] == '$':
                deleting_handler_names.append(handler_name)
                new_handlers.append(handler.to_dict())

        for name in deleting_handler_names:
            del self.config.logging.handlers[name]

        for new_handler in new_handlers:
            for arg in new_handler.keys():
                if isinstance(new_handler[arg], (str, int, float)):
                    new_handler[arg] = [new_handler[arg]] * len(new_handler['id'])

            for i, handler_name in enumerate(new_handler['id']):
                handler = {}
                for arg in new_handler.keys():
                    if arg != 'id':
                        handler[arg] = new_handler[arg][i]

                self.config.logging.handlers.set_conf(handler_name, handler, split_key=False)

        # Update logging logger
        deleting_logger_names = []
        new_loggers = []
        for logger_name in self.config.logging.loggers:
            if logger_name.startswith('$rolling') and logger_name[-1] == '$':
                deleting_logger_names.append(logger_name)
                new_loggers.append(self.config.logging.loggers[logger_name].to_dict())

        for logger_name in deleting_logger_names:
            del self.config.logging.loggers[logger_name]

        for new_logger in new_loggers:
            for arg in ['level', 'propagate']:
                if isinstance(new_logger[arg], (str, bool)):
                    new_logger[arg] = [new_logger[arg]] * len(new_logger['id'])

            if isinstance(new_logger['handlers'][0], str):
                new_logger['handlers'] = [new_logger['handlers']] * len(new_logger['id'])

            for i, logger_name in enumerate(new_logger['id']):
                logger = {arg: new_logger[arg][i] for arg in ['level', 'propagate', 'handlers']}
                self.config.logging.loggers.set_conf(logger_name, logger, split_key=False)

    def init(self):
        # Backup log file if needed
        for handler_name in self.config.logging.handlers:
            handler = self.config.logging.handlers[handler_name]

            if 'filename' in handler:
                if '__no_backup' in handler and handler.__no_backup:
                    # only backup file when required, and the file must be not empty
                    content = 0
                    if os.path.exists(handler.filename.as_path()):
                        with open(handler.filename.as_path(), 'rb') as f:
                            content = len(f.read(5))  # read first 5 bytes to determine if file is empty or not
                    if content > 0:
                        handler.filename.backup_path()

                if '__no_backup' in handler:
                    del handler.__no_backup

                handler.filename.ensure_path_existence()

    def get_logger(self, name):
        if name in self.loggers:
            return logging.getLogger(name)

        dict_config = self.config.logging.to_dict()

        ns_hierarchy = name.split(".")
        config_name = name
        for i in range(len(name) - 1, -1, -1):
            config_name = ".".join(ns_hierarchy[:i])
            if config_name in dict_config['loggers']:
                break

        assert config_name in dict_config['loggers'], 'Undefined logger: %s' % name

        self.loggers.add(name)

        logger_conf = dict_config['loggers'][config_name]

        logger = logging.getLogger(name)
        logger.propagate = logger_conf['propagate']
        # noinspection PyProtectedMember
        logger.setLevel(logging._checkLevel(logger_conf['level']))

        dict_configurator = logging.config.DictConfigurator(dict_config)

        formatters = dict_configurator.config.get('formatters', {})
        for fname in formatters:
            try:
                formatters[fname] = dict_configurator.configure_formatter(formatters[fname])
            except Exception as e:
                raise ValueError('Unable to configure formatter %r: %s' % (fname, e))

        for handler_name in logger_conf['handlers']:
            if handler_name not in self.handlers:
                # important to use configuration passed to dict_configurator instead of dict_config
                # because it has been processed to change file
                is_DynamicFileHandler = dict_configurator.config['handlers'][handler_name][
                    "class"
                ] == "pyutils.logging_utils.DynamicFileHandler"
                handler = dict_configurator.configure_handler(dict_configurator.config['handlers'][handler_name])

                if is_DynamicFileHandler:
                    # compare with class_path because dict_configurator create a wrapped class
                    handler.set_logger_name(name)
                else:
                    self.handlers[handler_name] = handler
            else:
                handler = self.handlers[handler_name]

            logger.addHandler(handler)

        if 'filters' in logger_conf:
            raise NotImplementedError('Not support filters')

        return logger
