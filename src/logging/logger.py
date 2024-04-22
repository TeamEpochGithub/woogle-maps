"""A logger that logs to the terminal."""

import logging
import sys
from types import TracebackType
from typing import Any

from epochalyst._core._logging._logger import _Logger
from epochalyst.logging.section_separator import print_section_separator

logger = logging.getLogger("logger")


def log_exception(exc_type: type[BaseException], exc_value: BaseException, exc_traceback: TracebackType | None) -> None:
    """Log any uncaught exceptions except KeyboardInterrupts.

    Based on https://stackoverflow.com/a/16993115.

    :param exc_type: The type of the exception.
    :param exc_value: The exception instance.
    :param exc_traceback: The traceback.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("A wild %s appeared!", exc_type.__name__, exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = log_exception


class Logger(_Logger):
    """A logger that logs to the terminal.

    To use this logger, inherit, this will make the following methods available:
    - log_to_terminal(message: str) -> None
    - log_to_debug(message: str) -> None
    - log_to_warning(message: str) -> None
    - log_to_external(message: dict[str, Any], **kwargs: Any) -> None
    - external_define_metric(metric: str, metric_type: str) -> None
    """

    def log_to_terminal(self, message: str) -> None:
        """Log a message to the terminal.

        :param message: The message to log.
        """
        logger.info(message)

    def log_to_debug(self, message: str) -> None:
        """Log a message to the debug level.

        :param message: The message to log.
        """
        logger.debug(message)

    def log_to_warning(self, message: str) -> None:
        """Log a message to the warning level.

        :param message: The message to log.
        """
        logger.warning(message)

    def log_to_external(self, message: dict[str, Any], **kwargs: Any) -> None:  # noqa: ANN401, DOC103
        """Log a message to an external service.

        :param message: The message to log.
        :param kwargs: Any additional arguments (UNUSED).
        """

    def external_define_metric(self, metric: str, metric_type: str) -> None:
        """Define a metric in an external service.

        :param metric: The metric to define
        :param metric_type: The type of the metric
        """

    def log_section_separator(self, message: str) -> None:
        """Log section separator method, if no logging override with empty.

        :param message: The message to log.
        """
        print_section_separator(message)
