from typing import *

from karma_cli.app_misc import CliSemanticType
from semantic_modeling.karma.semantic_model import SemanticType


class SemanticTyping:

    def __init__(self):
        pass

    def suggest(self, attr: str) -> List[CliSemanticType]:
        return []