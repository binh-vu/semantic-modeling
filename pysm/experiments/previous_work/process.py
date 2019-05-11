#!/usr/bin/python
# -*- coding: utf-8 -*-
import subprocess
from pathlib import Path
from typing import Tuple, Optional, Union

import time


def invoke_command(command: str, output2file: Optional[Union[str, Path]] = None,
                   check_call: bool = True, output2stdout: bool=True) -> int:
    p = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, cwd="/tmp"
    )

    if output2stdout:
        if output2file:
            with open(output2file, "w") as f:
                for line in iter(p.stdout.readline, ""):
                    print(line, end="")
                    f.write(line)
        else:
            for line in iter(p.stdout.readline, ""):
                print(line, end="")
    else:
        if output2file:
            with open(output2file, "w") as f:
                for line in iter(p.stdout.readline, ""):
                    f.write(line)
        else:
            for line in iter(p.stdout.readline, ""):
                pass

    p.stdout.close()
    return_code = p.wait(timeout=10)
    if return_code and check_call:
        raise subprocess.CalledProcessError(return_code, command)

    return return_code
