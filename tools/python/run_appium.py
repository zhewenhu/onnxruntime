#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import psutil
import signal
import sys
import subprocess

from util import get_logger

log = get_logger("run_appium")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manages the starting and stopping an Appium isntance.")

    parser.add_argument(
        "--start", action="store_true", help="Start an Appium process.")
    parser.add_argument(
        "--appium-path", help="Path to an Appium executable.")
    parser.add_argument(
        "--stop", action="store_true", help="Stop an Appium process.")

    args = parser.parse_args()

    if args.start and args.appium_path is None:
        raise ValueError("Appium path must be given on starting an Appium")

    return args


def get_appium_pid():
    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=['pid', 'name', 'cmdline'])
        except psutil.NoSuchProcess:
            pass
        else:
            if (pinfo['cmdline'] and 'node' in pinfo['name'] and 'appium' in " ".join(pinfo['cmdline'])):
                return pinfo['pid']
    return -1


_stop_signal = signal.CTRL_BREAK_EVENT if sys.platform.startswith("win") else signal.SIGTERM


def main():
    args = parse_args()

    if args.start:
        log.debug("Starting an Appium process")
        subprocess.Popen(
            args=["node", os.path.join(args.appium_path, "node_modules", "appium", "build", "lib", "main.js"),
                  "--allow-insecure", "chromedriver_autodownload"],
            creationflags=subprocess.CREATE_NEW_CONSOLE)
    elif args.stop:
        log.debug("Stopping an Appium process")
        pid = get_appium_pid()
        if pid != -1:
            psutil.Process(pid).kill()


if __name__ == "__main__":
    sys.exit(main())
