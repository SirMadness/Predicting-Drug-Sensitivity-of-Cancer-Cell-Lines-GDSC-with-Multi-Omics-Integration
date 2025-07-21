#!/usr/bin/python

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:00:00 2023
"""


import platform
import shutil
import subprocess

def get_os():
    """
    Returns:
      'mac'   if macOS,
      'linux' if Linux,
      otherwise the raw platform.system().lower()
    """
    name = platform.system().lower()
    if name == 'darwin':
        return 'mac'
    if name == 'linux':
        return 'linux'
    return name

def has_nvidia_smi():
    """
    Quick check: is the nvidia-smi binary on the PATH?
    """
    return shutil.which('nvidia-smi') is not None

def probe_nvidia_smi():
    """
    Runs `nvidia-smi -L` to list GPUs. Returns True if at least one GPU is found.
    """
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '-L'],
            stderr=subprocess.DEVNULL,
            universal_newlines=True
        )
        # output looks like:
        #   GPU 0: GeForce GTX 1080 (UUID: GPU-...)
        #   GPU 1: ...
        return bool(output.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    
def get_gpu_if_present():
    """
    Returns:
      A string with GPU information if available, otherwise 'No NVIDIA GPU detected'.
    """
    if has_nvidia_smi():
        if probe_nvidia_smi():
            return True
        else:
            return False
    else:
        return False
# Main execution
if __name__ == "__main__":
    os_name = get_os()
    print(f"Running on: {os_name}")
    if has_nvidia_smi() and probe_nvidia_smi():
        print("NVIDIA GPU detected via nvidia-smi 🎉")
    else:
        print("No NVIDIA GPU detected (nvidia-smi not found or no GPUs).")

# This script is used to determine the platform and GPU availability.
# It checks if the system is macOS or Linux, and if an NVIDIA GPU is available.
# It uses the `nvidia-smi` command to probe for NVIDIA GPUs.
# This is useful for configuring GPU-accelerated training in machine learning workflows.
# The script can be run directly to see the platform and GPU status.
# It is typically used in environments where GPU acceleration is desired, such as deep learning tasks.
# The output will indicate the operating system and whether an NVIDIA GPU is available.