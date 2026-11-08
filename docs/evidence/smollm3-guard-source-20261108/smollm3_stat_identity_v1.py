"""Stat-only source/payload identity; no result cache or import-time I/O."""
import os

def file_stat(path):
    value=os.stat(path)
    return dict(device=value.st_dev,inode=value.st_ino,size=value.st_size,
        mtime_ns=value.st_mtime_ns,ctime_ns=value.st_ctime_ns,
        realpath=os.path.realpath(path,strict=True))
