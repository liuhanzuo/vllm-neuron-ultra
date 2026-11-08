"""Linux O_PATH identity: no file-body read, result cache, or import-time I/O.

The kernel resolves the actual open path; fstat and /proc/self/fd identify the
same opened file. File descriptors close on every success/failure path.
"""
import os

def identity(value):
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)

def file_stat(path):
    fd=os.open(path,os.O_PATH|os.O_CLOEXEC)
    try:
        value=os.fstat(fd)
        resolved=os.readlink('/proc/self/fd/'+str(fd))
        if identity(value)!=identity(os.fstat(fd)):
            raise RuntimeError('File identity changed during O_PATH observation')
        return dict(device=value.st_dev,inode=value.st_ino,size=value.st_size,
            mtime_ns=value.st_mtime_ns,ctime_ns=value.st_ctime_ns,realpath=resolved)
    finally:os.close(fd)
