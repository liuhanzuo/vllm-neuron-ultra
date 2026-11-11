"""Finite signed archive transport and owned bounded decompressor; no installs."""
import hashlib,os,selectors,stat,subprocess,time,urllib.request,ssl
from pathlib import Path
from dory_ubuntu_metadata_adapter_v6 import NoRedirect,identity
MIB=2**20

def need(ok,msg):
    if not ok:raise ValueError(msg)

def file_ref(path,guard,expected=None,cap=64*MIB,expected_nlink=1):
    need(type(expected_nlink)is int and expected_nlink in (1,2),'Only reviewed link counts')
    path=Path(path);guard();before=path.lstat()
    need(path.resolve()==path and stat.S_ISREG(before.st_mode) and before.st_nlink==expected_nlink and 0<before.st_size<=cap,'Bounded canonical regular input')
    h=hashlib.sha256();n=0;fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb')as f:
        need(identity(os.fstat(f.fileno()))==identity(before),'Opened exact input')
        while True:
            guard();b=f.read(65536)
            if not b:break
            n+=len(b);need(n<=cap,'Input byte cap');h.update(b)
        need(identity(os.fstat(f.fileno()))==identity(before)==identity(path.lstat()),'Stable entire file')
    ref=dict(path=str(path),bytes=n,sha256=h.hexdigest())
    if expected is not None:need(all(ref[k]==expected[k]for k in ('bytes','sha256')),'Signed complete file size/hash')
    return ref,dict(zip(('dev','ino','mode','nlink','size','mtime_ns','ctime_ns'),identity(before)))

def download(row,path,guard,emit):
    need(type(row['bytes'])is int and 0<row['bytes']<=64*MIB,'Signed archive size bound')
    url=row['url'];need(url.startswith('https://archive.ubuntu.com/ubuntu/pool/') and '?'not in url and '#'not in url,'Fixed admitted official pool address')
    path=Path(path);need(not path.exists(),'Fresh archive file')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    req=urllib.request.Request(url,headers={'Accept-Encoding':'identity','User-Agent':'trn1-deb-member-verification/1.0'})
    start=time.monotonic();h=hashlib.sha256();total=0
    emit(dict(event='archive_download_start',package=row['package'],url=url,expected=dict(bytes=row['bytes'],sha256=row['sha256'])))
    guard()
    with opener.open(req,timeout=5)as response:
        need(response.status==200 and response.geturl()==url and response.headers.get('Content-Encoding','identity').lower()=='identity','Exact unencoded HTTPS response')
        lengths=response.headers.get_all('Content-Length',[])
        need(len(lengths)<=1 and (not lengths or lengths[0]==str(row['bytes'])),'Exact signed Content-Length')
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb')as f:
            while True:
                guard();need(time.monotonic()-start<60,'Archive request deadline')
                b=response.read1(min(65536,row['bytes']-total+1))
                if not b:break
                total+=len(b);need(total<=row['bytes'],'No archive overflow');h.update(b);need(f.write(b)==len(b),'Complete archive write')
            f.flush();os.fsync(f.fileno())
    need(total==row['bytes']and h.hexdigest()==row['sha256'],'Full signed archive bytes')
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(directory)
    finally:os.close(directory)
    ref,ident=file_ref(path,guard,row);emit(dict(event='archive_download_complete',package=row['package'],saved=ref,identity=ident));return ref

class TarPipe:
    """Never create a new process group: the worker owns all decompressor children."""
    def __init__(self,archive,tool,guard,emit):
        self.guard,self.emit=guard,emit;self.start=time.monotonic();self.stderr=bytearray();self.buffer=bytearray();self.total=0;self.returncode=None
        file_ref(tool['path'],guard,tool,16*MIB)
        self.argv=[tool['path'],'--fsys-tarfile',str(archive)]
        emit(dict(event='dpkg_deb_start',argv=self.argv))
        self.process=subprocess.Popen(self.argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,close_fds=True,start_new_session=False,env={'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C'},cwd=str(Path(archive).parent))
        self.selector=selectors.DefaultSelector()
        for name,stream in [('stdout',self.process.stdout),('stderr',self.process.stderr)]:os.set_blocking(stream.fileno(),False);self.selector.register(stream,selectors.EVENT_READ,name)
    def pump(self):
        self.guard();need(time.monotonic()-self.start<60,'Decompressor wall bound')
        for key,_ in self.selector.select(.05):
            b=os.read(key.fileobj.fileno(),65536)
            if not b:self.selector.unregister(key.fileobj);continue
            if key.data=='stdout':
                self.total+=len(b);need(self.total<=256*MIB,'Whole decompressed stream cap');self.buffer.extend(b)
            else:self.stderr.extend(b);need(len(self.stderr)<=MIB,'Complete decompressor stderr cap')
    def read(self,n):
        need(type(n)is int and 0<n<=65536,'Bounded tar reader request')
        while not self.buffer:
            if not self.selector.get_map():
                while self.process.poll()is None:self.pump()
                self.returncode=self.process.returncode;need(self.returncode==0,'Successful decompressor actual exit');return b''
            self.pump()
        b=bytes(self.buffer[:n]);del self.buffer[:n];return b
    def close(self):
        if self.process.poll()is None:self.process.kill()
        self.process.wait(timeout=5);self.selector.close();self.process.stdout.close();self.process.stderr.close()
    def receipt(self):
        need(not self.buffer and not self.selector.get_map()and self.returncode==0,'Full stream/pipe EOF and actual exit')
        return dict(argv=self.argv,returncode=0,stdout_bytes=self.total,stderr=self.stderr.decode('utf8','backslashreplace'),stderr_sha256=hashlib.sha256(self.stderr).hexdigest())

def inspect(row,archive,tool,guard,emit):
    from dory_deb_tar_members_v2 import verify
    ref,before=file_ref(archive,guard,row)
    targets={t['observed_path']:t['archive_member'] for t in row['targets']}
    # Saved package aliases are required; every alias must reach one fixed target.
    aliases={n.lstrip('/'):None for n in row['saved_package_names']}
    for name in list(aliases):
        exact=[k for k,v in targets.items()if v==name]
        if exact:aliases[name]=exact[0]
        elif row['package']=='libc6' and name=='lib64/ld-linux-x86-64.so.2':aliases[name]='/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2'
        elif row['package']=='libtcmalloc-minimal4' and name=='usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4':aliases[name]='/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.5.9'
        need(aliases[name] is not None,'Fixed known archive alias mapping')
    pipe=TarPipe(archive,tool,guard,emit)
    try:
        result=verify(pipe.read,guard,targets,aliases);receipt=pipe.receipt()
    except BaseException as error:
        cleanup_error=None
        try:pipe.close()
        except BaseException as secondary:cleanup_error=repr(secondary)
        failure=dict(event='dpkg_deb_failed',package=row['package'],argv=pipe.argv,
            returncode=pipe.process.returncode,stdout_bytes_received=pipe.total,
            stdout_bytes_unconsumed=len(pipe.buffer),stderr_captured_hex=bytes(pipe.stderr).hex(),
            stderr_captured_sha256=hashlib.sha256(pipe.stderr).hexdigest(),
            capture_complete=False,remaining_pipe_bytes_not_captured=True,
            original_error_type=type(error).__name__,original_error=str(error),cleanup_error=cleanup_error)
        try:emit(failure)
        except BaseException as secondary:
            error.dory_failure_event_error=repr(secondary)
        raise
    finally:pipe.close()
    after,after_identity=file_ref(archive,guard,row);need(after==ref and after_identity==before,'Original complete archive bytes and identity unchanged after decoding')
    file_ref(tool['path'],guard,tool,16*MIB)
    emit(dict(event='dpkg_deb_complete',package=row['package'],receipt=receipt))
    return dict(archive=ref,archive_identity=before,tar=result,decompressor=receipt)
