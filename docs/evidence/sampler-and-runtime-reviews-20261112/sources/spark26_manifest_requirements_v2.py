"""Finite stdlib-only requirement/marker grammar; unknown syntax is unresolved.

No installed packaging module is imported. This parser does not install/solve
packages. It checks the fixed installed releases selected by original metadata.
"""
import ast
import re


class Unresolved(ValueError):pass


def require(ok,message):
    if not ok:raise Unresolved(message)


def name(value):
    require(bool(re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?',value)),'Unsupported distribution/extra name')
    return re.sub('[-_.]+','_',value).lower()


def version(value):
    match=re.fullmatch(r'v?(?:(\d+)!)?(\d+(?:\.\d+)*)(?:[-_.]?(a|b|rc)[-_.]?(\d+))?(?:[-_.]?post[-_.]?(\d+))?(?:[-_.]?dev[-_.]?(\d+))?(?:\+([a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*))?',value,re.I)
    require(match is not None,'Unsupported installed/specifier version: '+value)
    epoch,release,pre,pre_n,post,dev,local=match.groups()
    release=tuple(int(x)for x in release.split('.'))
    require(len(release)<=16,'Bounded release components')
    padded=release+(0,)*(16-len(release))
    prekey=(-1,0) if pre is None and post is None and dev is not None else (4,0) if pre is None else ({'a':0,'b':1,'rc':2}[pre.lower()],int(pre_n))
    key=(int(epoch or 0),padded,prekey,int(post)if post is not None else -1,int(dev)if dev is not None else float('inf'))
    return dict(text=value,release=release,epoch=int(epoch or 0),key=key,pre=pre,post=post,dev=dev,local=local)


def compare(actual,operator,wanted):
    if operator=='===':return actual.lower()==wanted.lower()
    if wanted.endswith('.*'):
        require(operator in ('==','!='),'Wildcard only in equality clauses')
        a,b=version(actual),version(wanted[:-2])
        require(b['pre'] is b['post'] is b['dev'] is b['local'] is None,'Release-only wildcard')
        equal=a['epoch']==b['epoch'] and a['key'][1][:len(b['release'])]==b['release']
        return equal if operator=='==' else not equal
    a,b=version(actual),version(wanted)
    if b['local'] is not None:
        require(operator in ('==','!='),'Local version only in equality clauses')
        def local_key(text):
            return tuple(int(x)if x.isdigit()else x.lower()for x in re.split('[-_.]',text))if text else ()
        equal=a['key']==b['key'] and local_key(a['local'])==local_key(b['local'])
        return equal if operator=='==' else not equal
    if operator in ('==','!='):
        equal=a['key']==b['key'];return equal if operator=='==' else not equal
    if operator=='~=':
        require(len(b['release'])>=2,'Compatible release needs at least two components')
        prefix=b['release'][:-1]
        return a['key']>=b['key'] and a['epoch']==b['epoch'] and a['key'][1][:len(prefix)]==prefix
    require(operator in ('<','<=','>','>='),'Unknown comparison operator')
    # Strict public-release exclusions from PEP 440, conservatively rejecting
    # unknown spellings before evaluating rather than coercing them to strings.
    same_release=a['epoch']==b['epoch'] and a['key'][1]==b['key'][1]
    if operator=='<' and b['pre'] is None and b['dev'] is None and same_release and (a['pre'] is not None or a['dev'] is not None):return False
    if operator=='>' and same_release and ((b['post'] is None and a['post'] is not None) or a['local'] is not None):return False
    return {'<':a['key']<b['key'],'<=':a['key']<=b['key'],'>':a['key']>b['key'],'>=':a['key']>=b['key']}[operator]


def satisfies(actual,specifiers):
    if not specifiers:return True
    if specifiers.startswith('('):
        require(specifiers.endswith(')'),'Closed requirement specifier parentheses');specifiers=specifiers[1:-1].strip()
    result=True
    for clause in specifiers.split(','):
        match=re.fullmatch(r'\s*(===|~=|==|!=|<=|>=|<|>)\s*([^\s,]+)\s*',clause)
        require(match is not None,'Unsupported requirement version clause: '+clause)
        result=compare(actual,match[1],match[2]) and result
    return result


TOKEN=re.compile(r'''\s*(?:(?P<string>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(?P<op>===|~=|==|!=|<=|>=|<|>|\(|\))|(?P<word>[A-Za-z_][A-Za-z0-9_]*))''')


class Marker:
    def __init__(self,text,environment):
        self.tokens=[];self.position=0;self.environment=environment;offset=0
        while offset<len(text):
            if not text[offset:].strip():break
            match=TOKEN.match(text,offset);require(match is not None,'Unsupported marker syntax: '+text[offset:])
            self.tokens.append((match.lastgroup,match.group(match.lastgroup)));offset=match.end()
        require(len(self.tokens)<=512,'Finite marker grammar')

    def peek(self,value):return self.position<len(self.tokens)and self.tokens[self.position][1]==value
    def take(self,value):
        if self.peek(value):self.position+=1;return True
        return False
    def operand(self):
        require(self.position<len(self.tokens),'Missing marker operand')
        kind,value=self.tokens[self.position];self.position+=1
        if kind=='string':return ast.literal_eval(value),None
        require(kind=='word' and value not in ('and','or','not','in'),'Expected marker environment variable')
        return self.environment.get(value),value
    def atom(self):
        if self.take('('):
            result=self.disjunction();require(self.take(')'),'Unclosed marker parentheses');return result
        left,lvar=self.operand()
        require(self.position<len(self.tokens),'Missing marker operator')
        operator=self.tokens[self.position][1];self.position+=1
        if operator=='not':require(self.take('in'),'Only not in is supported');operator='not in'
        require(operator in ('===','~=','==','!=','<=','>=','<','>','in','not in'),'Invalid marker comparison')
        right,rvar=self.operand();require(lvar or rvar,'At least one marker operand is a variable')
        if left is None or right is None:return None
        if 'extra'in (lvar,rvar):
            left=re.sub('[-_.]+','-',left).lower();right=re.sub('[-_.]+','-',right).lower()
        if operator in ('in','not in'):return (left in right) if operator=='in' else (left not in right)
        try:return compare(left,operator,right)
        except Unresolved:
            require(not {'python_version','python_full_version','implementation_version'}&{lvar,rvar},
                    'Unsupported target version marker cannot fall back to lexical ordering')
            require(operator in ('==','!=','<=','>=','<','>'),'Unsupported string marker comparison')
            return {'==':left==right,'!=':left!=right,'<':left<right,'<=':left<=right,'>':left>right,'>=':left>=right}[operator]
    def conjunction(self):
        result=self.atom()
        while self.take('and'):
            other=self.atom();result=False if result is False or other is False else None if result is None or other is None else True
        return result
    def disjunction(self):
        result=self.conjunction()
        while self.take('or'):
            other=self.conjunction();result=True if result is True or other is True else None if result is None or other is None else False
        return result
    def evaluate(self):
        result=self.disjunction();require(self.position==len(self.tokens),'Unconsumed marker tokens');return result


def marker(text,environment,extras):
    if not text:return True
    values=[]
    for extra in sorted({''}|set(extras)):
        env=dict(environment,extra=extra.replace('_','-'))
        values.append(Marker(text,env).evaluate())
    return True if True in values else None if None in values else False


def requirement(text):
    head,separator,condition=text.partition(';')
    match=re.fullmatch(r'\s*([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\s*\[([^\]]*)\])?\s*(.*?)\s*',head)
    require(match is not None,'Unsupported requirement: '+text)
    package,extras,specifiers=match.groups()
    return dict(name=name(package),extras=sorted({name(x.strip())for x in extras.split(',')if x.strip()})if extras else [],
                specifiers=specifiers,marker=condition.strip()if separator else '',url=specifiers[1:].strip()if specifiers.startswith('@')else None)
