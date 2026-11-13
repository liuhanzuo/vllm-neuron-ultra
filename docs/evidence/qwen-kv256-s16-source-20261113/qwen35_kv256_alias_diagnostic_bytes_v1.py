"""Pure byte patterns/differences for the fixed Qwen KV alias diagnostic.

No files, model/native imports, current bindings or execution occur on import.
Bit i of differences marks byte offset i; every differing byte is represented.
"""
import hashlib
import struct

ROWS, TOKENS, WIDTH = 4, 4096, 256
STATE_BYTES = ROWS * TOKENS * WIDTH * 2
PROFILES = ('raw', 'finite', 'row_tagged_raw')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pattern(profile, row_words, *, salt, rows=ROWS):
    """Match original torch integer construction; add a separate tagged control.

    finite values are exactly representable in BF16, so the upper FP32 word
    reproduces the original finite construction without rounding ambiguity.
    """
    require(profile in PROFILES and type(row_words) is int and 0 < row_words <= TOKENS*WIDTH
            and type(rows) is int and 0 < rows <= ROWS, 'Bounded exact pattern geometry')
    result=[]
    for row in range(rows):
        start=row*row_words
        period=257 if profile=='finite' else 65536
        values=[]
        for index in range(period):
            if profile=='finite':
                value=((start+index+salt)%257-128)/16
                word=struct.unpack('<I',struct.pack('<f',value))[0]>>16
            else:
                word=((start+index)*637+salt+(row*257 if profile=='row_tagged_raw' else 0))%65536
            values.append(word)
        block=struct.pack('<'+'H'*period,*values)
        result.append((block*((row_words+period-1)//period))[:row_words*2])
    return b''.join(result)


def bf16_kind(word):
    exponent=word&0x7f80;fraction=word&0x7f
    if exponent==0x7f80:
        return ('quiet_nan' if word&0x40 else 'signaling_nan') if fraction else 'infinity'
    if exponent==0:
        return 'subnormal' if fraction else ('negative_zero' if word&0x8000 else 'positive_zero')
    return 'normal'


def compare(actual, expected, bucket, *, rows=ROWS, tokens=TOKENS, width=WIDTH, guard=lambda:None):
    """Return complete difference bitmap and bounded exact per-region metadata.

    Tiny explicit dimensions are for source-only synthetic tests. The real leaf
    always passes the fixed 4x1x4096x256 BF16 dimensions and bucket128/256.
    """
    require(type(actual) is bytes and type(expected) is bytes
            and len(actual)==len(expected)==rows*tokens*width*2 <= STATE_BYTES,
            'Complete equal-size BF16 storage')
    require(all(type(x)is int and x>0 for x in (rows,tokens,width,bucket))
            and bucket<=tokens and callable(guard), 'Explicit bounded partition geometry')
    guard();bitmap=bytearray((len(actual)+7)//8)
    regions=[]
    for row in range(rows):
        for part,start,end in (('prefix',0,bucket),('tail',bucket,tokens)):
            lo=(row*tokens+start)*width*2;hi=(row*tokens+end)*width*2
            regions.append(dict(row=row,partition=part,start_byte=lo,end_byte_exclusive=hi,
                bytes_compared=hi-lo,words_compared=(hi-lo)//2,differing_bytes=0,differing_words=0,
                actual_sha256=digest(actual[lo:hi]),expected_sha256=digest(expected[lo:hi])))
    differences=words=0;first=[];classes={}
    if actual!=expected:
        for index,(aa,bb) in enumerate(zip(struct.iter_unpack('<H',actual),struct.iter_unpack('<H',expected))):
            if index%65536==0:guard()
            a,b=aa[0],bb[0]
            if a==b:continue
            row=index//(tokens*width);token=index//width%tokens;column=index%width
            region=regions[row*2+(token>=bucket)];region['differing_words']+=1;words+=1
            offsets=[]
            for shift in (0,8):
                if (a>>shift)&255 != (b>>shift)&255:
                    offset=index*2+shift//8;bitmap[offset//8]|=1<<(offset%8)
                    differences+=1;region['differing_bytes']+=1;offsets.append(offset)
            pair=bf16_kind(b)+'->'+bf16_kind(a);classes[pair]=classes.get(pair,0)+1
            if len(first)<64:
                first.append(dict(word_offset=index,byte_offset=offsets[0],differing_byte_offsets=offsets,
                    row=row,token=token,feature=column,partition=region['partition'],
                    expected_word_hex=f'{b:04x}',actual_word_hex=f'{a:04x}',
                    expected_class=bf16_kind(b),actual_class=bf16_kind(a)))
    guard();proof=dict(bytes_compared=len(actual),actual_sha256=digest(actual),expected_sha256=digest(expected),
        identical=actual==expected,differing_bytes=differences,differing_words=words,
        first_differing_byte=None if not first else first[0]['byte_offset'],first_differing_words=first,
        preview_limit_words=64,preview_truncated=words>64,word_class_transitions=classes,
        partitions=regions,bitmap_encoding='bit_i_lsb_first_means_byte_offset_i_differs',
        bitmap_bytes=len(bitmap),bitmap_sha256=digest(bitmap),full_difference_positions_retained=True)
    require(sum(x['differing_bytes']for x in regions)==differences
            and sum(x['differing_words']for x in regions)==words,'Complete partition accounting')
    return proof,bytes(bitmap)


def validate_bitmap(proof, bitmap, actual, expected):
    require(len(bitmap)==proof['bitmap_bytes']==(len(actual)+7)//8 and digest(bitmap)==proof['bitmap_sha256'],
            'Exact complete difference bitset')
    require(len(actual)==len(expected)==proof['bytes_compared']
            and digest(actual)==proof['actual_sha256'] and digest(expected)==proof['expected_sha256'],
            'Exact complete comparison contents')
    count=0
    for i,(a,b)in enumerate(zip(actual,expected)):
        marked=bool(bitmap[i//8]&(1<<(i%8)))
        require(marked is (a!=b),'Omitted or invented difference position')
        count+=marked
    require(count==proof['differing_bytes'],'Full bitmap mismatch count')
    if len(actual)%8:require(bitmap[-1]>>(len(actual)%8)==0,'No out-of-range bits')
    return True
