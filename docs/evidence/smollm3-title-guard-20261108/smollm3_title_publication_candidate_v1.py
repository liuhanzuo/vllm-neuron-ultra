"""Dedicated intent publication with original guards; source candidate only."""
import hashlib,json,re,time
import smollm3_title_deadline_candidate_v1 as runtime
from smollm3_observer_invariance_protocol_v18 import require,MIB,directory

def record_engine_title_before(self,hook,witness):
    with self.callback_lock:
        self.poll()
        require(self.title_witness is hook and hook.called and not hook.committed
                and self._title_token is None,'one exact called title hook transaction')
        actual=runtime.observe(self.permit)
        before=runtime.title.make_before(self.permit,actual,witness,time.monotonic())
        name=runtime.names(self.permit['pid'])['before']
        raw=(json.dumps(before,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
        require(re.fullmatch('[A-Za-z0-9_-]+\.(json|raw)',name) and type(raw)is bytes and len(raw)<=4*MIB,'Bounded exclusive control/request receipt')
        result=self.external_writer(name,raw)
        require(result['durable'] is True and result['exclusive'] is True and result['path']==str(directory(self.row['run_id'])/name)
                and result['sha256']==hashlib.sha256(raw).hexdigest() and result['bytes']==len(raw),'Actual exclusive durable receipt')
        self.poll()
        return before
