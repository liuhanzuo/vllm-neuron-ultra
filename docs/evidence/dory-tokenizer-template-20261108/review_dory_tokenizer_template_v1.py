"""Post-exit stdlib review of saved prompt/token evidence; no tokenizer rerun."""
import ast, hashlib, json, sys
from pathlib import Path
Q=Path(__file__).resolve().parent;sys.path[:0]=[str(Q),str(Q.parent)]
import run_dory_tokenizer_template_v1 as t
EXIT=dict(path=str(Q/'dory-tokenizer-template-r1-20261108-v1-actual-external-exit.json'),sha256='a4868c6ebc94f738fe0e5d2f078de23ba2f42500274f5be8b112f847940eac72')

def main():
    t.d.m.p.user_bus();fd,owner,lock=t.d.m.old.post.external.acquire_lock()
    try:
        t.d.m.p.clear_window();t.d.m.p.active_heavy();ex=t.d.load(EXIT)
        t.d.actual_exit(ex['actual_exit']);t.d.require(ex['actual_tool_exit_observed'] and ex['exact_owner_gone'],'Actual successful exit')
        result=t.d.load(ex['result']);t.d.m.same(ex['owner'],result['owner'],'Native owner binding');t.d.m.p.gone(result['owner'])
        plan=t.d.load(result['plan']);capture=t.d.load(result['capture']);admission=t.d.load(result['admission'])
        t.d.m.same(lock,admission['lock_identity'],'Same serial lock');t.d.m.same(plan['sources'],t.local_sources(),'Sources retained')
        t.d.require(result['passed'] and result['complete'],'Complete native result')
        cases=capture['cases'];t.d.m.same([x['name']for x in cases],[x['name']for x in plan['fixtures']],'Exact case ordering')
        for row,fixture in zip(cases,plan['fixtures'],strict=True):
            text=row['rendered'];ids=row['input_ids']
            t.d.m.same(hashlib.sha256(text.encode()).hexdigest(),row['sha256'],'Complete prompt byte hash')
            if fixture['expected']is not None:t.d.m.same(text,fixture['expected'],'Saved independent prompt expectation')
            else:
                t.d.require(text.endswith(fixture['required_suffix']),'Saved tools suffix')
                for fragment in fixture['required_fragments']:t.d.require(fragment in text,'Saved tools schema')
            t.d.require(len(ids)==row['tokens'] and all(type(x)is int and 0<=x<131072 for x in ids),'Complete bounded saved tokens')
        original=json.loads((t.META/'tokenizer.json').read_text());added={x['id']:x['content']for x in original['added_tokens']}
        t.d.m.same(capture['specials'],[dict(id=x['id'],content=x['content'])for x in original['added_tokens']],'All1000 saved special token values')
        # Independently decode all saved IDs from the original JSON vocabulary.
        # This verifies serialization and byte-level decode; it does not re-run BPE.
        vocab={value:key for key,value in original['model']['vocab'].items()}
        byte_values=list(range(ord('!'),ord('~')+1))+list(range(0xA1,0xAD))+list(range(0xAE,0x100))
        characters=list(byte_values);extra=0
        for value in range(256):
            if value not in byte_values:byte_values.append(value);characters.append(256+extra);extra+=1
        inverse={chr(character):value for value,character in zip(byte_values,characters,strict=True)}
        for row in cases:
            out=bytearray()
            for token_id in row['input_ids']:
                if token_id in added:out.extend(added[token_id].encode())
                else:out.extend(inverse[char]for char in vocab[token_id])
            t.d.m.same(out.decode('utf-8'),row['rendered'],'Independent byte-level saved ID decode')
        t.d.m.same(capture['generation_policy'],json.loads((t.META/'generation_config.json').read_text()),'Original policy retained exactly')
        warning=t.d.load(ex['warning_review']);source=Path(warning['source']['path']);t.d.m.same(t.d.ref(source),warning['source'],'Warning source retained')
        f=next(x for x in ast.walk(ast.parse(source.read_text()))if isinstance(x,ast.FunctionDef)and x.name=='_patch_mistral_regex')
        regex=next(x.value for x in ast.walk(f)if isinstance(x,ast.Constant)and isinstance(x.value,str)and x.value.startswith('[^\\r\\n\\p{L}'))
        t.d.m.same(regex,original['pre_tokenizer']['pretokenizers'][0]['pattern']['Regex'],'Original and replacement regex equality')
        config=json.loads((t.META/'config.json').read_text());t.d.m.same((config['model_type'],config.get('transformers_version')),('dory',None),'Exact heuristic trigger')
        files=[]
        for path in sorted(t.OUT.rglob('*')):
            if path.is_file():files.append(dict(path=str(path),bytes=path.stat().st_size,sha256=t.d.ref(path)['sha256']))
        t.d.m.same({n:t.d.m.p.sha(t.R/n)for n in t.d.m.CANONICAL},t.d.m.CANONICAL,'Canonical retained')
        terminal=t.d.once(Q/'dory-tokenizer-template-r1-20261108-v1-root-terminal.json',dict(schema='dory_tokenizer_template_root_terminal_v1',complete=True,passed=True,reviewer='/root',verifier=owner,native_owner=result['owner'],lock_identity=lock,actual_exit=EXIT,result=ex['result'],capture=result['capture'],plan=result['plan'],warning_review=ex['warning_review'],verifier_source=t.d.ref(Path(__file__).resolve()),saved_files=files,
            cases=13,exact_prompt_oracles=12,tool_schema_fragment_case=1,special_ids=1000,independent_saved_token_byte_decode=True,independent_BPE_execution=False,native_owner_gone=True,actual_verifier_exit_claimed=False,model_forward_executed=False,full_model_qualified=False,new_speedup_claim=False))
        print(json.dumps(dict(terminal=terminal,owner=owner)),flush=True)
    finally:t.d.m.old.post.external.release_lock(fd)
if __name__=='__main__':main()
