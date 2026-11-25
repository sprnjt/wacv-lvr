import base64,sys
from pathlib import Path
H=Path(".")
src={"__CH__":H/"CultureHaystack_handoff/wacv-lvr/ch.py","__RM__":H/"CultureHaystack_handoff/wacv-lvr/run_models.py",
     "__G2__":H/"hf_release/culturehaystack-review/post_registration/fullpool_package/ext/ch_g2.py"}
t=open('runner_template.py').read()
for k,p in src.items(): t=t.replace(k,base64.b64encode(p.read_bytes()).decode())
acct,job,dtype,out=sys.argv[1:5]
gpuq=sys.argv[5]
t=t.replace("__ACCOUNT__",acct).replace("__JOB__",job).replace("__DTYPE__",dtype).replace("__GPUQ__",gpuq)
Path(out).parent.mkdir(parents=True,exist_ok=True); open(out,'w').write(t); print("wrote",out,len(t))
