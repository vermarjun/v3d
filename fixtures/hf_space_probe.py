import time, shutil, sys
from gradio_client import Client, handle_file
t=time.time()
c=Client("facebook/map-anything", verbose=False)
g=c.predict(files=[handle_file(sys.argv[1])], interval=0.5, api_name="/update_gallery_on_unified_upload")
print("upload ok", round(time.time()-t,1), "s;", str(g[1])[:200])
t=time.time()
r=c.predict(frame_filter="All", show_cam=False, filter_black_bg=False, filter_white_bg=False, apply_mask=True, show_mesh=True, api_name="/gradio_demo")
print("recon", round(time.time()-t,1), "s")
print("glb:", r[0]); print("log:", str(r[1])[:300])
shutil.copy(r[0], "out.glb")
m=c.predict(current_view_selector="View 1", api_name="/measure")
print("measure:", m)
