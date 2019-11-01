eta = 0
hh = int(eta//3600)
mm = int((eta//60)%60)
ss = int(eta%60)

print(f"{hh:02d}:{mm:02d}:{ss:02d}")
