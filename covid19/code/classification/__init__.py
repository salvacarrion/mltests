import os
import sys

if int(os.getenv("REMOTE_GPU", 0)) == 1:
    print("REMOTE ENVIRONMENT")
    sys.path.append("/home/scarrion/projects/mltests/")
else:
    print("LOCAL ENVIRONMENT")
