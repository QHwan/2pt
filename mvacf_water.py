# calculate the hb_density(r)
# input file :
#		water_coordinate_file, water_com_coordinate_file, hbmap_file, hbindex_file

import numpy as np 
import sys
import math
import time
from mvacf_water import main
import MDAnalysis as md

start = time.time()

main()

end = time.time() - start
print(end)
