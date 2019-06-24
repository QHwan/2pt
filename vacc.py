import numpy as np
import time
import MDAnalysis as md
import GromacsAnalysis
from GromacsAnalysis import (pyutil, option)

@pyutil.timefn
def main():
    opt = option.Option()
    args = opt.args

    trajfilename = args.trajfilename
    tprfilename = args.tprfilename
    ofilename = args.ofilename

    u = md.Universe(tprfilename, trajfilename)
    selection = "all"
    atom_selection = u.select_atoms(selection)

    vac = GromacsAnalysis.VACC(u, selection, tau=3.0)
    vac.run()
    pyutil.save(ofilename, vac.vacc)


main()




