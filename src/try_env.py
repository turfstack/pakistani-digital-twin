from pbpk import load_params, simulate
from environments import load_environment

p = load_params("acetaminophen")
for e in ["earth", "space", "mars"]:
    r = simulate(p, 1.0, env=load_environment(e))
    print(e, round(r["AUC"], 2), round(r["Cmax"], 2), round(r["tmax"], 2))