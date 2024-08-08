import pdb
from sw_data.data import *

# df = single_iri_ecmwf("2015-05-14", "precip", "all",
#                       forecast_type="reforecast",
#                       run_type="perturbed",
#                       grid="global1_5", recompute=True)

# # 20150501-20211231
df = iri_ecmwf("2015-05-14", "2015-06-14",
               "precip", "all",
               forecast_type="reforecast",
               run_type="perturbed",
               grid="global1_5")
            #    remote=True,
            #    remote_config={
                #    "name": "sheerwater-genevieve",
            #        "worker_memory": "50GiB",
            #        "worker_cpu": 10,
            #    })

pdb.set_trace()
