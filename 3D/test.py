import pickle
import numpy as np

for filename in ["Opt_buddha_.pkl","Opt_bunny_.pkl","Opt_dragon_.pkl"]:
    print("loading: "+filename)
    Func = pickle.load(open("SAVED_DATA/"+filename,"rb"))
    print("offset: ",0.01)
    perc1, err1 = Func.check_local_RMS_error(1,2)
    print("unsigned dist RMS error: ",np.mean(err1))
    perc2, err2 = Func.check_local_RMS_error_via_hicken(1,2)
    print("signed dist RMS error:   ",np.mean(err2))

    # perc1, err1 = Func.check_local_max_error(1,2)
    # print("unsigned dist Max error: ",np.mean(err1))
    # perc2, err2 = Func.check_local_max_error_via_hicken(1,2)
    # print("signed dist Max error:   ",np.mean(err1))

    print("offset: ",0.005)
    perc1, err1 = Func.check_local_RMS_error(0.5,2)
    print("unsigned dist RMS error: ",np.mean(err1))
    perc2, err2 = Func.check_local_RMS_error_via_hicken(0.5,2)
    print("signed dist RMS error:   ",np.mean(err2))

    # perc1, err1 = Func.check_local_max_error(0.5,2)
    # print("unsigned dist Max error: ",np.mean(err1))
    # perc2, err2 = Func.check_local_max_error_via_hicken(0.5,2)
    # print("signed dist Max error:   ",np.mean(err1))
