
########################################################################################
# 1. Combine predictions of the 5 best models: majority voting
# 2. Report recall, precision, F1, DSC, DSC on small lesions, DSC based on size
########################################################################################
# Cristina Almagro Pérez, 2023, ETH Zürich, Biomedical Imaging Computing lab

from init_2D import Options
opt = Options().parser()
########################################################################################
# Please specify the following in the init.py file:
combine_predictions = 1
report_metrics = 1

########################################################################################
if combine_predictions:
    if opt.num_models > 1:
        print("#### Combine predictions of best epochs ####")
        with open("fivefold_mymodel_2Dmodel.py") as f:
            exec(f.read())
#Only implemented when combining 5 epochs
#########################################################################################
if report_metrics:
    print("############################## Reporting metrics ######################################")
    with open("mymodel2D/evaluation2Dmodels.py") as f:
        exec(f.read())




