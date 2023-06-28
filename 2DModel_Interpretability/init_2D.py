import argparse
from email.policy import default 
import os 

class Options():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):

        #Some parameters
        parser.add_argument('--dataset_folder', type=str,
                            default="/scratch_net/biwidl311/Cristina_Almagro/research_project/dataset/Cristina_dataset_preprocessed_2D.json",
                            help="JSON dataset containing both training and validation set")
        parser.add_argument('--dataset_folder_test', type=str,
                            default="/scratch_net/biwidl311/Cristina_Almagro/research_project/dataset/Cristina_dataset_test_preprocessed_tight.json",
                            help="JSON dataset containing test set")
        parser.add_argument('--model_folder', type=str,
                            # 2D deformable UNet architecture
                            default="/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/deformable/model_2D_DUNetV1V2_08_01_23_exp1/",                           # 2D normal UNet
                            # 2D UNet architecture
                            #default = "/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/deformable/model_2D_UNetV1V2_ds_03_04_23/",
                            )
        parser.add_argument("--workers", default=16, type=int,
                            # I used 2 workers for the other computers (not the A100)
                                help='number of data loading workers')
        parser.add_argument("--gpus", default='0', type=str, #default=0
                                help='Number of available GPUs, e.g 0 0,1,2 0,2 use -1 for CPU')
        parser.add_argument("--pretrain",
                            default=None,
                            #default='/scratch_net/biwidl311/Cristina_Almagro/research_project/attention/wrad11/Attention154.pth',  #None, #15 - 178
                            type=str)

        #Model parameters
        parser.add_argument("--network", default='DUNetV1V2', help='UNetV1V2, DUNetV1V2')
        parser.add_argument("--spacing", default=[0.6, 0.6, 0.6], help="Resolution of the MRI")

        # Training parameters
        parser.add_argument("--batch_size", type=int, default=12)
        parser.add_argument("--validation_type", type=str, default='3D', help='2D or 3D')
        parser.add_argument('--in_channels', default=1, type=int, help='Channels of the input')
        parser.add_argument('--out_channels', default=2, type=int, help='Channels of the output')
        parser.add_argument('--epochs', default=700, type=int)
        parser.add_argument('--lr', default=0.0001, help="Learning rate")
        parser.add_argument('--loss', default='DiceCELoss')

        #Detection metrics
        # Inference parameters
        parser.add_argument('--num_models', default=5, type=int,
                            help="Predict inference in N best epochs according to the validation set, [1,5]")
        parser.add_argument("--outpth_inference",
                            default='/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/Cristina_Almagro/results/ResearchProject/model_2D_DUNetV1V2_08_01_23_exp1/',
                            # default='/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/Cristina_Almagro/results/ResearchProject/model_2D_DUNetV1V2_03_04_23_exp1/',
                            type=str)
        parser.add_argument("--date",
                            default='08_01_2023',
                            type=str)
        parser.add_argument("--dataset_new",
                            default=0,
                            type=str)

        self.initialized = True
        return parser
    
    def parser(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt = parser.parse_args()

        #Set gpus ids
        if opt.gpus != '-1':
            os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        
        return opt

