import argparse
from email.policy import default
import os


class Options():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):

        #Some parameters
        parser.add_argument('--dataset_folder', type=str,
                            default="/scratch_net/biwidl311/Cristina_Almagro/research_project/dataset/Cristina_dataset_preprocessed.json",
                            help="JSON dataset containing both training and validation set")
        parser.add_argument('--dataset_folder_test', type=str,
                            default="/scratch_net/biwidl311/Cristina_Almagro/research_project/dataset/Cristina_dataset_test_preprocessed1.json",
                            help="JSON dataset containing test set")
        parser.add_argument('--model_folder', type=str,
                            default="/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/deformable/prueba/"
                            )
        parser.add_argument("--workers", default=16, type=int,  #default=16
                            # I used 2 workers for the other computers (not the big one)
                                help='number of data loading workers')
        parser.add_argument("--gpus", default='0,1', type=str,  #default=0    # 0,1,2,3 for DUNetV1V2 14_01_23
                                help='Number of available GPUs, e.g 0 0,1,2 0,2 use -1 for CPU')
        parser.add_argument("--pretrain",
                            default=None,
                            #default = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/deformable/model_3D_Attention_17_02_23_exp2/Attention10.pth',
                            type=str)
        parser.add_argument("--test_pretrain",
                            default='/scratch_net/biwidl311/Cristina_Almagro/research_project/attention/my_model1/Attention426.pth',
                            type=str)

        #Model parameter
        parser.add_argument("--network", default='ThreeOffsetsAttentionUNet', help='Attention, DUNetV1V2, ThreeOffsetsAttentionUNet, DeformAttention')
        parser.add_argument("--features", default=[16, 32, 64, 128, 256])  #[16, 32, 64, 128, 256]
        parser.add_argument("--patch_size", default=(128, 128, 128), help="Size of the patches extracted from the image") #128 128 64 for BrainMetShare #128 128 128 for mydataset
        parser.add_argument("--spacing", default=[0.6, 0.6, 0.6], help="Resolution of the MRI") #default=[0.94, 0.94, 1.0] #this is not true #my_dataset = [0.6,0.6,0.6]

        parser.add_argument("--batch_size", type=int, default=2)
        parser.add_argument("--num_samples", type=int, default=4, help="Number of patches extracted per patient. Input scan dimensions: 350 x 350 x 350") # use 1 or 2 for deformable networks
        parser.add_argument("--prob_met", type=float, default=0.5, help="Probability that the extracted patch has as center a metastasis pixel")
        parser.add_argument('--in_channels', default=1, type=int, help='Channels of the input')
        parser.add_argument('--out_channels', default=2, type=int, help='Channels of the output')

        #Training parameters
        parser.add_argument('--epochs', default=1000, type=int)
        parser.add_argument('--lr', default=0.0001, help="Learning rate")
        parser.add_argument('--update_lr', default=False, help="Use poly LR?")
        parser.add_argument('--loss', default='DiceCELoss', help="DiceCELoss, DiceLoss, DiceFocalLoss, GeneralizedDiceLoss, FocalLoss ")  #BlobLoss #DiceCELoss
        parser.add_argument('--deep_supervision', default=False, help="Only implemented for Attention U-Net")
        parser.add_argument('--ds_loss_weights', default=[0.533333, 0.266666, 0.133333, 0.066666, 0])

        # Inference parameters
        parser.add_argument('--outpth0', # Path to save the predictions
                            default="/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/research_project_results/3Dmodels/prueba/")
        parser.add_argument('--date', default="30_01_2023")


        # #Detection metrics
        # parser.add_argument('--pth_gt',
        #                     default='/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/masks/',
        #                     type=str)
        # parser.add_argument('--pth_pred',
        #                     default='/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/research_project_results/attention/mymodel1/ensemble_02_12_2022/',
        #                     type=str)
        # parser.add_argument('--format_output',
        #                     default='.npy',
        #                     type=str)
        # parser.add_argument('--small_objects',
        #                     default=False,
        #                     type=bool)
        #parser.add_argument('--metric', default = 'dice', type =str, help = "Can also be HD (Hausdorff distance)")
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

