from .base_options import BaseOptions


def str2bool(v):
    return v.lower() in ('true')


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--which_epoch', type=str, default='200000', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.parser.add_argument("--Arc_path", type=str, default='./arcface_model/arcface_checkpoint.tar', help="run ONNX model via TRT")
        self.parser.add_argument("--pic_a_path", type=str, default='./dataset/CelebA_HQ_224/test_files/img1/26000.jpg', help="Person who provides identity information")
        self.parser.add_argument("--pic_b_path", type=str, default='./dataset/CelebA_HQ_224/test_files/img2/28006.jpg', help="Person who provides information other than their identity")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="results path.")
        self.parser.add_argument('--crop_size', type=int, default=224, help='Crop of size of input image')
        self.parser.add_argument('--Gdeep', type=str2bool, default='False')
        
        self.isTrain = False