from .base_options0 import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int,
                            default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str,
                            default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float,
                            default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test',
                            help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true',
                            help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50,
                            help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        parser.add_argument("--case", type=int,
                            help="The number of nodes to use for distributed "
                            "training")
        parser.add_argument('--vis', default=False, action='store_true',
                            help='whether visualzie result')
        parser.add_argument('--gpu', type=str,
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        parser.add_argument('--model_iter', type=str)
        parser.add_argument('--video', type=str,
                            help='eval one special video')
        parser.add_argument('--tracker_name', type=str)
        parser.add_argument('--eps', type=int)
        parser.add_argument('--istargeted', default=False, action='store_true',
                            help='whether visualzie result')

        parser.add_argument('--trajcase', type=str)
        parser.add_argument('--targetcase', type=str)
        parser.add_argument('--offsetx', type=int)
        parser.add_argument('--offsety', type=int)

        parser.add_argument('--attack_universal', default=False, action='store_true',
                            help='whether visualzie result')

        # FOR DIMP ----------------------------------------------------------------
        parser.add_argument('--exp_module', type=str,
                            help='Name of experiment module in the experiments/ folder.')
        parser.add_argument('--exp_name', type=str,
                            help='Name of the experiment function.')

        # parser.add_argument('experiment_module', type=str,
        #                     help='Name of experiment module in the experiments/ folder.')
        # parser.add_argument('experiment_name', type=str,
        #                     help='Name of the experiment function.')
        parser.add_argument('--debug', type=int, default=0, help='Debug level.')
        parser.add_argument('--threads', type=int, default=0,
                            help='Number of threads.')

        parser.add_argument('--config', default='', type=str, help='config file')
        parser.add_argument('--snapshot', default='', type=str, help='snapshot of models to eval')

        # ----------------------------------------------------------------

        parser.add_argument('--directions', type=int, default=12)
        parser.add_argument('--driftdistance', type=int, default=5)

        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
