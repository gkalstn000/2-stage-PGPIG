
def dptn_options(parser):
    parser.add_argument('--norm', type=str, default='instance',
                        help='[spectralinstance|spectralbatch|spectralsyncbatch]')

    # for generator
    parser.add_argument('--netG', type=str, default='dptn', help='selects model to use for netG (dptn | dualattn)')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--init_type', type=str, default='xavier',
                        help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
    parser.add_argument('--use_spect_g', action='store_false', help='use spectual normalization in generator')
    parser.add_argument('--use_coord', action='store_true', help='use coordconv')
    parser.add_argument('--layers_g', type=int, default=3, help='number of layers in G')
    parser.add_argument('--num_blocks', type=int, default=3, help="number of resblocks")
    parser.add_argument('--affine', action='store_true', default=True, help="affine in PTM")
    parser.add_argument('--nhead', type=int, default=2, help="number of heads in PTM")
    parser.add_argument('--num_CABs', type=int, default=2, help="number of CABs in PTM")
    parser.add_argument('--num_TTBs', type=int, default=2, help="number of CABs in PTM")
    parser.add_argument('--pos_encoding', action='store_true', help="pos_encoding")
    parser.add_argument('--step_size', type=int, default=5, help="gen step size")
    parser.add_argument('--img_f', type=int, default=512, help="the largest feature channels")
    # for optimizer
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')
    # the default values for beta1 and beta2 differ by TTUR option
    opt, _ = parser.parse_known_args()
    if opt.no_TTUR:
        parser.set_defaults(beta1=0.5, beta2=0.999)
    parser.add_argument('--D_steps_per_G', type=int, default=1,
                        help='number of discriminator iterations per generator iterations.')
    parser.add_argument('--gan_mode', type=str, default='lsgan', choices=['wgan-gp', 'hinge', 'lsgan'])
    # for discriminators
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--netD', type=str, default='res', help='(n_layers|multiscale|image)')
    parser.add_argument('--dis_layers', type=int, default=4, help='number of layers in D')
    parser.add_argument('--use_spect_d', action='store_false', help='use spectual normalization in generator')
    # for loss weights
    parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
    parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
    parser.add_argument('--lambda_rec', type=float, default=2.5, help='weight for image reconstruction loss')
    parser.add_argument('--lambda_style', type=float, default=250, help='weight for the VGG19 style loss')
    parser.add_argument('--lambda_content', type=float, default=0.25, help='weight for the VGG19 content loss')
    parser.add_argument('--lambda_g', type=float, default=2.0, help='weight for generation loss')
    parser.add_argument('--t_s_ratio', type=float, default=0.5, help='loss ratio between dual tasks')

    return parser