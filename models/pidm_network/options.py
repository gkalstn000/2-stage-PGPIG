
def pidm_options(parser):
    parser.add_argument('--cond_scale', type=int, default=2)
    parser.add_argument('--guidance_prob', type=int, default=0.1)
    parser.add_argument('--sample_algorithm', type=str, default='ddim')  # ddpm, ddim
    # for optimizer
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--adam_beta1', type=float, default=0.0)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    # for unet model

    # for diffusion
    parser.add_argument('--schedule', type=str, default='linear')
    parser.add_argument('--n_timestep', type=int, default=1000)
    parser.add_argument('--linear_start', type=float, default=0.0001)
    parser.add_argument('--linear_end', type=float, default=0.02)




    return parser

