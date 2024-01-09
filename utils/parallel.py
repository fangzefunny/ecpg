import multiprocessing as mp

# --------- Get pools  --------- #  

def get_pool(args):
    if hasattr(args, 'n_fit'): n=args.n_fit 
    elif hasattr(args, 'n_sim'): n=args.n_sim
    else: n=20
    n_cores = args.n_cores if args.n_cores else int(mp.cpu_count()*.7) 
    print(f'    Using {n_cores} parallel CPU cores\n ')
    return mp.Pool(n_cores)