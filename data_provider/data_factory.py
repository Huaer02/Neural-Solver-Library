from data_provider.data_loader import airfoil, ns, darcy, pipe, elas, plas, pdebench_autoregressive, \
    pdebench_steady_darcy, car_design, cfd3d, pdebench_npy, pdebench_unified


def get_data(args):
    data_dict = {
        'car_design': car_design,
        'pdebench_autoregressive': pdebench_autoregressive,
        'pdebench_steady_darcy': pdebench_steady_darcy,
        'elas': elas,
        'pipe': pipe,
        'airfoil': airfoil,
        'darcy': darcy,
        'ns': ns,
        'plas': plas,
        'cfd3d': cfd3d,
        'pdebench_npy': pdebench_npy,
        'pdebench_unified': pdebench_unified,
    }
    dataset = data_dict[args.loader](args)
    if args.loader == 'pdebench_unified':
        train_loader, val_loader, test_loader, shapelist = dataset.get_loader()
    else:
        train_loader, test_loader, shapelist = dataset.get_loader()

    if args.loader == 'pdebench_unified':
        return dataset, train_loader, val_loader, test_loader, shapelist
    return dataset, train_loader, test_loader, shapelist
