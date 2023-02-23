from detectors.methods.cpd.pelt import PeltL2

cpd_registry = {
    "peltl2": PeltL2,
}


def create_cpd(cpd_name, **kwargs):
    if cpd_name not in cpd_registry:
        raise ValueError(f"CPD {cpd_name} not found")
    return cpd_registry[cpd_name](**kwargs)
