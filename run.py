import subprocess


def list2str(l: list):
    return " ".join([str(i) for i in l])


loss_fns = (
    "ce",
    "nce",
    "sce",
    "rce",
    "nrce",
    "gce",
    "ngce",
    "mae",
    "nmae",
    "nce rce",
    "nce mae",
    "gce nce",
    "gce rce",
    "gce mae",
    "ngce nce",
    "ngce rce",
    "ngce mae",
    "mae rce",
)

if __name__ == '__main__':
    # ============================
    # No noise
    # mean -> 0
    # std  -> 0
    # loss -> ce
    # ============================
    user_mean = list2str([0])
    user_std = list2str([0])
    cmdline = f"python main_joint.py --user_mean {user_mean} --user_std {user_std} --loss ce"
    subprocess.run(cmdline, shell=True, check=True)

    # ============================
    # mean -> 0.2 * 10
    # std  -> 0.2 * 10
    # ============================
    user_mean = list2str([0.2] * 10)
    user_std = list2str([0.2] * 10)
    for loss_fn in loss_fns:
        cmdline = f"python main_joint.py --user_mean {user_mean} --user_std {user_std} --loss {loss_fn}"
        subprocess.run(cmdline, shell=True, check=True)

    # ============================
    # mean -> 0.4 * 10
    # std  -> 0.2 * 10
    # ============================
    user_mean = list2str([0.4] * 10)
    user_std = list2str([0.2] * 10)
    for loss_fn in loss_fns:
        cmdline = f"python main_joint.py --user_mean {user_mean} --user_std {user_std} --loss {loss_fn}"
        subprocess.run(cmdline, shell=True, check=True)

    # ============================
    # mean -> 0.4 * 10
    # std  -> 0.4 * 10
    # ============================
    user_mean = list2str([0.4] * 10)
    user_std = list2str([0.4] * 10)
    for loss_fn in loss_fns:
        cmdline = f"python main_joint.py --user_mean {user_mean} --user_std {user_std} --loss {loss_fn}"
        subprocess.run(cmdline, shell=True, check=True)

    # ============================
    # mean -> 0.6 * 10
    # std  -> 0.2 * 10
    # ============================
    user_mean = list2str([0.6] * 10)
    user_std = list2str([0.2] * 10)
    for loss_fn in loss_fns:
        cmdline = f"python main_joint.py --user_mean {user_mean} --user_std {user_std} --loss {loss_fn}"
        subprocess.run(cmdline, shell=True, check=True)
