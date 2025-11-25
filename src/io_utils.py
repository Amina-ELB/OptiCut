import os
import shutil

def init_output_folders(rank):
    if rank != 0:
        return

    shutil.rmtree("res", ignore_errors=True)
    os.mkdir("res")

    names = [
        "cost_func", "cost_compliance", "lagrangian", "constraint",
        "max_vm", "volume", "param_lagrangian",
        "vm_1_hist", "vm_final_hist"
    ]

    for n in names:
        open(f"res/{n}.txt", "x")
