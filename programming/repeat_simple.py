from utils import enumerate_resume, make_printv, write_jsonl, IMPORT_HEADER, count_solved
from executors import executor_factory
from generators import model_factory
from generators import PyGenerator
from typing import List
from filelock import FileLock
from multiprocessing import Process, Pool

def get_seed(i, item, model, is_leetcode, num_items, max_iters, gen, log_path):
    print(f'[Start] {i+1}')
    exe = executor_factory("python", False)
    cur_pass = 0
    is_solved = False
    cur_func_impl = ""
    num_success = 0
    dataset_type = item["task_id"].split("/")[0]
    
    cur_iter = 0
    while cur_iter < max_iters:
        cur_func_impl, _ = gen.func_impl(item["prompt"], model, "simple", given_tests=item["given_tests"], dataset_type=dataset_type)
        assert isinstance(cur_func_impl, str)
        cur_func_impl = item["prompt"] + "\n" + cur_func_impl
        is_solved = exe.evaluate(item["entry_point"], cur_func_impl, item["test"], timeout = 20 if is_leetcode else 10)
        if is_solved:
            num_success += 1
            cur_iter += 1
            break
        cur_iter += 1
    item["solution"] = cur_func_impl
    item["is_solved"] = is_solved
    item["debug_iter"] = cur_iter
    #with FileLock(log_path + ".lock"):
    write_jsonl(log_path, [item], append=True)
    print(f'Completed {i+1}/{num_items}')
    return 

def async_main(
        dataset: List[dict],
        model_name: str,
        language: str,
        max_iters: int,
        log_path: str,
        verbose: bool,
        is_leetcode: bool = False,
        port: str = "",
        testfile: str = None,
    ) -> None:
    
    gen = PyGenerator()
    model = model_factory(model_name, port)

    print_v = make_printv(verbose)
    
    num_items = len(dataset)
    num_success = 0
    # divide dataset into several groups
    n_proc = 10
    with Pool(n_proc) as pool:
        args = iter([(i, item, model, is_leetcode, num_items, max_iters, gen, log_path) for i, item in enumerate_resume(dataset, log_path, testfile=testfile)])
        pool.starmap(get_seed, args)

def run_repeat_simple(
        dataset: List[dict],
        model_name: str,
        language: str,
        max_iters: int,
        log_path: str,
        verbose: bool,
        is_leetcode: bool = False,
        port: str = "",
        testfile: str = None,
    ) -> None:
    async_main(dataset, model_name, language, max_iters, log_path, verbose, is_leetcode, port, testfile)
    print("Accuracy:", count_solved(log_path))
