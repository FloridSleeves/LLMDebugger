from utils import enumerate_resume, make_printv, write_jsonl, IMPORT_HEADER, count_solved
from executors import executor_factory
from generators import model_factory
from generators import PyGenerator
from typing import List
from filelock import FileLock
from multiprocessing import Process, Pool

def get_seed(i, item, model, num_items, pass_at_k, gen, log_path):
    print(f'[Start] {i+1}')
    exe = executor_factory("python", False)
    cur_pass = 0
    is_solved = False
    cur_func_impl = ""
    num_success = 0
    dataset_type = item["task_id"].split("/")[0]
    token_num = 0
    while cur_pass < pass_at_k:
        cur_func_impl, messages = gen.func_impl(item["prompt"], model, "simple", given_tests=item["given_tests"], dataset_type=dataset_type)
        assert isinstance(cur_func_impl, str)
        if cur_pass > 0:
            # We count the token number only when the first pass is failed to match debugging
            token_num += sum([len(msg.content) for msg in messages])
        cur_func_impl = item["prompt"] + "\n" + cur_func_impl
        is_solved = exe.evaluate(item["entry_point"], cur_func_impl, item["test"], timeout = 20)
        if is_solved:
            num_success += 1
            break
        cur_pass += 1
    item["solution"] = cur_func_impl
    item["is_solved"] = is_solved
    item['token_num'] = token_num
    item['debug_iter'] = cur_pass
    #with FileLock(log_path + ".lock"):
    write_jsonl(log_path, [item], append=True)
    print(f'Completed {i+1}/{num_items}')
    return 

def async_main(
        dataset: List[dict],
        model_name: str,
        pass_at_k: int,
        n_proc: int,
        log_path: str,
        verbose: bool,
        port = "",
        testfile: str = None,
    ) -> None:
    gen = PyGenerator()
    model = model_factory(model_name, port)
    print_v = make_printv(verbose)
    num_items = len(dataset)
    num_success = 0
    if n_proc == 1:
        for i, item in enumerate_resume(dataset, log_path, testfile=testfile):
            get_seed(i, item, model, num_items, pass_at_k, gen, log_path)
        return
    # divide dataset into several groups
    with Pool(n_proc) as pool:
        args = iter([(i, item, model, num_items, pass_at_k, gen, log_path) for i, item in enumerate_resume(dataset, log_path, testfile=testfile)])
        pool.starmap(get_seed, args)

def run_simple(
        dataset: List[dict],
        model_name: str,
        pass_at_k: int,
        n_proc: int,
        log_path: str,
        verbose: bool,
        port: str = "",
        testfile: str = None,
    ) -> None:
    async_main(dataset, model_name, pass_at_k, n_proc, log_path, verbose, port, testfile)
    print("Accuracy:", count_solved(log_path))
