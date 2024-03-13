from executors import PyExecutor
from generators import PyGenerator, model_factory
from typing import List
from multiprocessing import Pool
from filelock import FileLock
import random
from transformers import GPT2Tokenizer
from utils import *
import sys
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def debug(i, item, log_path, model_name, num_items, pass_at_k, max_iters, port="", level = "block"):
    exe = PyExecutor()
    gen = PyGenerator()
    model = model_factory(model_name, port)
    cur_pass = 0
    is_solved = False
    implementations = []
    test_feedback = []
    cur_func_impl = ""
    dataset_type = item["task_id"].split("/")[0]
    token_nums = 0
    while cur_pass < pass_at_k and not is_solved:
        cur_iter = 0
        tests_i = item['given_tests']
        # clean test_i
        tests_i = [test for test in tests_i if item['entry_point'] in test and 'assert False' not in test]
        # first attempt
        cur_func_impl = prepare_function_from_seed(dataset_type, item["prompt"], item["seed"], item["entry_point"])
        implementations.append(cur_func_impl)
        # call the executor to return failed_test
        is_passing, failed_tests, _ = exe.execute(cur_func_impl, tests_i)
        test_feedback.append(failed_tests)
        # if solved, exit early
        if is_passing:
            is_solved = exe.evaluate(item["entry_point"], cur_func_impl, item["test"], timeout=10)
            break
        # use debug to iteratively improve
        last_func_impl = ""
        if model.is_chat:
            messages = []
        else:
            messages = ""
        while cur_iter < max_iters:
            # get self-reflection by debugging a random failed tests
            # The output is 
            # 1. the wrong blocks [wrong block]
            # 2. the explanation [explanation]
            if dataset_type in ["HumanEval", "MBPP"]:
                # Add comments
                if not find_comment(cur_func_impl, item["entry_point"]):
                    debug_cur_func_impl = insert_comment(cur_func_impl, extrace_comment(item["prompt"]), item["entry_point"])
                else:
                    debug_cur_func_impl = cur_func_impl
            elif dataset_type in ["TransCoder"]:
                # Add C++ translation as comments
                debug_cur_func_impl = convert_comment(item["prompt"]) + cur_func_impl
            selected_test = failed_tests[random.randint(0,len(failed_tests)-1)] if len(failed_tests) >= 1 else None
            generate_function = None
            messages = gen.ldb_debug(item["prompt"], debug_cur_func_impl, selected_test, item["entry_point"], model, messages, dataset_type, level)
            cur_func_impl, cur_messages = gen.ldb_generate(
                func_sig=item["prompt"],
                model=model,
                prev_func_impl=cur_func_impl,
                messages=messages,
                failed_tests=selected_test,
                dataset_type=dataset_type)
            
            messages = cur_messages
            if isinstance(messages, str):
                token_nums += len(tokenizer.tokenize(messages))
            else:
                token_nums += sum([len(tokenizer.tokenize(msg.content)) for msg in messages])
            cur_func_impl = prepare_function_from_seed(dataset_type, item["prompt"], cur_func_impl, item["entry_point"])
            last_func_impl = cur_func_impl
            implementations.append(cur_func_impl)
            # check if all internal unit tests pass
            is_passing, failed_tests, _ = exe.execute(
                cur_func_impl, tests_i)
            test_feedback.append(failed_tests)
            # if passed, check if it passes the real tests, exit early
            if is_passing or cur_iter == max_iters - 1:
                if is_passing:
                    print(f'{item["task_id"]} pass generated tests, check real tests')
                else:
                    print(f'{item["task_id"]} fail generated tests, check real tests')
                is_solved = exe.evaluate(
                    item["entry_point"], cur_func_impl, item["test"], timeout=10)
                if is_solved:
                    item["solution"] = cur_func_impl
                cur_iter += 1
                sys.stdout.flush()
                break
            cur_iter += 1
            sys.stdout.flush()
        cur_pass += 1
    item["is_passing"] = is_passing
    item["is_solved"] = is_solved
    item["implementations"] = implementations
    item["test_feedback"] = test_feedback
    item["solution"] = cur_func_impl
    item["generated_test"] = tests_i
    item["debug_iter"] = cur_iter
    item["token_nums"] = token_nums
    with FileLock(log_path + ".lock"):
        write_jsonl(log_path, [item], append=True)
    print(f'completed {i+1}/{num_items}')

def run_ldb(
    dataset: List[dict],
    model_name: str,
    max_iters: int,
    n_proc: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    seedfile: str = None,
    testfile: str = None,
    port: str = "",
    level: str = "block"
) -> None:
    print("Number of proc:", n_proc)
    num_items = len(dataset)
    args = iter([(i, item, log_path, model_name, num_items, pass_at_k, max_iters, port, level) for i, item in enumerate_resume(dataset, log_path, seedfile, testfile)])
    if n_proc == 1:
        for item in args:
            debug(*item)
    else:
        with Pool(n_proc) as pool:
            pool.starmap(debug, args)
    print("Accuracy:", count_solved(log_path))
    