import os
import argparse
from ldb import run_ldb
from simple import run_simple
from repeat_simple import run_repeat_simple
from utils import read_jsonl, read_jsonl_gz

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, help="The name of the run")
    parser.add_argument("--root_dir", type=str,
                        help="The root logging directory", default="root")
    parser.add_argument("--dataset_path", type=str,
                        help="The path to the benchmark dataset", default="root")
    parser.add_argument("--strategy", type=str,
                        help="Strategy: `simple`, `ldb`")
    parser.add_argument(
        "--model", type=str, help="OpenAI models only for now. For best results, use GPT-4")
    parser.add_argument("--pass_at_k", type=int,
                        help="Pass@k metric", default=1)
    parser.add_argument("--max_iters", type=int,
                        help="The maximum number of self-improvement iterations", default=10)
    parser.add_argument("--n_proc", type=int,
                        help="The number of processes", default=1)
    parser.add_argument("--seedfile", type=str, help="seed file of the solutions", default="")
    parser.add_argument("--testfile", type=str, help="tests for debugging", default="")
    parser.add_argument("--port", type=str, help="tests for debugging", default="")
    parser.add_argument("--level", type=str, help="granularity for debugging", default="block")
    parser.add_argument("--verbose", action='store_true',
                        help="To print live logs")
    args = parser.parse_args()
    return args


def strategy_factory(strategy: str):
    def kwargs_wrapper_gen(func, delete_keys=[], add_keys={}):
        def kwargs_wrapper(**kwargs):
            for key in delete_keys:
                del kwargs[key]
            for key in add_keys:
                kwargs[key] = add_keys[key]
            return func(**kwargs)
        return kwargs_wrapper
    
    if strategy == "simple":
        return kwargs_wrapper_gen(run_simple, delete_keys=["max_iters", "seedfile", "port", "level"])
    if strategy == "repeat_simple":
        return kwargs_wrapper_gen(run_repeat_simple, delete_keys=["pass_at_k", "seedfile", "n_proc", "port", "level"])
    elif strategy == "ldb":
        return kwargs_wrapper_gen(run_ldb)
    else:
        raise ValueError(f"Strategy `{strategy}` is not supported")


def main(args):
    # check if the root dir exists and create it if not
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)

    # get the dataset name
    dataset_name = os.path.basename(args.dataset_path).replace("jsonl", "")

    # check if log path already exists
    log_dir = os.path.join(args.root_dir, args.run_name)
    seed_name = os.path.basename(args.seedfile).split('/')[-1].replace("jsonl", "")
    log_path = os.path.join(
        log_dir, f"{dataset_name}_{args.strategy}_{args.max_iters}_{args.model}_pass_at_{args.pass_at_k}_seed_{seed_name}.jsonl")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # check if the strategy is valid
    run_strategy = strategy_factory(args.strategy)

    # print starting message
    if args.verbose:
        print(f"""
Starting run with the following parameters:
strategy: {args.strategy}
pass@k: {args.pass_at_k}
""")
    else:
        print(f"Logs will be saved in `{log_dir}`")

    # load the dataset
    print(f'Loading the dataset...')
    if args.dataset_path.endswith(".jsonl"):
        dataset = read_jsonl(args.dataset_path)
    elif args.dataset_path.endswith(".jsonl.gz"):
        dataset = read_jsonl_gz(args.dataset_path)
    else:
        raise ValueError(
            f"Dataset path `{args.dataset_path}` is not supported")

    print(f"Loaded {len(dataset)} examples")
    # start the run
    # evaluate with pass@k
    run_strategy(
        dataset=dataset,
        model_name=args.model,
        max_iters=args.max_iters,
        n_proc=args.n_proc,
        pass_at_k=args.pass_at_k,
        log_path=log_path,
        verbose=args.verbose,
        seedfile=args.seedfile,
        testfile=args.testfile,
        port=args.port,
        level=args.level
    )

    print(f"Done! Check out the logs in `{log_path}`")


if __name__ == "__main__":
    args = get_args()
    main(args)
