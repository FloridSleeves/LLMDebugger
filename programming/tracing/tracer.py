# Split into blocks
import json
import os.path
import sys, os
from typing import Any, Dict, Optional, Set
import astroid
from astroid import nodes
from astroid.builder import AstroidBuilder
import time
sys.path.append(os.path.join(sys.path[0], '..'))
from cfg.graph import CFGBlock, ControlFlowGraph
from cfg.visitor import CFGVisitor
import ast
import re
import random

class VariableVisitor(object):
    def __init__(self):
        self.variables = set()

    def __call__(self, node):
        return node.accept(self)
    
    def visit_name(self, node):
        self.variables.add(node.name)
    
    def visit_slice(self, node):
        if node.lower is not None:
            node.lower.accept(self)
        if node.upper is not None:
            node.upper.accept(self)
        if node.step is not None:
            node.step.accept(self)  
    
    def visit_const(self, node):
        return

    def visit_subscript(self, node):
        node.value.accept(self)
        #node.slice.accept(self)

    def visit_return(self, node):
        self.variables.add(node.value.as_string())

    def visit_call(self, node):
        for arg in node.args:
            arg.accept(self)
        for kwarg in node.keywords:
            kwarg.accept(self)

    def visit_generatorexp(self, node):
        for gen in node.generators:
            gen.accept(self)

    def visit_comprehension(self, node):
        node.iter.accept(self)
        for gen in node.ifs:
            gen.accept(self)

    def visit_unaryop(self, node):
        node.operand.accept(self)

    def visit_binop(self, node):
        node.left.accept(self)
        node.right.accept(self)

    def visit_boolop(self, node):
        for val in node.values:
            val.accept(self)
    
    def visit_compare(self, node):
        node.left.accept(self)
        for op, val in node.ops:
            val.accept(self)

    def visit_list(self, node):
        for val in node.elts:
            val.accept(self)

    def visit_joinedstr(self, node):
        for val in node.values:
            val.accept(self)

    def visit_formattedvalue(self, node):
        node.value.accept(self)
    
    def visit_dict(self, node):
        for key, val in node.items:
            key.accept(self)
            val.accept(self)
    
    def __getattr__(self, attr: str):
        if attr.startswith("visit_"):
            return self.visit_generic
        else:
            raise AttributeError(f"'CFGVisitor' object has not attribute '{attr}'")

    def visit_generic(self, node):
        return

def get_use_def(block):
    uses = set()
    defs = set()
    visitor = VariableVisitor()
    for s in block.statements:
        if isinstance(s, nodes.Assign):
            # traverse all ast node in s.value
            visitor(s.value)
            uses = uses.union(visitor.variables)
            for t in s.targets:
                defs.add(t.as_string())
        elif isinstance(s, nodes.AugAssign):
            visitor(s.value)
            uses.add(s.target.as_string())
            uses.union(visitor.variables)
            defs.add(s.target.as_string())
    return uses, defs


def decorate_block(block, block_type):
    original_lineno = block.statements[-1].lineno
    original_col_offset = block.statements[-1].col_offset
    original_end_lineno = block.statements[-1].end_lineno
    original_end_col_offset = block.statements[-1].end_col_offset
    if block_type == "if":
        tmp = block.statements[-1].as_string()
        try:
            block.statements[-1] = astroid.extract_node("_cond = (" + tmp + ")" if "_cond" not in tmp else tmp)
        except Exception as e:
            block.statements[-1] = astroid.extract_node(tmp)
    elif block_type == "return":
        block.statements[-1] = astroid.extract_node(block.statements[-1].as_string().replace("return ", "_ret = "))
    block.statements[-1].lineno = original_lineno
    block.statements[-1].col_offset = original_col_offset
    block.statements[-1].end_lineno = original_end_lineno
    block.statements[-1].end_col_offset = original_end_col_offset

def visit_block(block):
    if len(block.statements) == 0 or isinstance(block.statements[0], nodes.Arguments):
        return None
    if len(block.successors) > 1:
        decorate_block(block, "if")
    if isinstance(block.statements[-1], nodes.Return):
        decorate_block(block, "return")
    block_prog = "\n".join([s.as_string() for s in block.statements])
    uses, defs = get_use_def(block)
    return uses, block_prog, defs
    #assemble_block_prompt(block_prog, uses, defs)

"""
divide(prog): 
    Return (None, "error message") or (divided_blocks, "")
"""
def divide(prog):
    try:
        module = AstroidBuilder().string_build(prog)
        visitor = CFGVisitor(options={"separate-condition-blocks": True})
        module.accept(visitor)
    except Exception as e:
        return None, str(e)
    cfgs = visitor.cfgs
    divided_block = []
    for node, cfg in cfgs.items():
        if isinstance(node, nodes.Module):
            continue    
        first_block = cfg.start
        block_queue = []
        visited_block = set()
        block_queue.append(first_block)
        # Visit all blocks, get uses, progs, and defs
        block_id = 0
        while len(block_queue) > 0:
            cur_block = block_queue.pop(0)
            orig_block_prog = "\n".join([s.as_string() for s in cur_block.statements])
            res = visit_block(cur_block)
            if res is not None:
                uses, block_prog, defs = res
                if '_cond' in defs:
                    defs.remove('_cond')
                    defs.add(orig_block_prog)
                elif '_ret' in defs:
                    defs.remove('_ret')
                    defs.add(cur_block.statements[-1].as_string().replace("_ret = ", ""))
                exist = False
                for exist_block in divided_block:
                    if exist_block[3] == block_prog:
                        exist = True
                if not exist:
                    divided_block.append([cur_block, uses, defs, block_prog, block_id])
                block_id += 1
            # traverse following blocks
            visited_block.add(cur_block)
            for succ in cur_block.successors:
                if succ.target not in visited_block:
                    block_queue.append(succ.target)
    return divided_block, ""

def get_trace_line(trace, funcname, fname):
    mark = f"--- modulename: .tmp.py, funcname: {funcname}" + "\n"
    lines = trace.split(mark)[1].split("\n")
    traces = []
    for l in lines:
        # trace also record comment lines for some reason
        if l.lstrip().startswith("\'\'\'") or l.lstrip().startswith("\"\"\"") or l.lstrip().startswith("#"):
            continue
        traces.append(l)
    return traces

def get_error_msg(error):
    error_lines = error.split('\n')
    error_msg = ""
    last_l = ""
    code = ""
    for l in error_lines:
        if "File \"" in last_l:
            code = l
        elif "Error: " in l:
            error_msg = ("This line is wrong: ```" + code + "```\n" + l) if "__var_list" not in code else l
            break
        last_l = l
    return error_msg

# Get trace
# Return: "*timeout*" or "*execution fail*{error_msg}" or "*parse fail*{ferr}" or line_traces(List)
def get_trace(prog, funcname):
    fname = '.tmp.py.' + str(random.randint(0, 10000))
    f = open(fname, "w")
    f.write(prog)
    f.close()
    # run in command line python -m trace -t tmp.py > trace
    import subprocess
    try:
        res=subprocess.run(["python3", "-m", "trace", "-t", fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
    except AssertionError:
        # This is expected if fail the test assetion
        pass
    except subprocess.TimeoutExpired:
        return "*timeout*"
    except Exception as e:
        error_msg = get_error_msg(res.stderr.decode('utf-8'))
        print("Trace Execution Fail:" + error_msg)
        return "*execution fail*" + error_msg
    finally:
        os.remove(fname)
    trace = res.stdout.decode('utf-8')
    # Find --- modulename: tmp, funcname: {funcname}
    try:
        trace = get_trace_line(trace, funcname, fname)
    except IndexError:
        ferr_name = "../error/.error.py" + str(time.time())
        ferr = open(ferr_name, 'w')
        ferr.write(prog)
        ferr.close()
        return f"*parse fail*{ferr_name}" 
    # Find all lines with .tmp.py
    line_trace = []
    for l in trace:
        if l.startswith(fname):
            import re
            m = re.search(f"^{fname}", l)
            if (not line_trace) or (line_trace[-1] not in l):
                line_trace.append(l[m.end():])
    return line_trace

def get_executed_block(trace_lines, divided_blocks):
    lineno_shift = 0
    executed_blocks = []
    for l in trace_lines:
        lineno = int(l.split('):')[0].split('(')[1])
        code_line = l.split('):')[1]
        lineno_in_func = lineno-lineno_shift
        for b in divided_blocks:
            if b[0].statements[0].lineno == lineno_in_func:
                end_lineno = b[0].statements[-1].end_lineno
                instrument_info = b + [lineno_in_func, end_lineno, get_indent(l)]
                if instrument_info not in executed_blocks:
                    executed_blocks.append(instrument_info)
            # if l.startswith(b[3]):
            #     executed_blocks.append(b)
    return executed_blocks #[block, uses, defs, block_prog, block_id, lineno, indent]

def instrument_simple(prog, instrument_input):
    stmts = prog.split("\n")
    instrumented_blocks = set()
    for instrument_payload in instrument_input:
        block, uses, defs, code, b_idx, start_lineno, end_lineno, indent = instrument_payload
        indent_before = indent
        indent_after = indent
        if b_idx in instrumented_blocks:
            continue
        instrumented_blocks.add(b_idx)
        if len(block.successors) > 1:
            continue
        else:
            if (stmts[end_lineno-1].lstrip().startswith("for ")) or (stmts[end_lineno-1].lstrip().startswith("if ")) or (stmts[end_lineno-1].lstrip().startswith("while ")):
                indent_after += 1
            payload_defs = "    "*indent_after + f"__var_list = vars();print(f'Value_After:{end_lineno}|' + '|'.join([(x + '=' + _str(__var_list[x])) for x in __var_list if not x.startswith('__')]));"
            stmts.insert(end_lineno, payload_defs)
            if 'return ' in stmts[end_lineno-1]:
                stmts[end_lineno-1] = stmts[end_lineno-1].replace('return ', '_ret = ')
        payload_uses = "    "*indent_before + f"__var_list = vars();print(f'Value_Before:{start_lineno}|' + '|'.join([(x + '=' + _str(__var_list[x])) for x in __var_list if not x.startswith('__')]));"
        stmts.insert(start_lineno-1, payload_uses)
        
    return "\n".join(stmts)

def collect_runtime_value_simple(value_prof_prog):
    hook = ""
    import sys
    hooked_prog = hook + "\n" + value_prof_prog
    fname = "tmp_line.py" + f".{random.randint(0,10000)}"
    with open(fname, "w") as f:
        f.write(hooked_prog)
    import subprocess
    try:
        res=subprocess.run(["python3", fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
    except subprocess.TimeoutExpired:
        return "*timeout*"
    finally:
        os.remove(fname)
    output = res.stderr.decode('utf-8')
    if "Traceback (most recent call last):" in output and ("AssertionError" not in output):
        output = get_error_msg(output)
        print("Execution Fail:\n" + output)
        return "*execution fail*" + output
    output = res.stdout.decode('utf-8')
    return output

def get_lineno(trace_line):
    match = re.search("\([0-9]+\):", trace_line)
    return int(trace_line[match.start()+1:match.end()-2])

def get_line(trace_line):
    match = re.search("\([0-9]+\):", trace_line)
    return trace_line[match.end()+1:]

def get_indent(trace_line):
    match = re.search("\([0-9]+\):", trace_line)
    len1 = len(trace_line[match.end()+1:])
    len2 = len(trace_line[match.end()+1:].lstrip())
    return int((len1-len2)/4)

def extract_value(output):
    output = output.split("\n")[:-1]
    output = [x for x in output if x.startswith('Value_')]
    return output

def parse_runtime_value_simple(output, trace_lines):
    trace_idx = 0
    blocks = []
    blk = []
    value_profiles = extract_value(output)
    trace_len = len(trace_lines)
    trace_linenos = [get_lineno(l) for l in trace_lines]
    for i, l in enumerate(value_profiles):
        pos = 0 if l[6:].split(':')[0] == 'Before' else 1
        lineno = int(l.split(':')[1].split('|')[0])
        if lineno not in trace_linenos:
            continue
        values = '\t'.join(l.split('|')[1:])
        values = values if len(values) < 100 else (values[:50] + "..." + values[-50:])
        while trace_idx < trace_len and get_lineno(trace_lines[trace_idx]) != lineno:
            trace_l = trace_lines[trace_idx]
            blk.append(get_line(trace_l))
            trace_idx += 1
        if trace_idx == trace_len:
            break
        if pos == 0:
            blk.append("    "*get_indent(trace_lines[trace_idx]) + "# " + values)
            if i+1 == len(value_profiles):
                break
            next_pos = 0 if value_profiles[i+1][6:].split(':')[0] == 'Before' else 1
            if next_pos == 0:
                # Block has multiple successors
                blk.append(get_line(trace_lines[trace_idx]))
                trace_idx += 1
        else:
            blk.append(get_line(trace_lines[trace_idx]))
            values_masked = ""
            matches = re.findall(r'(\w+)=([^|]+)', values)
            for m in matches:
                values_masked = values
                #values_masked += m[0] + "=?|"
            blk.append("    "*get_indent(trace_lines[trace_idx]) + "# " + values_masked)
            blocks.append(blk)
            blk = []
            trace_idx += 1
    return blocks

def get_code_traces_block(prog, test, entry):
    log_of_tracing = ""
    # Divide program into basic block units
    divided_blocks, error = divide(prog)
    if divided_blocks is None:
        return "*execution fail*" + error
    # Collect Execution Traces
    exec_prog = prog + "\n" + test
    trace_lines = get_trace(exec_prog, entry)
    if isinstance(trace_lines, str):
        if trace_lines == "*timeout*" or trace_lines.startswith("*execution fail*") or trace_lines.startswith("*parse fail*"):
            print("Collect Traces Fail: " + trace_lines)
            return trace_lines
    log_of_tracing += str("Trace:\n"+ '\n'.join(trace_lines[:10]))
    instrument_input = sorted(list(get_executed_block(trace_lines, divided_blocks)), key=lambda x: x[5], reverse=True)
    value_prof_prog = instrument_simple(prog, instrument_input)
    log_of_tracing += str("Value Profile Program:\n" + value_prof_prog + "\n" + test + "\n")
    output = collect_runtime_value_simple(value_prof_prog + "\n" + test)
    if output == "*timeout*" or output.startswith("*execution fail*"):
        return output
    log_of_tracing += "\n" + str("Value Profile Output:\n" + output)
    runtime_value = parse_runtime_value_simple(output, trace_lines)
    return runtime_value

def get_range(prog, entry):
    tree = AstroidBuilder().string_build(prog)
    for ele in tree.body:
        if isinstance(ele, nodes.FunctionDef) and ele.name == entry:
            return [ele.lineno-1, ele.end_lineno-1] # Lineno start from 0
    return None

def get_after(stmts):
    for s in stmts:
        if s == "":
            continue
        else:
            return s.strip(), int((len(s) - len(s.lstrip()))/4)