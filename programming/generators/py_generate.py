from .model import ModelBase, message_to_str
from .model import ModelBase, Message, StarCoder
from tracing import get_code_traces_block, get_code_traces_line, get_code_traces_function
from typing import Optional, List, Union
import ast
import re, random, time, json
from .parse import parse_code_block, add_code_block
from .prompt import *
from utils import IMPORT_HEADER

def trim_header(func_impl):
    if IMPORT_HEADER in func_impl:
        func_impl = func_impl.replace(IMPORT_HEADER, "")
    return func_impl

def parse_explanation(responses, trace_blocks, prev_func_impl):
    lines = responses.strip().strip('.').split("\n")
    wrong_block = None
    explanation = None
    for l in lines:
        try:
            item = json.loads(l)
            assert "block" in item
            assert "correct" in item
            assert "explanation" in item
        except Exception as e:
            continue
        # convert parsed_reponse['correct'] into bool
        if isinstance(item['correct'], bool):
            item['correct'] = item['correct']
        elif isinstance(item['correct'], str):
            item['correct'] = item['correct'].lower() == 'true'
        else:
            assert False, "Strange type of correct field"
        # Check correct
        if not item['correct']:
            block_id = int(item["block"].replace("BLOCK-", ""))
            if block_id < len(trace_blocks):
                wrong_block = "\n".join(trace_blocks[block_id])
            explanation = item['explanation']
    if wrong_block is None:
        wrong_block = "\n".join([(f"[BLOCK-{i}]\n" + "\n".join(b)) for i, b in enumerate(trace_blocks)])
    if explanation is None:
        explanation = responses
    return wrong_block, explanation

def print_messages(messages: List[Message], prefix = "") -> None:
    print("::CHAT MESSAGE::" +prefix)
    for msg in messages:
        print(msg.content)
    print("==================")

def parse_debug_response(text):
    text = text.replace('```json', '').replace('```', '')
    text = text.replace('\\', '\\\\').replace("\'correct\'", "\"correct\"").replace("\'explanation\'", "\"explanation\"").replace(" \'", " \"").replace("\'}", "\"}").strip()
    assert text[0] == '{'
    if text[-1] != '}':
        if text[-1] in ["\"", "\'"]:
            text += '}'
        else:
            text += "\"}"
    text = text.replace("\'", "\"")
    text = text.replace("\"", "\\\"")
    text = text.replace("\\\"correct\\\"", "\"correct\"")
    text = text.replace("\\\"explanation\\\"", "\"explanation\"")
    text = text.replace(": \\\"", ": \"")
    text = text.replace("\\\"", "\"")
    text = text.replace("\\\"}", "\"}")
    text = text.replace('False', 'false').replace('True', 'true')
    text = text.replace(": false,", ": \"false\",")
    try:
        json_item = json.loads(text, strict=False)
    except Exception as e:
        return {"correct": False, "explanation": "I don't know why this is wrong but it is incorrect."}
    return json_item

def check_block_correctness(messages, model: ModelBase,failed_test: str, block: str):
    if model.is_chat:
        debug_message = [
            Message(
                role="user",
                content=f"### Debug Start\n## Execution Trace\n{block}\n### Debug Response"
            )
        ]
        messages += debug_message
        max_trials = 10
        trials = 0
        debug_response = None
        while trials < max_trials:
            try:
                print_messages(debug_message)
                debug_response = model.generate_chat(messages=messages, stop=["### Debug End"])
            except Exception as e:
                print("Generate Debug Response Fail:\n" + str(e))
                trials += 1
                time.sleep(5)
                continue
            else:
                break
        if debug_response is None:
            debug_response = "{\"correct\": \"false\", \"explanation\": \"I don't know why this is wrong.\"}"
        debug_response = debug_response.strip()
        print(debug_response+"\n### Debug End")
    else:
        messages += f"### Debug Start\n### Execution Trace\n{block}\n### Debug Response"
        debug_response = model.generate_completion(messages, temperature=0, stop=["### Debug End"])
    parsed_response = parse_debug_response(debug_response)
    # Update messages
    if model.is_chat:
        messages.append(Message(role="assistant", content=debug_response + "\n### Debug End"))
    else:
        messages += debug_response + "### Debug End\n"
    # convert parsed_reponse['correct'] into bool
    if isinstance(parsed_response['correct'], bool):
        is_correct = parsed_response['correct']
    elif isinstance(parsed_response['correct'], str):
        is_correct = parsed_response['correct'].lower() == 'true'
    else:
        assert False, "Strange type of correct field"
    if 'explanation' not in parsed_response:
        parsed_response['explanation'] = "I don't know why this is wrong."
    return is_correct, parsed_response['explanation'], messages

def get_code_body(response):
    if "```" in response:
        return response.split("```python")[1].split("```")[0]
    else:
        return response

class PyGenerator:
    def ldb_debug(self, prompt: str, prev_func_impl: str, failed_test: str, entry: str, model: ModelBase, messages: List[Message], dataset_type: str = "", level: str = "block") -> str:
        prev_func_impl = trim_header(prev_func_impl)
        failed_test_string = failed_test.split("# Real Execution Output:")[0]
        real_test_output = failed_test.split("# Real Execution Output:")[1]
        if model.is_chat:
            if dataset_type in ["TransCoder"]:
                if len(messages) == 0:
                    messages = [
                        Message(
                            role = "system",
                            content = "You are an expert programming assistant.",
                        ),
                        Message(
                            role = "user",
                            content = f"Translate the C++ code into Python code. Please respond with code only (with the code inside a Markdown code block).\n{prompt}"
                        ),
                        Message(
                            role = "assistant",
                            content = f"{prev_func_impl}"
                        )
                    ]
                    print_messages(messages, "213:\n")
                feedback = f"The Python translation does not do the same thing as the C++ code. Help me debug this. \nThese are the results of one failed unit test that tests whether the Python translation’s outputs match the C++ program’s outputs:\n{failed_test}."
            elif dataset_type in ["HumanEval", "MBPP"]:
                if len(messages) == 0:
                    messages = [
                        Message(
                            role = "system",
                            content = "You are an expert programming assistant.",
                        ),
                        Message(
                            role = "user",
                            content = f"Complete the following task in Python. Please respond with code only (with the code inside a Markdown code block).\n{prompt}"
                        ),
                        Message(
                            role = "assistant",
                            content = f"{prev_func_impl}"
                        )
                    ]
                    print_messages(messages, "268:\n")
                feedback = f"The code above fails the given unit test:\n{failed_test}. \nHelp me debug this.\n"
            # Check whether the solution can be executed
            if level == "line":
                trace_blocks = get_code_traces_line(IMPORT_HEADER + prev_func_impl, failed_test.replace("assert ", "").split("==")[0], entry)
            elif level == "function":
                trace_blocks = get_code_traces_function(IMPORT_HEADER + prev_func_impl, failed_test.replace("assert ", "").split("==")[0], entry)
            else:
                trace_blocks = get_code_traces_block(IMPORT_HEADER + prev_func_impl, failed_test.replace("assert ", "").split("==")[0], entry)
            print("Get trace blocks...")
            # CANNOT EXECUTED
            if isinstance(trace_blocks, str):
                if trace_blocks == "*timeout*":
                    print("The program exceeds the time limit!")
                    msg = [Message(role = "user", content = f"Feedback: With the above function, the assertion is `{failed_test_string}` but the real execution output is `{real_test_output}`.")]
                elif trace_blocks.startswith("*execution fail*"):
                    print(trace_blocks.replace("*execution fail*", ""))
                    msg = [Message(role = "user", content = f"Feedback: With the above function, the assertion is `{failed_test_string}` but the real execution output is `{real_test_output}`.")]
                elif trace_blocks.startswith("*parse fail*"):
                    print("The program is weird")
                    msg = [Message(role = "user", content = f"Feedback: With the above function, the assertion is `{failed_test_string}` but the real execution output is `{real_test_output}`.")]
                else:
                    assert False, "Strange type of trace error: " + trace_blocks
                print_messages(msg)
                messages += msg
                return messages
            elif len(trace_blocks) == 0:
                print("No trace blocks found.")
                msg = [Message(role = "user", content = f"Feedback: With the above function, the assertion is `{failed_test_string}` but the real execution output is `{real_test_output}`.")]
                print_messages(msg)
                messages += msg
                return messages
            # Start debugging
            msg = [Message(
                        role = "user",
                        content = feedback + "\nHere is the code execution trace block by block with the intermediate variable values. Please explain the execution FOR EACH BLOCK and answer whether this block is correct or not. If not, give an explanation on what is wrong. Please wrap your response into a JSON object that contains keys `block` with the name of each block, key `correct` with value False or True, and key `explanation` with an explanation on the bug. \nExample Answers:\n{\"block\": \"BLOCK-1\", \"correct\": \"True\", \"explanation\": \"The block initializes variable `a` and `b`.\"}\n{\"block\": \"BLOCK-2\", \"correct\": \"False\", \"explanation\": \"The block is incorrect because the code does not add the two integers together, but instead subtracts the second integer from the first. To fix this issue, we should change the operator from `-` to `+` in the return statement. This will ensure that the function returns the correct output for the given input.\"}"
                )]
            
            if level == "line":
                max_num_blocks = 30
            elif level == "function":
                max_num_blocks = 1
                block_lines = trace_blocks[0]
                print("313:", len(block_lines))
                if len(block_lines) > 30:
                    trace_blocks[0] = block_lines[:15] + ["..."] + block_lines[-15:]
            else:
                max_num_blocks = 10
            if len(trace_blocks) > max_num_blocks:
                print("Sample trace block...")
                selected_blocks = trace_blocks[:int(max_num_blocks/2)] + trace_blocks[-int(max_num_blocks/2):]
                trace_blocks  = selected_blocks
            for i, b in enumerate(trace_blocks):
                b = "\n".join(b)
                b = f"\n[BLOCK-{i}]\n" + b
                msg[0].content += b
            msg[0].content += "\n"
            messages += msg
            print_messages(msg)
            explanation_all = model.generate_chat(messages=messages, num_comps=1, temperature=0, stop=['[debug end]', 'Here is the updated code:'])

            #wrong_block, explanation = parse_explanation(explanation_all, trace_blocks, prev_func_impl)
            msg = [
                Message(
                        role = "assistant",
                        content = explanation_all
                    )
            ]
            print_messages(msg)
            messages += msg
        else:
            if dataset_type in ["TransCoder"]:
                if len(messages) == 0:
                    # Few shot examples
                    messages = f"{PY_CHAINOFDEBUG_TRANSLATION_INSTRUCTION}"
                    print(messages)
                    # Explain C++
                    delta_msg = f"\n[c++]\n{self.get_last_cpp(prompt)}\n[/c++]\n[explanation]"
                    print(delta_msg)
                    messages += delta_msg
                    explanation = model.generate_completion(messages, temperature=0, stop=["[/explanation]"])
                    delta_msg = f"\n{explanation.strip()}\n[/explanation]\n[python]\n{prev_func_impl}\n[/python]"
                    print(delta_msg)
                    messages += delta_msg
                # Fix
                delta_msg = f"\nThe Python translation does not do the same thing as the C++ code. These are the results of one failed unit test that tests whether the Python translation’s outputs match the C++ program’s outputs:\nFailed: {failed_test_string}\nActual Result: {real_test_output}"
            else:
                if len(messages) == 0:
                    messages = f"{PY_CHAINOFDEBUG_TEXT2CODE_INSTRUCTION}\n{failed_test_string}\n\n{prev_func_impl}\n"
                    print(messages)
                else:
                    delta_msg = f"### Task Start ###\n# These are the assertions for your function:\n{failed_test_string}\n\n{prev_func_impl}\n"
                    messages += delta_msg
                    print(delta_msg)
                # Explain Python
                delta_msg = f"\nFeedback: With the above function, the assertion is `{failed_test_string}` but the real execution output is `{real_test_output}`.\n"
            if level == "line":
                trace_blocks = get_code_traces_line(IMPORT_HEADER + prev_func_impl, failed_test.replace("assert ", "").split("==")[0], entry)
            if level == "function":
                trace_blocks = get_code_traces_function(IMPORT_HEADER + prev_func_impl, failed_test.replace("assert ", "").split("==")[0], entry)
            else:
                trace_blocks = get_code_traces_block(IMPORT_HEADER + prev_func_impl, failed_test.replace("assert ", "").split("==")[0], entry)
            print("Get trace blocks...")
            # CANNOT EXECUTED
            if isinstance(trace_blocks, str):
                if trace_blocks == "*timeout*":
                    delta_msg += "\nThe program exceeds the time limit!"
                elif trace_blocks.startswith("*execution fail*"):
                    delta_msg += "\n" + trace_blocks.replace("*execution fail*", "")
                elif trace_blocks.startswith("*parse fail*"):
                    delta_msg += "\nThe program cannot be executed!"
                else:
                    assert False, "Strange type of trace error: " + trace_blocks
                print(delta_msg)
                messages += delta_msg
                return messages
            elif len(trace_blocks) == 0:
                print("No trace blocks found.")
                delta_msg += "\nThe program cannot be executed!"
                messages += delta_msg
                return messages
            delta_msg += "\nDebug the program trace block by block until find the incorrect block. Every block should have different feedbacks:"
            if level == "line":
                max_num_blocks = 30
            elif level == "function":
                max_num_blocks = 1
                block_lines = trace_blocks[0]
                if len(block_lines) > 30:
                    trace_blocks[0] = block_lines[:15] + ["..."] + block_lines[-15:]
            else:
                max_num_blocks = 10
            if len(trace_blocks) > max_num_blocks:
                print("Sample trace block...")
                selected_blocks = trace_blocks[:int(max_num_blocks/2)] + trace_blocks[-int(max_num_blocks/2):]
                trace_blocks  = selected_blocks
            for i, b in enumerate(trace_blocks):
                b = "\n".join(b)
                b = f"\n[BLOCK-{i}]\n" + b
                delta_msg += b
            delta_msg += "\n[debug]"
            messages += delta_msg
            print(delta_msg)
            explanation = model.generate_completion(messages=messages, stop=["[/debug]"], temperature=0)
            delta_msg = "\n" + explanation.strip() + "\n[/debug]"
            messages += delta_msg
            print(delta_msg)
        return messages

    def ldb_generate(
        self,
        func_sig: str,
        model: ModelBase,
        messages: List[Message],
        prev_func_impl: Optional[str] = None,
        failed_tests: Optional[str] = None,
        num_comps: int = 1,
        temperature: float = 0.0,
        dataset_type: str = "",
    ) -> Union[str, List[str]]:
        prev_func_impl = trim_header(prev_func_impl)
        if model.is_chat:
            if dataset_type in ["TransCoder"]:
                msg = [
                    Message(
                            role = "user",
                            content = f"Correct the Python translation."
                        )
                ]
                messages += msg
                print_messages(msg)
                func_bodies = model.generate_chat(messages=messages)
                msg = [
                    Message(
                            role = "assistant",
                            content = func_bodies
                        )
                ]
                messages += msg
                print_messages(msg)
            elif dataset_type in ["HumanEval", "MBPP"]:
                msg = [
                    Message(
                            role = "user",
                            content = f"Please fix the Python code."
                        )
                ]
                messages += msg
                print_messages(msg)
                func_bodies = model.generate_chat(messages=messages)
                msg = [
                    Message(
                            role = "assistant",
                            content = func_bodies
                        )
                ]
                messages += msg
                print_messages(msg)
        else:
            if dataset_type in ["TransCoder"]:
                delta_msg = "\nCorrect the translation.\n[python]"
            else:
                delta_msg = "\nPlease fix the Python code.\n[python]"
            print(delta_msg)
            messages += delta_msg
            func_bodies = model.generate_completion(messages, temperature=0, stop=["[/python]"])
        if num_comps == 1:
            assert isinstance(func_bodies, str)
            func_body_str = get_code_body(func_bodies).strip()
            if isinstance(messages, str):
                if dataset_type in ["TransCoder"]:
                    delta_msg = f"\n{func_body_str}\n[/python]"
                else:
                    delta_msg = f"\n{func_body_str}\n[/python]\n### Task End ###"
                print(delta_msg)
                messages += delta_msg
            else:
                messages.append(Message(role="assistant", content=func_body_str))
            return func_body_str, messages
        else:
            assert False, "Not Implemented!"
            func_bodies = [get_code_body(func_body) for func_body in func_bodies]
            return func_bodies, _
    
    def get_last_cpp(self, prompt):
        return prompt.split("[c++]\n")[-1].replace("\n[python]", "")

    def simple_translation(self, func_sig, model, prev_func_impl, feedback, given_tests, num_comps, temperature):
        assertion_string = "\n".join(given_tests)
        if len(assertion_string) > 3000:
            assertion_string = "\n".join(given_tests[:5])
        if model.is_chat:
            system_prompt = "You are an expert programming assistant."
            user_prompt = f"Translate the C++ code into Python code. Please respond with code only (with the code inside a Markdown code block). These are the assertions for your function for your reference. Answer with code only:\n{assertion_string}\n{func_sig}"
            print(system_prompt + "\n" + user_prompt)
            messages = [
                Message(
                    role="system",
                    content=system_prompt,
                ),
                Message(
                    role="user",
                    content=user_prompt,
                ),
            ]
            func_bodies = model.generate_chat(messages=messages, num_comps=num_comps, temperature=0)
        else:
            messages = f"Translate the following C++ program into Python\n{func_sig}"
            func_bodies = model.generate_completion(messages, temperature=0, stop=["[c++]", "[/code]"])
        return func_bodies

    def simple_text2code(self, func_sig, model, prev_func_impl, feedback, given_tests, num_comps, temperature):
        if model.is_chat:
            func_sig = func_sig.rstrip('\n')
            user_prompt = f"Complete the following task in Python. Remember to repeat all imports and function header. Here is a unit test:\n{given_tests[0].strip()}\n\n{func_sig}"
            messages = [
                Message(
                    role="system",
                    content=f"You are an expert programming assistant.",
                ),
                Message(
                    role="user",
                    content=user_prompt,
                ),
            ]
            func_bodies = model.generate_chat(messages=messages, num_comps=num_comps, temperature=0)
        else:
            messages = f"# Write Python function to complete the task and pass the assertion tests.\n\n### Task Start ###\n# These are the assertions for your function:\nassert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\n\ndef similar_elements(test_tup1, test_tup2):\n\"\"\" Write a function to find the similar elements from the given two tuple lists. \"\"\"\n    res = tuple(set(test_tup1) & set(test_tup2))\n    return (res)\n### Task End ###\n\n### Task Start ###\n# These are the assertions for your function:\nassert is_not_prime(2) == False\n\nimport math\ndef is_not_prime(n):\n    \"\"\" Write a python function to identify non-prime numbers. \"\"\"\n    result = False\n    for i in range(2,int(math.sqrt(n)) + 1):\n        if n % i == 0:\n            result = True\n    return result\n### Task End ###\n\n### Task Start ###\n# These are the assertions for your function:\nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]\n\nimport heapq as hq\ndef heap_queue_largest(nums,n):\n    \"\"\" Write a function to find the largest integers from a given list of numbers using heap queue algorithm. \"\"\"\n    largest_nums = hq.nlargest(n, nums)\n    return largest_nums\n### Task End ###\n\n### Task Start ###\n# These are the assertions for your function:\n{given_tests[0].strip()}\n\n{func_sig.strip()}"
            print(messages)
            func_bodies = model.generate_completion(messages, temperature=0, stop=["### Task End ###"])
        return func_bodies, messages

    def func_impl(
        self,
        func_sig: str,
        model: ModelBase,
        strategy: str,
        prev_func_impl: Optional[str] = None,
        feedback: Optional[str] = None,
        given_tests: Optional[str] = None,
        num_comps: int = 1,
        temperature: float = 0.0,
        dataset_type: str = "",
        prompt: str = ""
    ) -> Union[str, List[str]]:
        # Validate
        if strategy not in ["simple"]:
            raise ValueError(
                f"Invalid strategy: given `{strategy}` but expected `simple`")
        if model.is_chat:
            if strategy == "simple":
                # Translation Task
                if dataset_type in ["TransCoder"]:
                    func_bodies, messages = self.simple_translation(func_sig, model, prev_func_impl, feedback, given_tests, num_comps, temperature)
                else:
                    func_bodies, messages = self.simple_text2code(func_sig, model, prev_func_impl, feedback, given_tests, num_comps, temperature)
            else:
                assert False, "Not Impl!"
        else:
            if strategy == "simple":
                # Translation Task
                messages = "" # placeholder
                if dataset_type in ["TransCoder"]:
                    func_bodies = self.simple_translation(func_sig, model, prev_func_impl, feedback, given_tests, num_comps, temperature)
                else:
                    func_bodies, messages = self.simple_text2code(func_sig, model, prev_func_impl, feedback, given_tests, num_comps, temperature)
            else:
                assert False, "Not Impl!"
        
        if num_comps == 1:
            assert isinstance(func_bodies, str)
            func_body_str = get_code_body(func_bodies)
            if isinstance(messages, list):
                if strategy == 'simple':
                    messages.append(Message(role="assistant", content=func_bodies))
                else:    
                    messages.append(Message(role="assistant", content=func_body_str))
            elif isinstance(messages, str):
                messages += "\n" + func_body_str
            else:
                assert False, "Not Impl!"
            return func_body_str, messages
        else:
            messages += [Message(role="assistant", content=func_body) for func_body in func_bodies]
            func_bodies = [get_code_body(func_body) for func_body in func_bodies]
            return func_bodies, messages

DUMMY_FUNC_SIG = "def func():"
DUMMY_FUNC_CALL = "func()"

def handle_first_line_indent(func_body: str) -> str:
    if func_body.startswith("    "):
        return func_body
    split = func_body.splitlines()
    return f"    {split[0]}\n" + "\n".join(split[1:])

def handle_entire_body_indent(func_body: str) -> str:
    split = func_body.splitlines()
    res = "\n".join(["    " + line for line in split])
    return res

def fix_turbo_response(func_body: str) -> str:
    return fix_markdown(remove_unindented_signatures(func_body))

def fix_markdown(func_body: str) -> str:
    return re.sub("`{3}", "", func_body)

def remove_unindented_signatures(code: str) -> str:
    regex = r"^def\s+\w+\s*\("

    before_signature = []
    after_signature = []
    signature_found = False

    for line in code.split("\n"):
        if re.match(regex, line):
            signature_found = True
            continue

        if signature_found:
            after_signature.append(line)
        else:
            if not line.startswith("    ") and line.strip():
                line = "    " + line
            before_signature.append(line)

    return "\n".join(before_signature + after_signature)

def py_fix_indentation(func_body: str) -> str:
    func_body = fix_turbo_response(func_body)
    """
    3 cases:
        1. good syntax
        2. first line not good
        3. entire body not good
    """
    def parse_indent_rec(f_body: str, cur_state: int) -> str:
        f_body = fix_markdown(f_body)
        if cur_state > 1:
            return f_body
        code = f'{DUMMY_FUNC_SIG}\n{f_body}\n{DUMMY_FUNC_CALL}'
        try:
            exec(code)
            return f_body
        except (IndentationError, SyntaxError):
            p_func = handle_first_line_indent if cur_state == 0 else handle_entire_body_indent
            return parse_indent_rec(p_func(func_body), cur_state + 1)
        except Exception:
            return f_body
    return parse_indent_rec(func_body, 0)

def py_is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False
