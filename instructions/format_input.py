def format_instruction(entry, mode):
    instruction_text = ""
    if mode == "alpaca":
        instruction_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the requested task. "
            f"\n\n### Instruction:\n{entry['instruction']}"
        )
    elif mode == "phi3":
        instruction_text = (
            f"<|user|>\n{entry['instruction']}"
        )
    return instruction_text


def format_input(entry, mode):
    input_text = ""
    if mode == "alpaca":
        input_text = f"\n\n## Input:\n{entry['input']}" if entry['input'] else ''
    elif mode == "phi3":
        input_text = f"\'{entry['input']}\'" if entry['input'] else ''
    return input_text


def format_prompt(entry, mode="alpaca"):
    return format_instruction(entry) + format_input(entry, mode)
